import sqlite3
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import logging


# Configure logging
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = configure_logging()


def load_and_clean_quotes(limit=50):
    """
    Load quotes from a SQLite database and perform cleaning:
      - Remove rows with Arabic text.
      - Replace unwanted characters.
      - Sort quotes by number of likes.
      - Clean the TAGS column.
      - Normalize column names to lowercase.
    """
    logger.info(f"Loading up to {limit} quotes from database")
    conn = sqlite3.connect("data/quotes/quotes.sqlite")
    try:
        df = pd.read_sql_query(f"SELECT * FROM quotes LIMIT {limit}", conn)
        logger.info(f"Retrieved {len(df)} rows from the database")
    except Exception as e:
        logger.exception("Failed to load quotes from SQLite database")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

    # Data cleaning steps
    logger.info("Filtering out quotes containing Arabic text")
    df = df[~df["QUOTE"].str.contains("[\u0600-\u06ff]", na=False)]
    logger.info("Replacing unwanted characters in quotes")
    df["QUOTE"] = df["QUOTE"].str.replace("“", "", regex=False)
    df["QUOTE"] = df["QUOTE"].str.replace("”", "", regex=False)
    df["QUOTE"] = df["QUOTE"].str.replace(";", ".", regex=False)
    logger.info("Sorting quotes by number of likes")
    df = df.sort_values(by="LIKES", ascending=False)
    logger.info("Cleaning TAGS column")
    df["TAGS"] = df["TAGS"].apply(lambda x: np.nan if x in ["[]", "None"] else x)
    df.columns = df.columns.str.lower()
    logger.info("Finished cleaning quotes DataFrame")
    return df


def init_model_and_tokenizer():
    """
    Initialize and return the BERT tokenizer and model (moved to GPU if available)
    in evaluation mode.
    """
    logger.info("Initializing tokenizer and model from finetuned path")
    model_path = "data/BERT/models/finetuned"
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)
    except Exception as e:
        logger.exception("Failed to load tokenizer or model")
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()
    logger.info("Model set to evaluation mode")
    return tokenizer, model, device


def get_bert_embedding(text, tokenizer, model, device, max_length=128):
    """
    Generate a BERT embedding for the given text using the [CLS] token.
    This function processes a single text string.
    """
    logger.debug(f"Generating BERT embedding for text of length {len(text)}")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    logger.debug("Generated embedding with shape %s", cls_embedding.shape)
    return cls_embedding


def compute_embeddings_batch(
    quotes_df, tokenizer, model, device, max_length=128, batch_size=8
):
    """
    Compute embeddings for all quotes in a batch using the GPU (if available).
    """
    total = len(quotes_df)
    logger.info(f"Computing embeddings for {total} quotes in batches of {batch_size}")
    embeddings_list = []
    texts = quotes_df["quote"].tolist()

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        logger.info(f"Processing batch {start_idx}-{end_idx}")
        batch_texts = texts[start_idx:end_idx]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings_list.extend(batch_embeddings)

        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache")

    embeddings_array = np.array(embeddings_list)
    quotes_df["embedding"] = list(embeddings_array)
    logger.info("Completed computing all embeddings")
    return quotes_df, embeddings_array


def save_embeddings_to_csv(quotes_df, filename="quotes_embeddings.csv"):
    """
    Save the quotes and their embeddings to a CSV file.
    """
    logger.info(f"Saving embeddings to {filename}")
    df_to_save = quotes_df.copy()
    df_to_save["embedding"] = df_to_save["embedding"].apply(
        lambda emb: ",".join(map(str, emb.tolist()))
    )
    try:
        df_to_save.to_csv(filename, index=False)
        logger.info(f"Embeddings successfully saved to {filename}")
    except Exception as e:
        logger.exception("Failed to save embeddings to CSV")
        raise


def recommend_quotes(
    query,
    quotes_df,
    embeddings_matrix,
    tokenizer,
    model,
    device,
    top_n=5,
    max_length=128,
):
    """
    Given a query, compute its embedding and return the top_n most similar quotes.
    """
    logger.info(f"Computing recommendations for query: '{query}'")
    query_embedding = get_bert_embedding(query, tokenizer, model, device, max_length)
    sim_scores = cosine_similarity([query_embedding], embeddings_matrix)[0]
    top_indices = np.argsort(sim_scores)[-top_n:][::-1]
    recommendations = quotes_df.iloc[top_indices].copy()
    recommendations["similarity"] = sim_scores[top_indices]
    logger.info("Top %d recommendations computed", top_n)
    return recommendations


def main():
    logger.info("Starting main execution flow")
    try:
        quotes_df = load_and_clean_quotes(limit=50)
        tokenizer, model, device = init_model_and_tokenizer()
        quotes_df, embeddings_matrix = compute_embeddings_batch(
            quotes_df, tokenizer, model, device, max_length=128, batch_size=8
        )

        user_query = input("Enter quote: ")
        recommended_quotes = recommend_quotes(
            user_query, quotes_df, embeddings_matrix, tokenizer, model, device
        )
        print("\nTop Recommended Quotes:")
        print(recommended_quotes[["quote", "similarity"]])
    except Exception as e:
        logger.exception("An error occurred during execution")


if __name__ == "__main__":
    main()
