import sqlite3
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


def load_and_clean_quotes(limit=50):
    """
    Load quotes from a SQLite database and perform cleaning:
      - Remove rows with Arabic text.
      - Replace unwanted characters.
      - Sort quotes by number of likes.
      - Clean the TAGS column.
      - Normalize column names to lowercase.
    """
    conn = sqlite3.connect("quotes.sqlite")
    # For testing, use a limit; remove it for full dataset processing.
    df = pd.read_sql_query(f"SELECT * FROM quotes LIMIT {limit} ", conn)
    conn.close()

    # Remove quotes containing Arabic text and unwanted punctuation
    df = df[~df["QUOTE"].str.contains("[\u0600-\u06ff]", na=False)]
    df["QUOTE"] = df["QUOTE"].str.replace("“", "", regex=False)
    df["QUOTE"] = df["QUOTE"].str.replace("”", "", regex=False)
    df["QUOTE"] = df["QUOTE"].str.replace(";", ".", regex=False)
    df = df.sort_values(by="LIKES", ascending=False)
    df["TAGS"] = df["TAGS"].apply(lambda x: np.nan if x in ["[]", "None"] else x)
    df.columns = df.columns.str.lower()

    return df


def init_model_and_tokenizer():
    """
    Initialize and return the BERT tokenizer and model (moved to GPU if available)
    in evaluation mode.
    """
    # if using vanilla, use these
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')

    # if using fine tuned, use this
    model_path = "./fine_tuned_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to GPU if available
    model.eval()
    return tokenizer, model, device


def get_bert_embedding(text, tokenizer, model, device, max_length=128):
    """
    Generate a BERT embedding for the given text using the [CLS] token.
    This function processes a single text string.
    """
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
    # Extract the [CLS] token embedding and move it back to CPU.
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding


def compute_embeddings_batch(
    quotes_df, tokenizer, model, device, max_length=128, batch_size=8
):
    """
    Compute embeddings for all quotes in a batch using the GPU (if available).
    This function processes the quotes in mini-batches to limit GPU memory usage.

    Args:
        quotes_df (DataFrame): Cleaned quotes DataFrame.
        tokenizer (BertTokenizer): Pre-trained BERT tokenizer.
        model (BertModel): Pre-trained BERT model.
        device (torch.device): Device (GPU/CPU) to run computation.
        max_length (int): Maximum sequence length.
        batch_size (int): Number of quotes to process per batch.

    Returns:
        - Updated DataFrame with an 'embedding' column containing NumPy arrays.
        - A NumPy matrix of stacked embeddings.
    """
    embeddings_list = []
    texts = quotes_df["quote"].tolist()

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx : start_idx + batch_size]
        # Tokenize the batch.
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        # Move tensors to the chosen device.
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the [CLS] token embeddings and move them to CPU.
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings_list.extend(batch_embeddings)

        # Optionally clear cache after each batch to free up memory.
        torch.cuda.empty_cache()

    embeddings_array = np.array(embeddings_list)
    quotes_df["embedding"] = list(embeddings_array)
    return quotes_df, embeddings_array


def save_embeddings_to_csv(quotes_df, filename="quotes_embeddings.csv"):
    """
    Save the quotes and their embeddings to a CSV file.

    Each embedding (a NumPy array) is converted into a comma-separated string.
    """
    df_to_save = quotes_df.copy()
    df_to_save["embedding"] = df_to_save["embedding"].apply(
        lambda emb: ",".join(map(str, emb.tolist()))
    )
    df_to_save.to_csv(filename, index=False)
    print(f"Embeddings saved to {filename}")


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
    query_embedding = get_bert_embedding(query, tokenizer, model, device, max_length)
    sim_scores = cosine_similarity([query_embedding], embeddings_matrix)[0]
    top_indices = np.argsort(sim_scores)[-top_n:][::-1]
    recommendations = quotes_df.iloc[top_indices].copy()
    recommendations["similarity"] = sim_scores[top_indices]
    return recommendations


def main():
    # Load and clean quotes.
    quotes_df = load_and_clean_quotes(limit=50)

    # Initialize tokenizer, model, and device.
    tokenizer, model, device = init_model_and_tokenizer()

    # Compute embeddings in mini-batches.
    quotes_df, embeddings_matrix = compute_embeddings_batch(
        quotes_df, tokenizer, model, device, max_length=128, batch_size=8
    )

    # Save the quotes with embeddings to a CSV file.
    # save_embeddings_to_csv(quotes_df, "quotes_embeddings.csv")

    # Optional: Accept a user query and provide recommendations.
    user_query = input("Enter quote: ")
    recommended_quotes = recommend_quotes(
        user_query, quotes_df, embeddings_matrix, tokenizer, model, device
    )
    print("\nTop Recommended Quotes:")
    print(recommended_quotes[["quote", "similarity"]])


if __name__ == "__main__":
    main()
