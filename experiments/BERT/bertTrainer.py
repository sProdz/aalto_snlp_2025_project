import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import TripletMarginLoss
import torch.optim as optim
from transformers import BertTokenizer, BertModel


def load_quotes_with_pandas():
    """
    Load quotes from the SQLite database using pandas.
    Limits the number of quotes to 50 for testing purposes.
    """
    conn = sqlite3.connect("data/quotes/quotes.sqlite")
    df = pd.read_sql_query("SELECT * FROM quotes", conn)
    conn.close()
    return df


def clean_quotes_dataframe(df):
    """
    Clean and preprocess the quotes DataFrame.
      - Remove quotes with Arabic text.
      - Replace unwanted characters.
      - Sort by likes (in descending order).
      - Clean the TAGS column and normalize column names.
    """
    df = df[~df["QUOTE"].str.contains("[\u0600-\u06ff]", na=False)]
    df["QUOTE"] = df["QUOTE"].str.replace("“", "")
    df["QUOTE"] = df["QUOTE"].str.replace("”", "")
    df["QUOTE"] = df["QUOTE"].str.replace(";", ".")

    df = df.sort_values(by="LIKES", ascending=False)

    df["TAGS"] = df["TAGS"].apply(lambda x: np.nan if x == "[]" else x)
    df["TAGS"] = df["TAGS"].apply(lambda x: np.nan if x == "None" else x)

    df.columns = df.columns.str.lower()

    return df


class QuoteTripletDataset(Dataset):
    """
    PyTorch Dataset that generates triplets from quotes data.

    Each triplet consists of:
        - Anchor: a randomly selected quote.
        - Positive: a quote with a similar tag (assumed as the first tag in the tags string).
        - Negative: a quote with a different tag.
    """

    def __init__(self, dataframe, quote_column="quote", tag_column="tags"):
        # Drop rows without tags to allow for pairing
        self.data = dataframe.dropna(subset=[tag_column]).reset_index(drop=True)
        self.quote_column = quote_column
        self.tag_column = tag_column

        # Build a dictionary that maps a (simplified) tag to list of row indices.
        self.tag_to_indices = {}
        for idx, tags in enumerate(self.data[self.tag_column]):
            # Assume tags are comma-separated; we take the first tag as the representative.
            tag = tags.split(",")[0].strip() if isinstance(tags, str) else "unknown"
            if tag not in self.tag_to_indices:
                self.tag_to_indices[tag] = []
            self.tag_to_indices[tag].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the anchor quote and its tag.
        anchor = self.data.iloc[idx][self.quote_column]
        tags = self.data.iloc[idx][self.tag_column]
        anchor_tag = tags.split(",")[0].strip() if isinstance(tags, str) else "unknown"

        # Select a positive example (same tag but not the anchor itself if possible)
        pos_indices = self.tag_to_indices[anchor_tag]
        pos_idx = idx
        while pos_idx == idx and len(pos_indices) > 1:
            pos_idx = np.random.choice(pos_indices)
        positive = self.data.iloc[pos_idx][self.quote_column]

        # Select a negative example (from a different tag)
        negative_tag = anchor_tag
        while negative_tag == anchor_tag:
            negative_tag = np.random.choice(list(self.tag_to_indices.keys()))
        negative_idx = np.random.choice(self.tag_to_indices[negative_tag])
        negative = self.data.iloc[negative_idx][self.quote_column]

        return anchor, positive, negative


def train_fine_tune_bert(
    quotes_df, num_epochs=3, batch_size=8, learning_rate=2e-5, max_length=128
):
    """
    Fine-tune the pre-trained BERT model using triplet loss.

    Args:
        quotes_df (DataFrame): DataFrame containing quotes and their tags.
        num_epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        learning_rate (float): Learning rate for optimizer.
        max_length (int): Maximum tokenized sequence length.

    Returns:
        model (BertModel): The fine-tuned BERT model.
        tokenizer (BertTokenizer): The tokenizer corresponding to the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and BERT model from Hugging Face
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.train()
    model.to(device)

    # Define the triplet loss function and the optimizer
    loss_fn = TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dataset and dataloader for triplet training
    dataset = QuoteTripletDataset(quotes_df, quote_column="quote", tag_column="tags")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Fine-tuning loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            anchors, positives, negatives = batch  # Each is a list of quote strings

            # Tokenize each list of quotes
            encoded_anchor = tokenizer(
                list(anchors),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            encoded_positive = tokenizer(
                list(positives),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            encoded_negative = tokenizer(
                list(negatives),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )

            # Move tokenized inputs to the correct device
            encoded_anchor = {k: v.to(device) for k, v in encoded_anchor.items()}
            encoded_positive = {k: v.to(device) for k, v in encoded_positive.items()}
            encoded_negative = {k: v.to(device) for k, v in encoded_negative.items()}

            # Forward pass through BERT for each component of the triplet
            output_anchor = model(**encoded_anchor)
            output_positive = model(**encoded_positive)
            output_negative = model(**encoded_negative)

            # Extract the [CLS] token embeddings as the representation of the sequence
            emb_anchor = output_anchor.last_hidden_state[
                :, 0, :
            ]  # (batch_size, hidden_dim)
            emb_positive = output_positive.last_hidden_state[:, 0, :]
            emb_negative = output_negative.last_hidden_state[:, 0, :]

            # Compute the triplet loss
            loss = loss_fn(emb_anchor, emb_positive, emb_negative)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # Set the model to evaluation mode after training
    model.eval()
    return model, tokenizer


def get_finetuned_embedding(
    model, tokenizer, text, max_length=128, device=torch.device("cpu")
):
    """
    Generate an embedding vector for the given text using the fine-tuned BERT.

    Args:
        model (BertModel): The fine-tuned BERT model.
        tokenizer (BertTokenizer): Corresponding tokenizer.
        text (str): The input quote.
        max_length (int): Maximum tokenized sequence length.
        device (torch.device): Device used for computation.

    Returns:
        numpy.ndarray: The embedding vector from the [CLS] token.
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
    # Extract the [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding


def main():
    # Load and clean quotes data from SQLite
    quotes_df = load_quotes_with_pandas()
    quotes_df = clean_quotes_dataframe(quotes_df)

    # Display a preview of the cleaned DataFrame for verification
    print("Sample Data:")
    print(quotes_df.head())

    # Fine-tune the BERT model on the quotes dataset
    print("\nFine-tuning BERT on quotes dataset...")
    model, tokenizer = train_fine_tune_bert(
        quotes_df, num_epochs=15, batch_size=8, learning_rate=2e-5, max_length=128
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Example usage: Get the embedding for a sample quote
    sample_quote = quotes_df.iloc[0]["quote"]
    embedding = get_finetuned_embedding(
        model, tokenizer, sample_quote, max_length=128, device=device
    )
    # Save the fine-tuned model and tokenizer
    output_model_dir = "fine_tuned_model"
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"\nFine-tuned model and tokenizer saved to '{output_model_dir}'.")

    print("\nSample Quote:")
    print(sample_quote)
    print("\nEmbedding Vector Shape:")
    print(embedding.shape)


if __name__ == "__main__":
    main()
