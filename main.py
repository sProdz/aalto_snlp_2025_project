#!/usr/bin/env python3
import os
import argparse
import joblib
import numpy as np
import logging
from qdrant_client import QdrantClient
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch


# Configure logging for setup only
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# Embedding loaders and helper functions
def load_bow_vectorizer(path="data/BOW/tfidf_vectorizer.joblib"):
    logger.info("Loading TF-IDF vectorizer from '%s'", path)
    return joblib.load(path)


def load_w2v_model(path="data/word2vec/model/word2vec.model"):
    logger.info("Loading Word2Vec model from '%s'", path)
    return Word2Vec.load(path)


def load_bert_model(path="data/BERT/models/finetuned"):
    logger.info("Loading BERT model and tokenizer from '%s'", path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    model.eval()
    return tokenizer, model


def preprocess_w2v(text, stop_words):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens


def embed_bow(vectorizer, text):
    vec = vectorizer.transform([text])
    return vec.toarray()[0].tolist()


def embed_w2v(w2v_model, stop_words, text):
    tokens = preprocess_w2v(text, stop_words)
    valid = [t for t in tokens if t in w2v_model.wv]
    if not valid:
        logger.warning("No valid tokens found for Word2Vec; returning zero vector.")
        return np.zeros(w2v_model.vector_size).tolist()
    emb = np.mean([w2v_model.wv[t] for t in valid], axis=0)
    return emb.tolist()


def embed_bert(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**inputs)
    reps = out.last_hidden_state.squeeze(0)
    emb = reps.mean(dim=0).cpu().tolist()
    return emb


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Qdrant quote search (BOW/W2V/BERT)."
    )
    parser.add_argument(
        "--collections", "-c", required=True, help="Comma-separated from BERT,W2V,BOW"
    )
    parser.add_argument(
        "--top_k", "-k", type=int, default=5, help="Number of top hits per collection"
    )
    parser.add_argument(
        "--qdrant_url",
        "-u",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant HTTP URL",
    )
    args = parser.parse_args()

    name_map = {
        "BERT": "bert_embeddings",
        "W2V": "w2v_embeddings",
        "BOW": "bow_embeddings",
    }
    try:
        selected = [name_map[c.strip().upper()] for c in args.collections.split(",")]
    except KeyError:
        logger.error("--collections must be a comma-separated list of BERT,W2V,BOW")
        return

    logger.info("Connecting to Qdrant at %s", args.qdrant_url)
    client = QdrantClient(url=args.qdrant_url)

    # Suppress HTTP logs from underlying libraries during search
    logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Load embedders
    if "bow_embeddings" in selected:
        bow_vectorizer = load_bow_vectorizer()
    if "w2v_embeddings" in selected:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
        w2v_model = load_w2v_model()
    if "bert_embeddings" in selected:
        tokenizer, bert_model = load_bert_model()

    logger.info(
        "Setup complete. Ready to query collections: %s; top_k=%d", selected, args.top_k
    )
    print("Type your query and press Enter. Type 'exit' to quit.\n")

    try:
        while True:
            query = input(">> ").strip()
            if not query or query.lower() == "exit":
                print("Goodbye!")
                break

            # Embed query
            embeds = {}
            if "bow_embeddings" in selected:
                embeds["bow_embeddings"] = embed_bow(bow_vectorizer, query)
            if "w2v_embeddings" in selected:
                embeds["w2v_embeddings"] = embed_w2v(w2v_model, stop_words, query)
            if "bert_embeddings" in selected:
                embeds["bert_embeddings"] = embed_bert(tokenizer, bert_model, query)

            # Collect all search results first
            results = {}
            for coll in selected:
                results[coll] = client.search(
                    collection_name=coll,
                    query_vector=embeds[coll],
                    limit=args.top_k,
                    with_payload=True,
                )

            # Print results after collection
            for coll in selected:
                print(f"--- Top {args.top_k} in '{coll}' ---")
                hits = results.get(coll, [])
                if not hits:
                    print("  (no results)")
                else:
                    for rank, pt in enumerate(hits, start=1):
                        score = pt.score
                        quote = pt.payload.get("quote", "<no-quote-field>")
                        print(f"{rank:2d}. [{score:.4f}] {quote}")
                print()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")


if __name__ == "__main__":
    main()
