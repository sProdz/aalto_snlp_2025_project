import os
import os.path
import logging
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import joblib


def parse_embedding(emb_str: str) -> list[float]:
    """Convert a comma-separated string into a list of floats."""
    return [float(x) for x in emb_str.split(",")]


def ingest_bow(
    logger: logging.Logger,
    parquet_path: str,
    client: QdrantClient,
    collection_name: str,
    batch_size: int,
    total_limit: int,
) -> None:
    """Load preprocessed quotes from Parquet, vectorize with TF-IDF, and upsert into Qdrant."""
    vect_path = os.getenv("BOW_VECTORIZER_PATH", "data/BOW/tfidf_vectorizer.joblib")
    if not os.path.exists(vect_path):
        logger.error("BOW vectorizer not found at %s. Skipping BOW ingest.", vect_path)
        return
    try:
        vectorizer = joblib.load(vect_path)
        logger.info("Loaded BOW vectorizer from %s", vect_path)
    except Exception as e:
        logger.error("Failed to load BOW vectorizer: %s. Skipping BOW ingest.", e)
        return

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(
            "Failed to read Parquet %s: %s. Skipping BOW ingest.", parquet_path, e
        )
        return

    if df.empty:
        logger.warning("No data in %s. Skipping BOW ingest.", parquet_path)
        return

    # Transform text into TF-IDF embeddings
    tfidf_matrix = vectorizer.transform(df["processed_quote"])

    pbar = tqdm(total=total_limit, desc=f"BOW → {collection_name}", unit="pts")
    points: list[PointStruct] = []

    for i, idx in enumerate(df.index):

        if i >= total_limit:
            break

        emb = tfidf_matrix[i].toarray().ravel().tolist()
        payload = {"quote": df.at[idx, "quote"]}
        points.append(PointStruct(id=int(idx), vector=emb, payload=payload))
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            pbar.update(len(points))
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points)
        pbar.update(len(points))

    pbar.close()
    logger.info("Finished BOW ingestion.")


def ingest_w2v(
    logger: logging.Logger,
    name: str,
    csv_path: str,
    chunksize: int,
    client: QdrantClient,
    collection_name: str,
    batch_size: int,
    total_limit: int,
) -> None:
    """Stream CSV of Word2Vec embeddings and upsert into Qdrant."""
    try:
        reader = pd.read_csv(csv_path, usecols=["quote", "vector"], chunksize=chunksize)
    except Exception as e:
        logger.error("Cannot read W2V CSV %s: %s. Skipping W2V ingest.", csv_path, e)
        return

    pbar = tqdm(total=total_limit, desc=f"{name} → {collection_name}", unit="rows")
    offset = 0
    ingested = 0
    points: list[PointStruct] = []

    for chunk in reader:
        for local_idx, row in chunk.iterrows():
            global_id = offset + local_idx
            emb = parse_embedding(row["vector"])
            points.append(
                PointStruct(
                    id=int(global_id), vector=emb, payload={"quote": row["quote"]}
                )
            )
            if len(points) >= batch_size:
                client.upsert(collection_name=collection_name, points=points)
                pbar.update(len(points))
                points = []

            ingested += 1

            if ingested >= total_limit:
                break

        if ingested >= total_limit:
            break
        offset += len(chunk)

    if points:
        client.upsert(collection_name=collection_name, points=points)
        pbar.update(len(points))

    pbar.close()
    logger.info("Finished Word2Vec ingestion.")


def ingest_bert(
    logger: logging.Logger,
    name: str,
    csv_path: str,
    chunksize: int,
    client: QdrantClient,
    collection_name: str,
    batch_size: int,
    total_limit: int,
) -> None:
    """Stream CSV of precomputed BERT embeddings and upsert into Qdrant."""
    try:
        reader = pd.read_csv(
            csv_path, usecols=["quote", "embedding"], chunksize=chunksize
        )
    except Exception as e:
        logger.error("Cannot read BERT CSV %s: %s. Skipping BERT ingest.", csv_path, e)
        return

    pbar = tqdm(total=total_limit, desc=f"{name} → {collection_name}", unit="rows")
    offset = 0
    ingested = 0
    points: list[PointStruct] = []

    for chunk in reader:
        for local_idx, row in chunk.iterrows():
            global_id = offset + local_idx
            emb = parse_embedding(row["embedding"])
            points.append(
                PointStruct(
                    id=int(global_id), vector=emb, payload={"quote": row["quote"]}
                )
            )
            if len(points) >= batch_size:
                client.upsert(collection_name=collection_name, points=points)
                pbar.update(len(points))
                points = []

            ingested += 1

            if ingested >= total_limit:
                break

        if ingested >= total_limit:
            break
        offset += len(chunk)

    if points:
        client.upsert(collection_name=collection_name, points=points)
        pbar.update(len(points))

    pbar.close()
    logger.info("Finished BERT ingestion.")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ingest")

    # Environment variables
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    batch_size = int(os.getenv("BATCH_SIZE", "500"))
    csv_chunksize = int(os.getenv("CSV_CHUNKSIZE", batch_size))
    total_limit = int(os.getenv("MAX_POINTS", "345000"))

    paths = {
        "BERT": os.getenv("BERT_CSV", "data/BERT/embeddings/embeddings_tuned_bert.csv"),
        "Word2Vec": os.getenv("W2V_CSV_PATH", "data/word2vec/quotes_with_vectors.csv"),
        "BOW": os.getenv("BOW_PARQUET_PATH", "data/BOW/data_tfidf_tsne.parquet"),
    }
    collections = {
        "BERT": os.getenv("BERT_COLLECTION", "bert_embeddings"),
        "Word2Vec": os.getenv("W2V_COLLECTION", "w2v_embeddings"),
        "BOW": os.getenv("BOW_COLLECTION", "bow_embeddings"),
    }

    client = QdrantClient(url=qdrant_url)
    logger.info("Connected to Qdrant at %s", qdrant_url)

    # Process each source
    for name in ["BERT", "Word2Vec", "BOW"]:
        path = paths[name]
        coll = collections[name]
        log = logging.getLogger(name)

        # Skip missing data files
        if not os.path.exists(path):
            log.warning("Data file not found: %s. Skipping %s.", path, name)
            continue

        # Determine vector dimension
        if name == "BERT":
            try:
                peek = pd.read_csv(path, usecols=["embedding"], nrows=1)
                dim = len(parse_embedding(peek.loc[0, "embedding"]))
            except Exception as e:
                log.error(
                    "Failed to peek embedding in %s: %s. Skipping %s.", path, e, name
                )
                continue

        elif name == "Word2Vec":
            try:
                peek = pd.read_csv(path, usecols=["vector"], nrows=1)
                dim = len(parse_embedding(peek.loc[0, "vector"]))
            except Exception as e:
                log.error(
                    "Failed to peek embedding in %s: %s. Skipping %s.", path, e, name
                )
                continue

        elif name == "BOW":  # BOW
            vect_path = os.getenv(
                "BOW_VECTORIZER_PATH", "data/BOW/tfidf_vectorizer.joblib"
            )
            if not os.path.exists(vect_path):
                log.error("BOW vectorizer not found at %s. Skipping BOW.", vect_path)
                continue
            try:
                vectorizer = joblib.load(vect_path)
                dim = len(getattr(vectorizer, "vocabulary_", []))
            except Exception as e:
                log.error("Unable to load BOW vectorizer: %s. Skipping BOW.", e)
                continue

        # Create or reuse collection
        existing = {c.name for c in client.get_collections().collections}
        if coll not in existing:
            client.recreate_collection(
                collection_name=coll,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info("Created collection '%s' (dim=%d, COSINE).", coll, dim)
        else:
            log.info("Using existing collection '%s'.", coll)

        # Call ingest function
        if name == "BERT":
            ingest_bert(
                log, name, path, csv_chunksize, client, coll, batch_size, total_limit
            )
        elif name == "Word2Vec":
            ingest_w2v(
                log, name, path, csv_chunksize, client, coll, batch_size, total_limit
            )
        else:
            ingest_bow(log, path, client, coll, batch_size, total_limit)

    logger.info("All ingestions complete.")


if __name__ == "__main__":
    main()
