import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(config_path):
    logger.info("Starting visualization script")

    # Load config from YAML
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info("Loaded config from %s", config_path)
    except Exception as e:
        logger.exception("Failed to load config file")
        sys.exit(1)

    # Extract config values with defaults
    csv_path = config.get("csv_path", "./embeddings_vanilla_bert.csv")
    nrows = config.get("nrows", 1000)
    n_clusters = config.get("n_clusters", 10)
    pca_components = config.get("pca_components", 50)
    tsne_perplexity = config.get("tsne_perplexity", 15)
    random_state = config.get("random_state", 42)
    logger.debug(
        "Config values – csv_path=%s, nrows=%d, n_clusters=%d, "
        "pca_components=%d, tsne_perplexity=%d, random_state=%d",
        csv_path,
        nrows,
        n_clusters,
        pca_components,
        tsne_perplexity,
        random_state,
    )

    # Load the CSV
    try:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", nrows=nrows)
        logger.info("Read %d rows from %s", len(df), csv_path)
    except Exception as e:
        logger.exception("Error reading CSV")
        sys.exit(1)

    # Convert embedding strings to numpy arrays
    try:
        df["embedding"] = df["embedding"].apply(
            lambda x: np.array(list(map(float, x.split(","))))
        )
        logger.info("Converted embeddings to numpy arrays")
    except Exception as e:
        logger.exception("Failed to parse embeddings")
        sys.exit(1)

    # Stack embeddings
    embeddings = np.stack(df["embedding"].values)
    logger.info("Stacked embeddings into array of shape %s", embeddings.shape)

    # PCA reduction
    logger.info("Performing PCA reduction to %d components", pca_components)
    pca = PCA(n_components=pca_components, random_state=random_state)
    embeddings_pca = pca.fit_transform(embeddings)
    logger.info(
        "PCA explained variance ratio sum: %.4f", pca.explained_variance_ratio_.sum()
    )

    # t-SNE reduction
    logger.info("Running t-SNE (perplexity=%d)", tsne_perplexity)
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=tsne_perplexity,
        init="pca",
    )
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    df["x"], df["y"] = embeddings_2d[:, 0], embeddings_2d[:, 1]
    logger.info("Completed t-SNE transformation")

    # KMeans clustering
    logger.info("Clustering into %d clusters with KMeans", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df["cluster"] = kmeans.fit_predict(embeddings)
    logger.info("KMeans inertia: %.4f", kmeans.inertia_)

    # Plot
    logger.info("Generating scatter plot")
    plt.figure(figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for cluster in range(n_clusters):
        idx = df["cluster"] == cluster
        plt.scatter(
            df.loc[idx, "x"],
            df.loc[idx, "y"],
            label=f"Cluster {cluster}",
            color=colors[cluster % len(colors)],
            s=60,
            alpha=0.7,
        )

    plt.title("t-SNE Visualization of Embeddings Colored by Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    try:
        plt.show()
        logger.info("Plot displayed successfully")
    except Exception:
        logger.exception("Failed to display plot")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python %s <config.yaml>", sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])
