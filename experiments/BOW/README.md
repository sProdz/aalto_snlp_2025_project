# Goodreads Quote Clustering Pipeline & Query Engine

This project processes a dataset of Goodreads quotes, performs text cleaning, and applies Bag-of-Words (BoW) and TF-IDF based K-Means clustering to group similar quotes. It also generates t-SNE embeddings for visualization.

## Pipeline Overview

The main processing logic is in `main.py`. It reads configuration from `experiment.yaml` and performs the following stages:

1.  **Configuration Loading:** Reads parameters like file paths, number of clusters, SVD components, etc., from `experiment.yaml`.
2.  **Preprocessing:** Loads raw data (defined in `raw_data_file` in the config), cleans the quote text (removing punctuation, stopwords, lemmatizing using `TextPreprocessor` from `bow.py`), and saves the preprocessed data and the preprocessor object.
3.  **BoW Clustering (Optional):** If `bow_results_suffix` is defined in the config, creates a Bag-of-Words representation (`BoWClusterer` from `bow.py`), applies K-Means clustering, reduces dimensions using SVD+t-SNE (`reduce_and_visualize` from `bow.py`), and saves intermediate results.
4.  **TF-IDF Clustering:** Creates a TF-IDF representation (`TfidfClusterer` from `bow.py`), applies K-Means clustering, reduces dimensions using SVD+t-SNE, saves the fitted TF-IDF vectorizer, and saves the final combined results (including BoW results if generated).
5.  **Analysis & Visualization:** The script performs basic analysis by plotting the TF-IDF t-SNE results (if generated) using matplotlib/seaborn and sampling quotes from each TF-IDF cluster.

## Setup and Execution

1.  **Environment:** Ensure you have a Python environment with the necessary libraries installed (e.g., pandas, scikit-learn, matplotlib, seaborn, pyarrow, joblib, pyyaml). Refer to the project's main requirements file if available.
2.  **Configuration:**
    - Edit `experiment.yaml` to specify:
      - `raw_data_file`: Path to the input CSV file (must contain a 'quote' column).
      - `intermediate_dir`: Directory to save processed data and model objects.
      - `preprocessed_file_suffix`: Suffix for the preprocessed data file.
      - `bow_results_suffix`: (Optional) Suffix for the BoW intermediate results. If commented out or removed, the BoW stage is skipped.
      - `tfidf_results_suffix`: Suffix/path for the final results file (containing TF-IDF, t-SNE, and optionally BoW data).
      - `n_clusters`: Number of clusters for K-Means.
      - `n_svd_components`: Number of components for SVD dimensionality reduction before t-SNE.
      - `vectorizer_min_df`, `vectorizer_max_df`: Parameters for CountVectorizer/TfidfVectorizer.
      - `n_samples_per_cluster`: Number of sample quotes to print per cluster during analysis.
      - `random_state`: Seed for reproducibility.
3.  **Run the Pipeline:** Execute the main script from the root directory:
    ```bash
    python experiments/BOW/main.py
    ```
    The script will print progress messages and save intermediate/final files to the specified `intermediate_dir`.

## Outputs

The script generates the following key outputs in the `intermediate_dir` (or as specified in `experiment.yaml`):

- **Preprocessed Data:** e.g., `data_preprocessed.parquet` (contains original data + `processed_quote` column).
- **Text Preprocessor:** `text_preprocessor.joblib` (saved `TextPreprocessor` object).
- **BoW Results (Optional):** e.g., `data_bow_tsne.parquet` (if `bow_results_suffix` is configured).
- **TF-IDF Vectorizer:** `tfidf_vectorizer.joblib` (saved fitted `TfidfVectorizer` object).
- **Final Results:** e.g., `data_tfidf_tsne.parquet` (defined by `tfidf_results_suffix`). This is the primary output containing original data, processed text, cluster assignments, and t-SNE coordinates.

## Key Output for Query Engine

The primary output file relevant for building a query engine is the final results file specified by `tfidf_results_suffix` in `experiment.yaml` (e.g., `intermediate_data/data_tfidf_tsne.parquet`).

This Parquet file contains the processed data along with cluster assignments and embeddings. Key columns include:

- `quote`: The original quote text.
- `processed_quote`: The cleaned and processed version of the quote used for vectorization and clustering.
- `bow_cluster` (Optional): The cluster ID assigned by K-Means based on BoW.
- `tfidf_cluster`: The cluster ID assigned by K-Means based on TF-IDF. **(Likely the most useful for semantic querying)**.
- `tsne_bow_1`, `tsne_bow_2` (Optional): 2D t-SNE coordinates based on BoW.
- `tsne_tfidf_1`, `tsne_tfidf_2`: 2D t-SNE coordinates based on TF-IDF.
- _(Other original columns from the input CSV might also be present)_

## Embed a New Query

To find quotes similar to a new query or to determine its potential cluster, you need to transform the query using the saved preprocessor and TF-IDF vectorizer:

1.  Load the preprocessor: `preprocessor = joblib.load('intermediate_data/text_preprocessor.joblib')`
2.  Load the vectorizer: `vectorizer = joblib.load('intermediate_data/tfidf_vectorizer.joblib')`
3.  Preprocess the raw query text: `processed_query = preprocessor.transform([raw_query])[0]`
4.  Vectorize the processed query: `query_vector = vectorizer.transform([processed_query])`

The resulting `query_vector` (a sparse matrix) represents the query in the same TF-IDF space as the original quotes.

## Building a Query Engine (Suggestions)

The cluster assignments provide a way to group semantically similar quotes. Here are some ideas for leveraging the `tfidf_cluster` column:

1.  **Keyword Search within Clusters:**

    - Allow users to search for keywords.
    - Instead of searching the entire dataset, first identify which clusters contain quotes relevant to the keywords (e.g., by searching the `processed_quote` text within each cluster or using the query vector).
    - Return results primarily from the most relevant cluster(s).

2.  **Find Similar Quotes:**

    - Given a specific quote ID or text, find its `tfidf_cluster`.
    - Retrieve other quotes belonging to the _same_ `tfidf_cluster`.
    - Alternatively, calculate cosine similarity between the `query_vector` and the TF-IDF vectors of all documents (requires loading/calculating the document matrix).

3.  **Cluster Exploration:**
    - Allow users to browse quotes belonging to a specific `tfidf_cluster` ID.
    - Provide representative quotes or keywords for each cluster.

**Implementation Notes:**

- Use a library like `pandas` or `pyarrow` to efficiently read the Parquet file(s).
- Consider indexing the `tfidf_cluster` column for faster lookups.
- The `processed_quote` column is useful for text matching after cleaning.
