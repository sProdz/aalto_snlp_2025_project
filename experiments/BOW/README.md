# Experiment: Goodreads Quote Clustering (BoW & TF-IDF)

## Context within `aalto_snlp_2025_project`

This directory contains the code and configuration for a specific experiment focused on clustering Goodreads quotes using classical NLP vectorization techniques: Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF).

This experiment is part of the larger `aalto_snlp_2025_project`, which explores various methods for quote analysis and retrieval. It sits alongside other experiments in the `experiments/` directory, such as those potentially using `word2vec` or `BERT` embeddings (`experiments/word2vec/`, `experiments/BERT/`). The outputs of this experiment (like processed data, cluster assignments, and trained vectorizers) might be used for comparison with other methods or potentially ingested into downstream systems like the Qdrant vector database (see `qdrant_ingest.py` in the root).

## Experiment Goal

The primary goal of _this_ specific experiment is to:

1.  Preprocess raw quote text.
2.  Generate BoW and TF-IDF vector representations of the quotes.
3.  Apply K-Means clustering to group similar quotes based on these representations.
4.  Reduce dimensionality using SVD and visualize the clusters using t-SNE.
5.  Save the preprocessing steps, trained vectorizers, cluster assignments, and visualizations for analysis and potential reuse.

## Setup and Execution

1.  **Environment:** Ensure you have set up the Python environment for the main `aalto_snlp_2025_project` by navigating to the **repository root** and installing dependencies defined in `pyproject.toml`. This is typically done using `poetry install` or potentially `pip install .` if using pip directly.
2.  **Configuration:**
    - The configuration for this experiment is managed in `experiments/BOW/experiment.yaml`.
    - Edit `experiment.yaml` to specify necessary parameters:
      - `raw_data_file`: Path to the input CSV file (e.g., located in the root `data/` directory). Must contain a 'quote' column. Paths should be specified relative to the **repository root** or as absolute paths.
      - `intermediate_dir`: Directory where all outputs specific to _this experiment_ (processed data, models, results) will be saved. It is recommended to keep this within the experiment's directory (e.g., `experiments/BOW/intermediate_data`). Paths should be relative to the **repository root**.
      - `preprocessed_file_suffix`: File suffix for the saved preprocessed data within `intermediate_dir`.
      - `bow_results_suffix`: (Optional) File suffix for BoW-specific intermediate results. Comment out or remove this key to skip the BoW part of the pipeline.
      - `tfidf_results_suffix`: File suffix for the final results Parquet file (containing TF-IDF clusters, t-SNE coordinates, and optionally BoW results).
      - `n_clusters`, `n_svd_components`: Parameters for K-Means and SVD.
      - `vectorizer_min_df`, `vectorizer_max_df`: Parameters for `CountVectorizer` and `TfidfVectorizer`.
      - `n_samples_per_cluster`: Number of sample quotes to print per cluster during the analysis phase.
      - `random_state`: Seed for reproducibility across runs.
3.  **Run the Experiment:** Execute the main script for this experiment from the **repository root**:
    ```bash
    python experiments/BOW/main.py
    ```
    The script will print progress messages and save all outputs to the configured `intermediate_dir`.

## Outputs

This experiment generates the following key files within the specified `intermediate_dir`:

- **Preprocessed Data:** e.g., `intermediate_data/data_preprocessed.parquet`. Contains original data plus the `processed_quote` column generated by `TextPreprocessor`.
- **Text Preprocessor:** e.g., `intermediate_data/text_preprocessor.joblib`. The saved `TextPreprocessor` object used for cleaning text in this experiment. Essential for consistent processing of new data/queries relative to this experiment's results.
- **BoW Results (Optional):** e.g., `intermediate_data/data_bow_tsne.parquet`. Contains BoW cluster IDs and t-SNE coordinates if `bow_results_suffix` was configured.
- **TF-IDF Vectorizer:** e.g., `intermediate_data/tfidf_vectorizer.joblib`. The saved, fitted `TfidfVectorizer` object. Crucial for transforming new text into the vector space learned in this experiment.
- **Final Results:** e.g., `intermediate_data/data_tfidf_tsne.parquet`. The primary output, containing original data, processed text, `tfidf_cluster` assignments, `tsne_tfidf_*` coordinates, and potentially BoW results (`bow_cluster`, `tsne_bow_*`).

## Using Experiment Outputs

The outputs from this experiment can be used for various purposes:

1.  **Analysis:** Analyze the cluster quality, visualize the t-SNE plots, and examine sample quotes per cluster provided in the output logs and the final results file.
2.  **Comparison:** Compare the clustering results (e.g., silhouette scores, visual separation) with those from other experiments (`word2vec`, `BERT`).
3.  **Downstream Integration:** Use the final results Parquet file and the saved `tfidf_vectorizer.joblib` and `text_preprocessor.joblib` as inputs for other processes. For instance, the cluster IDs could serve as categorical features, or the vectorizer could be used to transform text data consistently according to the TF-IDF space defined by this experiment.
