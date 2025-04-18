# Goodreads Quote Clustering Pipeline & Query Engine

This project processes a dataset of Goodreads quotes, performs text cleaning, and applies Bag-of-Words (BoW) and TF-IDF based K-Means clustering to group similar quotes. It also generates t-SNE embeddings for visualization.

## Pipeline Overview

The main processing logic is in `main.py` (converted from `main.ipynb`). It reads configuration from `experiment.yaml` and performs the following stages:

1.  **Preprocessing:** Loads raw data (`quotes.csv`), cleans the quote text (removing punctuation, stopwords, lemmatizing), and saves the result.
2.  **BoW Clustering:** Creates a Bag-of-Words representation, applies K-Means clustering, reduces dimensions using SVD+t-SNE, and saves intermediate results.
3.  **TF-IDF Clustering:** Creates a TF-IDF representation, applies K-Means clustering, reduces dimensions using SVD+t-SNE, and saves the final combined results.
4.  **Analysis:** Includes example visualization and sampling from clusters.

## Visualization and Sanity Checking Results (Artur)

You can use the `visualization.ipynb` notebook to explore the results and perform a sanity check without rerunning the time-consuming steps. Optionally you can run the main pipeline (`main.py`) to run the whole process

1.  **Prerequisites:**

    - Ensure the main pipeline (`python main.py`) has run successfully using the desired configuration in `experiment.yaml`.
    - Confirm that the final output file `intermediate_data/data_tfidf_tsne.parquet` exists.

2.  **Launch Jupyter:** Start Jupyter Lab or Jupyter Notebook from your activated virtual environment:

    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```

3.  **Run the Notebook:**

    - Open `visualization.ipynb`.
    - Run the cells sequentially.

4.  **Expected Output (Sanity Check):**
    - The notebook should load the `data_tfidf_tsne.parquet` file without errors.
    - It will display the shape and column names of the loaded DataFrame.
    - A t-SNE scatter plot visualizing the TF-IDF clusters will be generated. Check if the clusters look reasonably separated (though t-SNE results can vary).
    - Sample quotes from each TF-IDF cluster will be printed. Review these samples to see if quotes within the same cluster seem thematically related.

This notebook helps confirm that the pipeline generated the expected output file with the necessary columns and that the clustering results are loaded correctly.

## Key Output for Query Engine (Artur)

The primary output file relevant for building a query engine is:

- **File:** `intermediate_data/data_tfidf_tsne.parquet`
  _(Note: Ensure the pipeline (`main.py`) has run successfully to generate this file.)_

This Parquet file contains the processed data along with cluster assignments and embeddings. Key columns include:

- `quote`: The original quote text.
- `processed_quote`: The cleaned and processed version of the quote used for vectorization and clustering.
- `bow_cluster`: The cluster ID assigned by the K-Means algorithm based on the BoW representation.
- `tfidf_cluster`: The cluster ID assigned by the K-Means algorithm based on the TF-IDF representation. **(Likely the most useful for semantic querying)**.
- `tsne_bow_1`, `tsne_bow_2`: 2D t-SNE coordinates based on BoW.
- `tsne_tfidf_1`, `tsne_tfidf_2`: 2D t-SNE coordinates based on TF-IDF.
- _(Other original columns from `quotes.csv` might also be present)_

## Embed a New Query

To find quotes similar to a new query or to see where a query would fall in the vector space, we first need to transform the query using the same preprocessing steps and the same fitted TF-IDF vectorizer used on the original data.

## Building a Query Engine (Suggestions)

The cluster assignments provide a way to group semantically similar quotes. Here are some ideas for leveraging the `tfidf_cluster` column:

1.  **Keyword Search within Clusters:**

    - Allow users to search for keywords.
    - Instead of searching the entire dataset, first identify which clusters contain quotes relevant to the keywords (e.g., by checking TF-IDF scores of keywords within cluster centroids, or simply searching the `processed_quote` text within each cluster).
    - Return results primarily from the most relevant cluster(s). This can be much faster than a global search.

2.  **Find Similar Quotes:**

    - Given a specific quote ID or text, find its `tfidf_cluster`.
    - Retrieve other quotes belonging to the _same_ `tfidf_cluster`. These are likely to be semantically similar.

3.  **Cluster Exploration:**

    - Allow users to browse quotes belonging to a specific `tfidf_cluster` ID.
    - Provide representative quotes or keywords for each cluster (e.g., using techniques like finding terms with the highest TF-IDF scores near the cluster centroid).

4.  **Visualization Integration (Optional):**
    - Use the `tsne_tfidf_1` and `tsne_tfidf_2` columns to create a 2D scatter plot of the quotes, colored by `tfidf_cluster`. This can provide a visual way for users to explore the data and identify interesting regions.

**Implementation Notes:**

- Use a library like `pandas` or `pyarrow` to efficiently read the `data_tfidf_tsne.parquet` file.
- Consider indexing the `tfidf_cluster` column for faster lookups if performance is critical.
- The `processed_quote` column is useful for text matching after cleaning.
