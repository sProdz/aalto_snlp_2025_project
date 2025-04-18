#!/usr/bin/env python
# coding: utf-8

# In[5]:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# --- Load Configuration from YAML ---
# We only need file paths and cluster info from the config
CONFIG_FILE = "experiment.yaml"
try:
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"ERROR: Configuration file '{CONFIG_FILE}' not found.")
    # Define fallback defaults if needed (adjust if necessary)
    config = {
        "intermediate_dir": "intermediate_data",
        "tfidf_results_suffix": "_tfidf_tsne.parquet",
        "n_clusters": 10,
        "random_state": 42,
        "n_samples_per_cluster": 3,
    }
except Exception as e:
    print(f"Error loading configuration from {CONFIG_FILE}: {e}")
    raise

# --- Configuration Values ---
INTERMEDIATE_DIR = config["intermediate_dir"]
TFIDF_RESULTS_FILE = os.path.join(
    INTERMEDIATE_DIR, f"data{config['tfidf_results_suffix']}"
)
N_CLUSTERS = config["n_clusters"]
RANDOM_STATE = config.get("random_state", 42)
N_SAMPLES_PER_CLUSTER = config["n_samples_per_cluster"]

# Display options
pd.set_option("display.max_colwidth", 150)  # Show more quote text
pd.set_option("display.max_rows", 100)

print("--- Configuration Relevant for Visualization ---")
print(f"TF-IDF Results File: {TFIDF_RESULTS_FILE}")
print(f"N Clusters: {N_CLUSTERS}")
print("-" * 45)

# --- Load Processed Data ---
try:
    print(f"Loading processed data from: {TFIDF_RESULTS_FILE}")
    df_final = pd.read_parquet(TFIDF_RESULTS_FILE)
    print(f"Data loaded successfully. Shape: {df_final.shape}")
    # Optional: Display first few rows and info to verify
    # print(df_final.head())
    # print(df_final.info())
except FileNotFoundError:
    print(f"ERROR: Processed data file not found at '{TFIDF_RESULTS_FILE}'.")
    print(
        "Please ensure the main pipeline (e.g., main.py) has run successfully to generate this file."
    )
    df_final = None  # Set to None so subsequent steps are skipped gracefully
except Exception as e:
    print(f"Error loading data from {TFIDF_RESULTS_FILE}: {e}")
    df_final = None  # Set to None on other errors

# In[6]:


# --- Visualize TF-IDF Clusters using t-SNE ---

if (
    df_final is not None
    and "tsne_tfidf_1" in df_final.columns
    and "tsne_tfidf_2" in df_final.columns
):
    print("Generating TF-IDF t-SNE plot...")
    plt.figure(figsize=(14, 12))
    sns.scatterplot(
        x="tsne_tfidf_1",
        y="tsne_tfidf_2",
        hue="tfidf_cluster",
        palette=sns.color_palette("hsv", N_CLUSTERS),  # Use N_CLUSTERS from config
        data=df_final,
        legend="full",
        alpha=0.3,
    )
    plt.title(f"TF-IDF t-SNE Projection (k={N_CLUSTERS})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="TF-IDF Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
    plt.show()
else:
    print("Skipping TF-IDF t-SNE plot: Data not loaded or t-SNE columns missing.")


# In[10]:


# --- Sample Quotes per TF-IDF Cluster ---

if df_final is not None and "tfidf_cluster" in df_final.columns:
    print(f"\n--- Sampled Quotes per TF-IDF Cluster (k={N_CLUSTERS}) ---")
    n_samples_per_cluster = N_SAMPLES_PER_CLUSTER  # Use value from config

    # Ensure cluster IDs are integers if they are not already
    if not pd.api.types.is_integer_dtype(df_final["tfidf_cluster"]):
        # Attempt conversion, handle potential errors if conversion fails
        try:
            df_final["tfidf_cluster"] = df_final["tfidf_cluster"].astype(int)
        except ValueError:
            print(
                "Warning: Could not convert 'tfidf_cluster' column to integer type. Skipping sampling."
            )
            df_final = None  # Prevent further processing

    if df_final is not None:
        # Check if tfidf_cluster column exists after potential removal due to conversion error
        if "tfidf_cluster" in df_final.columns:
            # Sort unique cluster IDs numerically before iterating
            unique_clusters = sorted(df_final["tfidf_cluster"].unique())
            for cluster_id in unique_clusters:
                print(f"\n--- Cluster {cluster_id} Samples (TF-IDF) ---")
                cluster_df = df_final[df_final["tfidf_cluster"] == cluster_id]
                # Ensure we don't try to sample more than available
                n_to_sample = min(n_samples_per_cluster, len(cluster_df))

                if n_to_sample > 0:
                    cluster_samples = cluster_df.sample(
                        n=n_to_sample,
                        random_state=RANDOM_STATE,  # Use random state from config
                    )
                    # Display relevant columns - adjust 'quote' if needed
                    for index, row in cluster_samples.iterrows():
                        # Check if 'quote' column exists
                        if "quote" in row:
                            print(f"  Quote {index}: {row['quote']}")
                        else:
                            print(f"  Quote {index}: ('quote' column missing)")

                else:
                    print("  No quotes found for this cluster.")
                print("-" * 30)
        else:
            print("Skipping quote sampling: 'tfidf_cluster' column processing failed.")

else:
    print("Skipping quote sampling: Data not loaded or 'tfidf_cluster' column missing.")


# In[ ]:


import joblib
from bow import TextPreprocessor  # Need the class definition to load the object

# Define file paths (should match paths used for saving in main.py)
VECTORIZER_FILE = os.path.join(INTERMEDIATE_DIR, "tfidf_vectorizer.joblib")
PREPROCESSOR_FILE = os.path.join(INTERMEDIATE_DIR, "text_preprocessor.joblib")

# --- Load the Preprocessor and Fitted Vectorizer ---
try:
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    print("Loaded preprocessor and TF-IDF vectorizer.")
except FileNotFoundError:
    print(
        f"ERROR: Could not load preprocessor ('{PREPROCESSOR_FILE}') or vectorizer ('{VECTORIZER_FILE}')."
    )
    print(
        "Please ensure the main pipeline (main.py) has been run successfully to generate these files."
    )
    # Set objects to None to prevent errors in the next step
    preprocessor = None
    vectorizer = None

# --- Define and Embed Your Query ---
if preprocessor and vectorizer:
    # Example Query:
    raw_query = "The meaning of life and the universe"
    print(f"\nRaw Query: '{raw_query}'")

    # 1. Preprocess the query using the loaded preprocessor
    #    The preprocessor's transform method likely expects a list/Series
    processed_query = preprocessor.transform([raw_query])[0]
    print(f"Processed Query: '{processed_query}'")

    # 2. Transform the processed query using the loaded vectorizer's `transform` method
    #    IMPORTANT: Use `.transform()`, NOT `.fit_transform()`.
    #    We want to use the existing vocabulary and IDF weights, not re-learn them.
    query_vector = vectorizer.transform([processed_query])

    print(f"\nQuery TF-IDF Vector (Shape: {query_vector.shape}):")
    # This is a sparse matrix, showing (row, column_index) -> value
    print(query_vector)

    # You can convert to a dense array if needed, but it might be large
    # query_vector_dense = query_vector.toarray()
    # print("\nQuery TF-IDF Vector (Dense):")
    # print(query_vector_dense)

else:
    print(
        "\nSkipping query embedding because preprocessor or vectorizer failed to load."
    )
