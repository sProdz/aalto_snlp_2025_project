#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os # To check for file existence
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import csv
import yaml
import sys # Import sys

# Import custom classes from bow.py using relative import
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script's directory to sys.path if it's not already there
# This helps ensure the relative import works reliably
if script_dir not in sys.path:
    sys.path.append(script_dir)
from bow import TextPreprocessor, BoWClusterer, TfidfClusterer, reduce_and_visualize

# In[3]:

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import yaml 

# Import custom classes from bow.py
from bow import TextPreprocessor, BoWClusterer, TfidfClusterer, reduce_and_visualize

# --- Load Configuration from YAML ---
CONFIG_FILE = os.path.join(script_dir, 'experiment.yaml') # <--- Use path relative to script
try:
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"ERROR: Configuration file '{CONFIG_FILE}' not found. Using default fallback values.")
    # Define fallback defaults here if config file is missing 
    config = {
        'raw_data_file': 'quotes.csv',
        'intermediate_dir': 'intermediate_data',
        'preprocessed_file_suffix': '_preprocessed.parquet',
        'bow_results_suffix': '_bow_tsne.parquet',
        'tfidf_results_suffix': '_tfidf_tsne.parquet',
        'n_clusters': 10,
        'n_svd_components': 100,
        'random_state': 42,
        'vectorizer_min_df': 0.001,
        'vectorizer_max_df': 0.95,
        'n_samples_per_cluster': 3
    }
except Exception as e:
    print(f"Error loading configuration from {CONFIG_FILE}: {e}")
    raise # Stop execution if YAML is invalid

# --- Use Configuration Values ---
RAW_DATA_FILE = config['raw_data_file']
INTERMEDIATE_DIR = config['intermediate_dir']

# Construct intermediate file paths dynamically
# Extract base name from raw file path if needed, or define separately in YAML
# For simplicity, assuming we might base intermediate names on the raw file or a specific key
# Let's use the intermediate dir directly:
PREPROCESSED_FILE = os.path.join(INTERMEDIATE_DIR, f"data{config['preprocessed_file_suffix']}")

# Get BoW suffix safely, defaulting to None if commented out
bow_suffix = config.get('bow_results_suffix', None)
BOW_RESULTS_FILE = os.path.join(INTERMEDIATE_DIR, f"data{bow_suffix}") if bow_suffix else None

# Construct TF-IDF path using the dedicated key from YAML
TFIDF_RESULTS_FILE = config['tfidf_results_suffix'] # Path is now absolute from YAML

# Create intermediate directory if it doesn't exist (should be ../../data/BOW_intermediate now)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
# Also create the directory for the final TF-IDF results if it doesn't exist
tfidf_parent_dir = os.path.dirname(TFIDF_RESULTS_FILE)
os.makedirs(tfidf_parent_dir, exist_ok=True)

N_CLUSTERS = config['n_clusters']
N_SVD_COMPONENTS = config['n_svd_components']
RANDOM_STATE = config.get('random_state', 42) # Use .get for optional keys with defaults

# Vectorizer Params (used later)
VECTORIZER_PARAMS = {
    'min_df': config['vectorizer_min_df'],
    'max_df': config['vectorizer_max_df']
    # Add other vectorizer params from config if needed
}

# Analysis Params (used later)
N_SAMPLES_PER_CLUSTER = config['n_samples_per_cluster']


# Set display options (can still be hardcoded or loaded from config if added)
pd.set_option('display.max_colwidth', config.get('display_max_colwidth', 100))
pd.set_option('display.max_rows', config.get('display_max_rows', 100))


print("--- Configuration Loaded ---")
print(f"Raw Data: {RAW_DATA_FILE}")
print(f"Intermediate Dir: {INTERMEDIATE_DIR}")
print(f"N Clusters: {N_CLUSTERS}")
print(f"N SVD Components: {N_SVD_COMPONENTS}")
print(f"Vectorizer Params: {VECTORIZER_PARAMS}")
print("-" * 28)


# In[4]:


# --- Stage 1: Preprocessing ---

if os.path.exists(PREPROCESSED_FILE):
    print(f"Loading preprocessed data from {PREPROCESSED_FILE}...")
    df_processed = pd.read_parquet(PREPROCESSED_FILE)
    print("Loaded.")
else:
    print(f"Preprocessing file {PREPROCESSED_FILE} not found. Running preprocessing...")
    # Load raw data
    try:
        df_raw = pd.read_csv(RAW_DATA_FILE, sep=';', header=0)
        # Basic check for necessary column
        if 'quote' not in df_raw.columns:
             raise ValueError("Raw data must contain a 'quote' column.")
        # Handle potential NaN quotes early
        df_raw.dropna(subset=['quote'], inplace=True)
        df_raw.reset_index(drop=True, inplace=True) # Ensure clean index
        print(f"Loaded raw data with shape: {df_raw.shape}")
    except FileNotFoundError:
        print(f"ERROR: Raw data file '{RAW_DATA_FILE}' not found.")
        raise # Stop execution if raw data is missing

    # Initialize and run preprocessor
    preprocessor = TextPreprocessor()
    df_processed = df_raw.copy() # Work on a copy
    df_processed['processed_quote'] = preprocessor.transform(df_processed['quote'])

    PREPROCESSOR_FILE = os.path.join(INTERMEDIATE_DIR, 'text_preprocessor.joblib')
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print(f"Saved Text Preprocessor to {PREPROCESSOR_FILE}")

    # Save the result
    print(f"Saving preprocessed data to {PREPROCESSED_FILE}...")
    df_processed.to_parquet(PREPROCESSED_FILE, index=False)
    print("Saved.")

print(f"Shape of processed data: {df_processed.shape}")
print(df_processed[['quote', 'processed_quote']].head())


# In[5]:


# --- Stage 2: BoW Clustering and t-SNE ---

df_bow = None # Initialize df_bow to None

if BOW_RESULTS_FILE: # Only run if BoW output is configured
    if os.path.exists(BOW_RESULTS_FILE):
        print(f"Loading BoW results from {BOW_RESULTS_FILE}...")
        df_bow = pd.read_parquet(BOW_RESULTS_FILE)
        print("Loaded.")
        # Recreate necessary objects if needed later (e.g., for sampling cluster centers)
        # bow_clusterer = BoWClusterer(n_clusters=N_CLUSTERS)
        # bow_clusterer.fit(df_bow['processed_quote']) # Need to re-fit to get matrix/centers if not saved
        # Or load saved clusterer object if you implement saving/loading for it
    else:
        print(f"BoW results file {BOW_RESULTS_FILE} not found. Running BoW steps...")
        # Use the DataFrame from the previous stage
        df_bow = df_processed.copy() # Start BoW processing from processed data

        # Initialize and fit BoWClusterer
        print("Initializing and fitting BoWClusterer...")
        bow_clusterer = BoWClusterer(n_clusters=N_CLUSTERS,
                                     vectorizer_params=VECTORIZER_PARAMS)
        bow_clusterer.fit(df_bow['processed_quote'])
        df_bow['bow_cluster'] = bow_clusterer.labels_
        print("BoW clustering complete.")

        # Reduce dimensions
        print("Reducing BoW dimensions using SVD+t-SNE...")
        bow_embedding = reduce_and_visualize(
            matrix=bow_clusterer.bow_matrix_,
            labels=df_bow['bow_cluster'],
            title_prefix="BoW",
            method='tsne_svd',
            n_svd_components=N_SVD_COMPONENTS
        )

        if bow_embedding is not None:
            df_bow['tsne_bow_1'] = bow_embedding[:, 0]
            df_bow['tsne_bow_2'] = bow_embedding[:, 1]
            print("BoW t-SNE coordinates added.")
        else:
            print("Warning: BoW dimensionality reduction failed.")

        # Save the result
        print(f"Saving BoW results (including t-SNE) to {BOW_RESULTS_FILE}...")
        df_bow.to_parquet(BOW_RESULTS_FILE, index=False)
        print("Saved.")

    print(f"Shape after BoW stage: {df_bow.shape}")
    print(df_bow[['processed_quote', 'bow_cluster', 'tsne_bow_1', 'tsne_bow_2']].head())
else:
    print("Skipping BoW stage as 'bow_results_suffix' is not configured.")


# In[ ]:


# --- Stage 3: TF-IDF Clustering and t-SNE ---

if os.path.exists(TFIDF_RESULTS_FILE):
    print(f"Loading final TF-IDF results from {TFIDF_RESULTS_FILE}...")
    df_final = pd.read_parquet(TFIDF_RESULTS_FILE)
    print("Loaded.")
    # Again, optionally recreate/load clusterer object if needed
else:
    print(f"TF-IDF results file {TFIDF_RESULTS_FILE} not found. Running TF-IDF steps...")
    # Load the result from the PREVIOUS completed stage
    # If BoW was skipped, df_bow will be None, so use df_processed
    # Otherwise, use df_bow as it contains the intermediate BoW results
    df_start_tfidf = df_bow.copy() if df_bow is not None else df_processed.copy()
    df_final = df_start_tfidf

    # Initialize and fit TfidfClusterer
    print("Initializing and fitting TfidfClusterer...")
    tfidf_clusterer = TfidfClusterer(n_clusters=N_CLUSTERS,
                                     vectorizer_params=VECTORIZER_PARAMS) # Example params
    # Fit on the same processed text
    tfidf_clusterer.fit(df_final['processed_quote'])
    df_final['tfidf_cluster'] = tfidf_clusterer.labels_
    print("TF-IDF clustering complete.")

    # Reduce dimensions
    print("Reducing TF-IDF dimensions using SVD+t-SNE...")
    tfidf_embedding = reduce_and_visualize(
        matrix=tfidf_clusterer.tfidf_matrix_,
        labels=df_final['tfidf_cluster'],
        title_prefix="TF-IDF",
        method='tsne_svd',
        n_svd_components=N_SVD_COMPONENTS
    )

    VECTORIZER_FILE = os.path.join(INTERMEDIATE_DIR, 'tfidf_vectorizer.joblib')
    if hasattr(tfidf_clusterer, 'vectorizer_'):
        joblib.dump(tfidf_clusterer.vectorizer_, VECTORIZER_FILE)
        print(f"Saved TF-IDF Vectorizer to {VECTORIZER_FILE}")
    else:
        print("Warning: Could not find attribute 'vectorizer_' in tfidf_clusterer to save.")

    if tfidf_embedding is not None:
        df_final['tsne_tfidf_1'] = tfidf_embedding[:, 0]
        df_final['tsne_tfidf_2'] = tfidf_embedding[:, 1]
        print("TF-IDF t-SNE coordinates added.")
    else:
        print("Warning: TF-IDF dimensionality reduction failed.")

    # Save the FINAL result (includes BoW and TF-IDF results)
    print(f"Saving final results (BoW+TF-IDF+t-SNE) to {TFIDF_RESULTS_FILE}...")
    df_final.to_parquet(TFIDF_RESULTS_FILE, index=False)
    print("Saved.")

print(f"Shape after TF-IDF stage: {df_final.shape}")
print("Final DataFrame columns:", df_final.columns.tolist())
# Print head, excluding the BoW columns which might not exist if BoW stage was skipped
print(df_final[['processed_quote', 'tfidf_cluster', 'tsne_tfidf_1', 'tsne_tfidf_2']].head())


# In[ ]:


# --- Stage 4: Analysis and Visualization ---

print("Performing analysis on final data...")

# Example: Plot TF-IDF results
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x="tsne_tfidf_1", y="tsne_tfidf_2",
    hue="tfidf_cluster",
    palette=sns.color_palette("hsv", N_CLUSTERS),
    data=df_final,
    legend="full",
    alpha=0.3
)
plt.title(f'TF-IDF t-SNE Projection (k={N_CLUSTERS})')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# Example: Sample quotes from TF-IDF clusters
print(f"\n--- Sampled Quotes per TF-IDF Cluster (k={N_CLUSTERS}) ---")
n_samples_per_cluster = N_SAMPLES_PER_CLUSTER
for cluster_id in sorted(df_final['tfidf_cluster'].unique()):
    print(f"\n--- Cluster {cluster_id} Samples (TF-IDF) ---")
    cluster_samples = df_final[df_final['tfidf_cluster'] == cluster_id].sample(
        n=min(n_samples_per_cluster, len(df_final[df_final['tfidf_cluster'] == cluster_id])),
        random_state=RANDOM_STATE
    )
    for index, row in cluster_samples.iterrows():
        print(f"  Quote {index}: {row['quote']}")
    print("-" * 30)

print("Analysis done")
# Add other analysis: compare BoW vs TF-IDF clusters, run word searches, etc.



