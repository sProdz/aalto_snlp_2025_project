import pandas as pd
import numpy as np
import os
import yaml
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Import the TextPreprocessor class definition
# Make sure bow.py is in the same directory or accessible in the Python path
try:
    from bow import TextPreprocessor
except ImportError:
    print("Error: Could not import TextPreprocessor from bow.py.")
    print("Ensure bow.py is in the same directory or in the Python path.")
    exit()

# --- Configuration Loading ---
CONFIG_FILE = 'experiment.yaml' # Assume config is in the same directory

def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return None

def load_artifacts(config):
    """Loads the saved preprocessor, vectorizer, and data."""
    if not config:
        return None, None, None

    intermediate_dir = config.get('intermediate_dir', 'intermediate_data')
    preprocessor_file = os.path.join(intermediate_dir, 'text_preprocessor.joblib')
    vectorizer_file = os.path.join(intermediate_dir, 'tfidf_vectorizer.joblib')
    # Construct the final data file path using config suffixes
    final_data_file = os.path.join(intermediate_dir, f"data{config.get('tfidf_results_suffix', '_tfidf_tsne.parquet')}")

    try:
        print(f"Loading preprocessor from {preprocessor_file}...")
        preprocessor = joblib.load(preprocessor_file)
        print("Loaded.")
    except FileNotFoundError:
        print(f"ERROR: Preprocessor file not found at {preprocessor_file}")
        return None, None, None
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        return None, None, None

    try:
        print(f"Loading TF-IDF vectorizer from {vectorizer_file}...")
        vectorizer = joblib.load(vectorizer_file)
        print("Loaded.")
    except FileNotFoundError:
        print(f"ERROR: Vectorizer file not found at {vectorizer_file}")
        return None, None, None
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return None, None, None

    try:
        print(f"Loading data from {final_data_file}...")
        data_df = pd.read_parquet(final_data_file)
        # Ensure necessary columns exist
        if 'quote' not in data_df.columns or 'processed_quote' not in data_df.columns:
            print("ERROR: Loaded data must contain 'quote' and 'processed_quote' columns.")
            return None, None, None
        print(f"Loaded data with shape: {data_df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {final_data_file}")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

    return preprocessor, vectorizer, data_df


def find_similar_quotes(raw_query, preprocessor, vectorizer, data_df, data_tfidf_matrix, top_n=5):
    """
    Finds quotes in the dataset most similar to the input query.

    Args:
        raw_query (str): The user's input query.
        preprocessor: The loaded TextPreprocessor object.
        vectorizer: The loaded TfidfVectorizer object.
        data_df (pd.DataFrame): DataFrame containing 'quote' and 'processed_quote'.
        data_tfidf_matrix (sparse matrix): Precomputed TF-IDF matrix for the dataset.
        top_n (int): Number of similar quotes to return.

    Returns:
        pd.DataFrame: DataFrame containing the top_n similar quotes and their similarity scores.
                      Returns None if an error occurs.
    """
    if not raw_query:
        print("Query cannot be empty.")
        return None

    try:
        # Preprocess the query (handle list input for the preprocessor)
        processed_query_list = preprocessor.transform([raw_query])
        if not processed_query_list:
             print("Preprocessing failed.")
             return None
        processed_query = processed_query_list[0]
        print(f"Processed query: '{processed_query}'")

        # Transform the query using the loaded vectorizer
        query_vector = vectorizer.transform([processed_query])
        print(f"Query vector shape: {query_vector.shape}")

        # Calculate cosine similarities
        # data_tfidf_matrix should have shape (n_samples, n_features)
        # query_vector has shape (1, n_features)
        print(f"Dataset TF-IDF matrix shape: {data_tfidf_matrix.shape}")
        similarities = cosine_similarity(query_vector, data_tfidf_matrix)

        # Get the scores as a flat array
        cosine_scores = similarities[0]

        # Get the indices of the top_n scores
        # Use argpartition for efficiency if top_n is small compared to dataset size
        # For simplicity and smaller datasets, argsort is fine.
        top_indices = np.argsort(cosine_scores)[::-1][:top_n]

        # Get the results
        results = data_df.iloc[top_indices].copy()
        results['similarity'] = cosine_scores[top_indices]

        return results[['quote', 'similarity']]

    except Exception as e:
        print(f"An error occurred during similarity search: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None

# --- Main Execution ---
if __name__ == "__main__":
    config = load_config(CONFIG_FILE)
    if config:
        preprocessor, vectorizer, data_df = load_artifacts(config)

        if preprocessor and vectorizer is not None and data_df is not None:
            # Precompute the TF-IDF matrix for the dataset *once* upon loading
            # This avoids recomputing it for every query
            try:
                 print("Computing TF-IDF matrix for the dataset...")
                 # Use the 'processed_quote' column which was used for fitting the vectorizer
                 dataset_tfidf_matrix = vectorizer.transform(data_df['processed_quote'])
                 print("Dataset TF-IDF matrix computed.")
            except Exception as e:
                 print(f"Error computing dataset TF-IDF matrix: {e}")
                 dataset_tfidf_matrix = None

            if dataset_tfidf_matrix is not None:
                print("\n--- Quote Similarity Search ---")
                print("Enter your query (or type 'quit' to exit):")

                while True:
                    user_query = input("> ")
                    if user_query.lower() == 'quit':
                        break

                    top_quotes = find_similar_quotes(
                        user_query,
                        preprocessor,
                        vectorizer,
                        data_df,
                        dataset_tfidf_matrix, # Pass the precomputed matrix
                        top_n=config.get('search_top_n', 5) # Get top_n from config or default
                    )

                    if top_quotes is not None:
                        print("\n--- Top Similar Quotes ---")
                        if not top_quotes.empty:
                             # Set display options for better readability
                             pd.set_option('display.max_colwidth', 150)
                             print(top_quotes.to_string(index=False))
                             pd.reset_option('display.max_colwidth') # Reset option
                        else:
                            print("No similar quotes found (this might indicate an issue).")
                        print("-" * 26)
                    else:
                        print("Search failed.")

                print("Exiting search.")