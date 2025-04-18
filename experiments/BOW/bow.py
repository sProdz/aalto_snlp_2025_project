import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present (optional, can be done elsewhere)
try:
    stopwords.words('english')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import time # For timing operations if needed later


class TextPreprocessor:
    """
    Handles basic text cleaning tasks like lowercasing, punctuation removal,
    and stopword filtering for NLP preprocessing pipelines.
    """
    def __init__(self, remove_punctuation=True):
        """
        Initializes the preprocessor.

        Args:
            remove_punctuation (bool): Flag to indicate whether to remove punctuation.
        """
        self.remove_punctuation = remove_punctuation
        self.punctuation_map_ = str.maketrans('', '', string.punctuation)
        self.nltk_stop_words_ = set(stopwords.words('english'))

    def _preprocess(self, text):
        """Applies the preprocessing steps to a single text string."""
        if not isinstance(text, str):
            # Handle potential non-string inputs gracefully
            return ""

        text = text.lower()

        if self.remove_punctuation:
            text = text.translate(self.punctuation_map_)

        tokens = text.split()
        tokens = [word for word in tokens if word not in self.nltk_stop_words_]
        return ' '.join(tokens)

    def transform(self, text_input):
        """
        Applies preprocessing to a pandas Series of text documents.

        Args:
            text_series (pd.Series): The series containing text documents.

        Returns:
            pd.Series: The series with processed text.
        """
        if isinstance(text_input, pd.Series):
            return text_input.apply(self._preprocess)
        elif isinstance(text_input, list):
            # Process each item in the list
            return [self._preprocess(text) for text in text_input]
        else:
            raise TypeError("Input must be a pandas Series or a list of strings.")


class BoWClusterer:
    """
    Performs clustering on text data using a Bag-of-Words representation
    followed by the KMeans algorithm.
    """
    def __init__(self, n_clusters=5, vectorizer_params=None, kmeans_params=None):
        """
        Sets up the clustering pipeline components.

        Args:
            n_clusters (int): Desired number of clusters for KMeans.
            vectorizer_params (dict, optional): Custom parameters for CountVectorizer.
            kmeans_params (dict, optional): Custom parameters for KMeans (excluding n_clusters).
        """
        self.n_clusters = n_clusters
        # Use defaults if no specific params are provided
        self.vectorizer_params = vectorizer_params if vectorizer_params is not None else {}
        self.kmeans_params = kmeans_params if kmeans_params is not None else {}

        self.vectorizer_ = CountVectorizer(**self.vectorizer_params)

        # Ensure reproducible results if random_state isn't specified
        if 'random_state' not in self.kmeans_params:
            self.kmeans_params['random_state'] = 42
        # Use n_init='auto' for modern scikit-learn compatibility
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, **self.kmeans_params, n_init='auto')

        # Attributes to store results after fitting
        self.labels_ = None
        self.feature_names_ = None
        self.bow_matrix_ = None

    def fit(self, texts):
        """
        Creates the BoW matrix and fits the KMeans model.

        Args:
            texts (pd.Series or list): Input text documents.

        Returns:
            self: The fitted clusterer instance.
        """
        print("Fitting CountVectorizer...")
        self.bow_matrix_ = self.vectorizer_.fit_transform(texts)
        self.feature_names_ = self.vectorizer_.get_feature_names_out()
        print(f"BoW matrix shape: {self.bow_matrix_.shape}")

        print(f"Fitting KMeans with {self.n_clusters} clusters...")
        self.kmeans_.fit(self.bow_matrix_)
        self.labels_ = self.kmeans_.labels_
        print("Clustering complete.")

        return self

    def predict(self, new_texts):
        """Assigns cluster labels to new, unseen text documents."""
        if self.labels_ is None:
            raise ValueError("Model must be fitted before predicting.")

        print("Transforming new texts using fitted CountVectorizer...")
        new_bow_matrix = self.vectorizer_.transform(new_texts)
        print("Predicting clusters using fitted KMeans...")
        predictions = self.kmeans_.predict(new_bow_matrix)
        return predictions

    def get_cluster_centers(self, n_top_words=10):
        """Retrieves the most frequent terms for each cluster centroid."""
        if self.labels_ is None or self.feature_names_ is None:
            raise ValueError("Model must be fitted to get cluster centers.")

        cluster_centers = {}
        # Identify the indices of terms with highest values in cluster centroids
        order_centroids = self.kmeans_.cluster_centers_.argsort()[:, ::-1]

        for i in range(self.n_clusters):
            top_words = [self.feature_names_[ind] for ind in order_centroids[i, :n_top_words]]
            cluster_centers[i] = top_words

        return cluster_centers


class TfidfClusterer:
    """
    Performs clustering on text data using a TF-IDF representation
    followed by the KMeans algorithm.
    """
    def __init__(self, n_clusters=5, vectorizer_params=None, kmeans_params=None):
        """
        Sets up the clustering pipeline components using TF-IDF.

        Args:
            n_clusters (int): Desired number of clusters for KMeans.
            vectorizer_params (dict, optional): Custom parameters for TfidfVectorizer.
                                                Consider adding min_df/max_df.
            kmeans_params (dict, optional): Custom parameters for KMeans (excluding n_clusters).
        """
        self.n_clusters = n_clusters
        self.vectorizer_params = vectorizer_params if vectorizer_params is not None else {}
        self.kmeans_params = kmeans_params if kmeans_params is not None else {}

        # Use TfidfVectorizer for term weighting
        self.vectorizer_ = TfidfVectorizer(**self.vectorizer_params)

        if 'random_state' not in self.kmeans_params:
            self.kmeans_params['random_state'] = 42
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, **self.kmeans_params, n_init='auto')

        self.labels_ = None
        self.feature_names_ = None
        self.tfidf_matrix_ = None

    def fit(self, texts):
        """
        Creates the TF-IDF matrix and fits the KMeans model.

        Args:
            texts (pd.Series or list): Input text documents.

        Returns:
            self: The fitted clusterer instance.
        """
        print("Fitting TfidfVectorizer...")
        self.tfidf_matrix_ = self.vectorizer_.fit_transform(texts)
        self.feature_names_ = self.vectorizer_.get_feature_names_out()
        print(f"TF-IDF matrix shape: {self.tfidf_matrix_.shape}")

        print(f"Fitting KMeans with {self.n_clusters} clusters on TF-IDF matrix...")
        self.kmeans_.fit(self.tfidf_matrix_)
        self.labels_ = self.kmeans_.labels_
        print("Clustering complete.")

        return self

    def predict(self, new_texts):
        """Assigns cluster labels to new texts based on the TF-IDF model."""
        if self.labels_ is None:
            raise ValueError("Model must be fitted before predicting.")

        print("Transforming new texts using fitted TfidfVectorizer...")
        new_tfidf_matrix = self.vectorizer_.transform(new_texts)
        print("Predicting clusters using fitted KMeans...")
        predictions = self.kmeans_.predict(new_tfidf_matrix)
        return predictions

    def get_cluster_centers(self, n_top_words=10):
        """
        Retrieves the terms with the highest TF-IDF weights for each cluster centroid.
        Terms listed represent those most characteristic of a cluster according
        to the TF-IDF weighted space.
        """
        if self.labels_ is None or self.feature_names_ is None:
            raise ValueError("Model must be fitted to get cluster centers.")

        cluster_centers = {}
        order_centroids = self.kmeans_.cluster_centers_.argsort()[:, ::-1]

        for i in range(self.n_clusters):
            top_words = [self.feature_names_[ind] for ind in order_centroids[i, :n_top_words]]
            cluster_centers[i] = top_words

        return cluster_centers


# Helper function for dimensionality reduction and visualization 
def reduce_and_visualize(matrix, labels, title_prefix="Projection", method='tsne_svd',
                         n_svd_components=100, tsne_params=None):
    """
    Reduces the dimensionality of the input matrix using SVD+t-SNE or just SVD
    and returns the 2D embedding.

    Args:
        matrix (sparse matrix): The BoW or TF-IDF matrix.
        labels (array-like): Cluster labels for coloring the plot (used for title only here).
        title_prefix (str): Prefix for plot titles if visualization is added later.
        method (str): 'tsne_svd' or 'svd'.
        n_svd_components (int): Number of components for TruncatedSVD.
        tsne_params (dict, optional): Parameters for TSNE.

    Returns:
        np.ndarray: The 2D embedding. Returns None if reduction fails.
    """
    if method not in ['tsne_svd', 'svd']:
        print("Error: method must be 'tsne_svd' or 'svd'")
        return None

    print(f"\nApplying TruncatedSVD (k={n_svd_components})...")
    svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
    start_svd_time = time.time()
    reduced_matrix = svd.fit_transform(matrix)
    end_svd_time = time.time()
    print(f"SVD reduced matrix shape: {reduced_matrix.shape}")
    print(f"SVD took: {end_svd_time - start_svd_time:.2f} seconds")

    if method == 'svd':
        # If only SVD is requested, return the first 2 components
        if n_svd_components < 2:
            print("Warning: Need at least 2 SVD components for 2D visualization.")
            return reduced_matrix # Return whatever was computed
        return reduced_matrix[:, :2]

    elif method == 'tsne_svd':
        print("\nApplying t-SNE to the SVD-reduced matrix... (This will take a while)")
        default_tsne_params = {
            'n_components': 2, 'perplexity': 30, 'metric': 'euclidean',
            'init': 'random', 'learning_rate': 'auto', 'n_iter': 300,
            'n_jobs': -1, 'random_state': 42, 'verbose': 10
        }
        if tsne_params is not None:
            default_tsne_params.update(tsne_params)

        tsne = TSNE(**default_tsne_params)
        start_tsne_time = time.time()
        embedding = tsne.fit_transform(reduced_matrix)
        end_tsne_time = time.time()
        print(f"t-SNE embedding shape: {embedding.shape}")
        print(f"t-SNE (after SVD) took: {end_tsne_time - start_tsne_time:.2f} seconds")
        return embedding

    return None # Should not be reached
