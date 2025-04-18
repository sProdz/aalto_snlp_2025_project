# Quote Embeddings and BERT Fine-Tuning

nopee selitys mitä kaikki kolme scriptia tekee

## 1. Embedding Visualization with t-SNE and K-Means

**Purpose:**  
- **Load and Process Embeddings:** Reads in a CSV file containing BERT-based embeddings (stored as strings), converts them back into numerical arrays, and stacks them into a matrix.
- **Dimensionality Reduction:** Uses PCA to reduce the initial high-dimensional embeddings to a more manageable size and then applies t‑SNE to further reduce them to 2D for visualization.
- **Clustering:** Performs K-Means clustering on the embeddings to group similar quotes.
- **Visualization:** Creates a scatter plot with t‑SNE coordinates, color-coding each point by its cluster assignment so that you can visually inspect the grouping.

You can edit the amount of clusters, number of rows read from the csv and more from the bert_visualization_settings.yaml file.
run the visualizer with 
```python visualize_bert_clusters.py```

OBS! Kestää ikuisuus jos lataa koko csv tiedoston.

## 2. BERT Embedding Computation and Quote Recommendation

**Purpose:**  
- **Data Loading and Cleaning:** Loads quotes from a SQLite database and cleans the data (removes unwanted characters, filters based on content, and normalizes column names).
- **Embedding Computation:** Initializes a BERT tokenizer and model (e.g., a fine-tuned version) and computes embeddings for each quote in batches.
- **Saving and Retrieval:** Optionally saves the computed embeddings to a CSV file.
- **Recommendation Functionality:** Implements cosine similarity-based quote recommendation by comparing a user’s query to the computed embeddings, returning the most semantically similar quotes.

OBS! Kestää ikuisuus jos lataa ja tekee embeddaukset koko csv tiedostolle

## 3. Fine-Tuning BERT Using Triplet Loss

**Purpose:**  
- **Data Preparation:** Loads the full quotes dataset from a SQLite database and cleans it by removing unwanted characters and normalizing data.
- **Triplet Dataset:** Constructs a custom PyTorch `Dataset` that generates triplets (anchor, positive, negative) from the quotes based on their tags, ensuring that quotes with similar tags are grouped together.
- **Model Fine-Tuning:** Fine-tunes a pre-trained BERT model using the triplet loss function to encourage the model to learn an embedding space where similar quotes are closer together.
- **Model Saving:** After fine-tuning, saves the updated model and tokenizer for future use and demonstrates generating an embedding for a sample quote.


