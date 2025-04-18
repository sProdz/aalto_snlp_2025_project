# aalot_snlp_2025_project
Project repository for the Aalto 2025 SNLP course

Data structure:
````
.
├── BERT
│   ├── embeddings
│   │   ├── embeddings_tuned_bert.csv
│   │   └── embeddings_vanilla_bert.csv
│   └── models
│       └── finetuned
│           ├── config.json
│           ├── model.safetensors
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           └── vocab.txt
├── BOW
│   ├── data_bow_tsne.parquet
│   ├── data_preprocessed.parquet
│   ├── data_tfidf_tsne.parquet
│   ├── text_preprocessor.joblib
│   └── tfidf_vectorizer.joblib
├── quotes
│   ├── quotes.csv
│   ├── quotes.sqlite
│   └── tags.csv
└── word2vec
    ├── quotes_with_clusters.csv
    └── quotes_with_vectors.csv