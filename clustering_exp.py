import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import time
import multiprocessing
# Load IMDb dataset (make sure it's preprocessed)
file_path = "datasets/imdb_reviews_cleaned.csv"  # Update the file path if needed
df = pd.read_csv(file_path)

# Tokenize reviews
nltk.download('punkt')
df['tokenized_review'] = df['cleaned_review'].apply(lambda x: word_tokenize(str(x).lower()))

# Display sample data
print(df[['tokenized_review', 'sentiment']].head())


# Train Word2Vec model
word2vec_model = Word2Vec(sentences=df['tokenized_review'], vector_size=100, window=5, min_count=15, workers=4)

# Save model for later use
word2vec_model.save("word2vec_imdb.model")

# Print most similar words to "great"
print(word2vec_model.wv.most_similar("great", topn=10))

# Check words most similar to "terrible"
print(word2vec_model.wv.most_similar("terrible", topn=10))

# Load trained Word2Vec model
w2v_model = Word2Vec.load("word2vec_imdb.model")

# Extract word vectors for clustering
word_vectors = Word2Vec.load("word2vec_imdb.model").wv


# Apply k-Means with precomputed cosine distances
kmeans = KMeans(n_clusters=10, random_state=42, n_init=20)
kmeans.fit(word_vectors.vectors)

# # Assign clusters to words
# word_clusters = {word: kmeans.labels_[i] for i, word in enumerate(word_list)}

# Print sample words per cluster
for cluster in range(10):
    print(word_vectors.similar_by_vector(kmeans.cluster_centers_[cluster], topn=10, restrict_vocab=None))

# # Load IMDb dataset (make sure it's preprocessed)
# file_path = "datasets/imdb_reviews_cleaned.csv"  # Update the file path if needed
# df = pd.read_csv(file_path)
#
# # Tokenize reviews
# nltk.download('punkt')
# df['tokenized_review'] = df['cleaned_review'].apply(lambda x: word_tokenize(str(x).lower()))
#
# # Display sample data
# print(df[['tokenized_review', 'sentiment']].head())
#
#
#
# # Train Word2Vec model
# word2vec_model = Word2Vec(
#     sentences=df['tokenized_review'],  # Use cleaned corpus
#     vector_size=200,  # Reduce embedding size to focus on key features
#     window=5,  # Larger context window to understand word relationships
#     min_count=10,  # Ignore words that appear < 5 times (removes rare noisy words)
#     sample=1e-3,  # Subsampling for frequent words (reduces noise)
#     sg=1,  # Use Skip-gram (better for rare words)
#     negative=10,  # Negative sampling to improve word similarity
#     workers=multiprocessing.cpu_count() - 1,  # Use all available CPU cores
#     epochs=20  # More epochs to stabilize training
# )
#
#
# # Save model for later use
# word2vec_model.save("word2vec_imdb.model")
#
# word_vectors = Word2Vec.load("word2vec_imdb.model").wv
# model = KMeans(n_clusters=3, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors)
# positive_cluster_center = model.cluster_centers_[0]
# negative_cluster_center = model.cluster_centers_[1]
# print(word_vectors.similar_by_vector(model.cluster_centers_[0], topn=10, restrict_vocab=None))
# print(word_vectors.similar_by_vector(model.cluster_centers_[1], topn=10, restrict_vocab=None))
# print(word_vectors.similar_by_vector(model.cluster_centers_[2], topn=10, restrict_vocab=None))
#
#
#
# print(word2vec_model.wv.most_similar("great", topn=10))
#
# # Check words most similar to "terrible"
# print(word2vec_model.wv.most_similar("terrible", topn=10))