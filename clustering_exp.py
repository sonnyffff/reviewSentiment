import pandas as pd
import nltk
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import time

start = time.time()
# Load IMDb dataset (make sure it's preprocessed)
file_path = "datasets/imdb_reviews_cleaned.csv"  # Update the file path if needed
df = pd.read_csv(file_path)

# Tokenize reviews
nltk.download('punkt')
df['tokenized_review'] = df['cleaned_review'].apply(lambda x: word_tokenize(str(x).lower()))

# Convert sentiment labels to strings
df['sentiment'] = df['sentiment'].map({1: "Positive", 0: "Negative"})

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=df['tokenized_review'], vector_size=100, window=5, min_count=30, workers=4)

# Save model for later use
word2vec_model.save("word2vec_imdb.model")

# Load trained Word2Vec model
w2v_model = Word2Vec.load("word2vec_imdb.model")

# Extract word vectors for clustering
word_vectors = Word2Vec.load("word2vec_imdb.model").wv

# ✅ Initialize accumulators for metrics
num_runs = 1
total_accuracy = 0
total_precision = 0
total_recall = 0
total_f1 = 0
total_miss = 0

# Define classification function ONCE (before loop)
def classify_review_kmeans(review, true_label, word_clusters):
    words = word_tokenize(review.lower())

    # Ensure "great" and "unoriginal" exist in clusters
    if "great" not in word_clusters or "unoriginal" not in word_clusters:
        return "Miss"

    # Count words from "Positive" and "Negative" clusters
    pos_count = sum(1 for word in words if word_clusters.get(word, -1) == word_clusters["great"])
    neg_count = sum(1 for word in words if word_clusters.get(word, -1) == word_clusters["unoriginal"])

    # Classification logic
    if pos_count > neg_count:
        return "Positive"
    elif pos_count < neg_count:
        return "Negative"
    else:
        return "Miss"


for i in range(num_runs):
    print(f"\n Run {i + 1}/{num_runs}...")

    num_clusters = 10
    # Apply k-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=50)
    kmeans.fit(word_vectors.vectors)

    word_list = word2vec_model.wv.index_to_key
    word_clusters = {word: kmeans.labels_[i] for i, word in enumerate(word_list)}

    ###
    # cluster_counts = Counter(kmeans.labels_)
    # for cluster_id in cluster_counts.keys():
    #     print(f"\nSample words in Cluster {cluster_id}:")
    #     words_in_cluster = [word for word, lbl in word_clusters.items() if lbl == cluster_id]
    #     print(words_in_cluster[:20])  # Print first 20 words
    #
    # word_vectors_reduced = PCA(n_components=2).fit_transform(word_vectors.vectors[:1000])  # Reduce to 2D for plotting
    # import pandas as pd
    # import seaborn as sns
    #
    # df_visual = pd.DataFrame(word_vectors_reduced, columns=["PCA 1", "PCA 2"])
    # df_visual["Word"] = word_list
    # df_visual["Cluster"] = kmeans.labels_[:1000]
    #
    # # Plot the clusters
    # plt.figure(figsize=(10, 7))
    # sns.scatterplot(data=df_visual, x="PCA 1", y="PCA 2", hue="Cluster", palette="tab10", s=100, edgecolor="black")
    #
    # # Annotate words in the scatter plot
    # for i, word in enumerate(df_visual["Word"]):
    #     plt.annotate(word, (df_visual["PCA 1"][i], df_visual["PCA 2"][i]), fontsize=9, alpha=0.75)
    #
    # plt.title("K-Means Clustering of Word Embeddings (Word2Vec)")
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    # plt.legend(title="Cluster")
    # plt.grid(True)
    # plt.show()
    ###

    # Apply classification function to reviews
    df['predicted_sentiment'] = df.apply(lambda row: classify_review_kmeans(row['cleaned_review'], row['sentiment'], word_clusters), axis=1)

    # Filter out "Miss" cases
    df_filtered = df[df['predicted_sentiment'] != "Miss"]

    if df_filtered.empty:  # Skip if there are no valid predictions
        print(f"Run {i+1} - No valid predictions, skipping...")
        continue

    # Compute accuracy
    accuracy = accuracy_score(df_filtered['sentiment'], df_filtered['predicted_sentiment'])
    total_accuracy += accuracy

    # Compute precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_filtered['sentiment'], df_filtered['predicted_sentiment'], average='weighted', zero_division=1
    )
    total_precision += precision
    total_recall += recall
    total_f1 += f1

    # Compute Miss Rate
    miss_count = (df['predicted_sentiment'] == "Miss").sum()
    total_reviews = len(df)
    miss_rate = miss_count / total_reviews
    total_miss += miss_rate

    print(f"Run {i+1} - Accuracy: {accuracy:.4f}, Miss Rate: {miss_rate:.4%}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Compute Average Results
num_valid_runs = num_runs  # Adjust in case some runs were skipped

avg_accuracy = total_accuracy / num_valid_runs
avg_precision = total_precision / num_valid_runs
avg_recall = total_recall / num_valid_runs
avg_f1 = total_f1 / num_valid_runs
avg_miss_rate = total_miss / num_valid_runs

# Print Final Summary
print("\n Final Average Results Over Multiple Runs:")
print(f"Average Accuracy: {avg_accuracy:.2f}")
print(f"Average Miss Rate: {avg_miss_rate:.2%}")
print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall: {avg_recall:.2f}")
print(f"Average F1-Score: {avg_f1:.2f}")

end = time.time()
print(end - start)