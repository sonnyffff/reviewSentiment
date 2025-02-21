import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
import pickle

import numpy as np





# Load IMDb dataset (Ensure it's preprocessed)
file_path = "datasets/imdb_reviews_cleaned.csv"  # Update if needed
df = pd.read_csv(file_path)

# Split data into training and testing sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
)


# Tokenization parameters
vocab_size = 20000  # Define vocabulary size
max_length = 200  # Maximum review length (in words)
embedding_dim = 100  # Word embedding dimensions
oov_token = "<OOV>"  # Token for out-of-vocabulary words

# Tokenize the text
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)

# Save the tokenizer after fitting it
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Tokenizer saved successfully as tokenizer.pkl")

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

# Pad sequences to ensure uniform length
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


# Load GloVe embeddings (make sure you have "glove.6B.100d.txt" in your working directory)#############################
embedding_index = {}

with open("glove.6B/glove.6B.100d.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

print(f"✅ Loaded {len(embedding_index)} word vectors from GloVe.")

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print("✅ Embedding matrix created!")

embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],  # Use GloVe embeddings
    input_length=max_length,
    trainable=False  # Freeze embeddings (optional)
)


# Define the MLP model using Word Embeddings
mlp_model = Sequential([
    # Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    embedding_layer,
    # GlobalAveragePooling1D(),  # Averages word embeddings for classification
    Bidirectional(LSTM(64, return_sequences=True)),  # Captures forward & backward context
    LSTM(32, return_sequences=False),  # Additional LSTM layer for deeper feature extraction
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
mlp_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 32

history = mlp_model.fit(
    train_padded, train_labels,
    validation_data=(test_padded, test_labels),
    epochs=epochs, batch_size=batch_size,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = mlp_model.evaluate(test_padded, test_labels, verbose=1)
# Print final training and validation accuracy
train_acc = history.history['accuracy'][-1]  # Last epoch training accuracy
val_acc = history.history['val_accuracy'][-1]  # Last epoch validation accuracy

print(f"\nFinal Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Save the model for future use
mlp_model.save("mlp_word_embeddings.keras")
