import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

import pickle

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

print("Tokenizer saved successfully as tokenizer.pkl")

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

print(f"Loaded {len(embedding_index)} word vectors from GloVe.")

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print("Embedding matrix created!")

embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],  # Use GloVe embeddings
    input_length=max_length,
    trainable=False  # Freeze embeddings (optional)
)


# Define different optimizers and learning rates to test ##############################################################
optimizers = {
    # "Adam (lr=0.001)": Adam(learning_rate=0.001),
    "Adam (lr=0.0001)": Adam(learning_rate=0.0001),
    # "RMSprop (lr=0.001)": RMSprop(learning_rate=0.001),
    # "RMSprop (lr=0.0001)": RMSprop(learning_rate=0.0001),
    # "SGD (lr=0.01, momentum=0.9)": SGD(learning_rate=0.01, momentum=0.9),
    # "SGD (lr=0.001, momentum=0.9)": SGD(learning_rate=0.001, momentum=0.9)
}

# # Store results
results = {}

for name, optimizer in optimizers.items():
    print(f"\n🔹 Training with {name}...")
    # Define the MLP model using Word Embeddings ##############################################################
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
    # mlp_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    mlp_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    epochs = 5
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
    results[name] = test_loss

    # Save the model for future use
    mlp_model.save("mlp_word_embeddings.keras")

# Display results
import pandas as pd

# Convert results to a DataFrame
df_results = pd.DataFrame(list(results.items()), columns=['Optimizer', 'Test Loss'])

# Print table format
print("\nOptimizer and Learning Rate Comparison:\n")
print(df_results.to_string(index=False))


############################################################
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, GlobalAveragePooling1D
# from tensorflow.keras.callbacks import EarlyStopping
#
# # # Define different learning rates for Adam optimizer
# learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
#
# # Store results
# results = {}
#
# # Train and evaluate models with different learning rates
# for lr in learning_rates:
#     print(f"\n🔹 Training with Adam (lr={lr})...")
#
#     # Define the original model architecture
#     model = Sequential([
#         Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
#         GlobalAveragePooling1D(),
#         Dense(128, activation='relu'),
#         Dense(64, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#
#     # Compile model with the selected learning rate
#     optimizer = Adam(learning_rate=lr)
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#
#     # Early stopping to prevent overfitting
#     early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#
#     epochs = 20
#     batch_size = 16
#     # Train model
#     history = model.fit(
#         train_padded, train_labels,
#         validation_data=(test_padded, test_labels),
#         epochs=epochs, batch_size=batch_size,
#         callbacks=[early_stopping],
#         verbose=1
#     )
#
#     # Evaluate model
#     test_loss, test_accuracy = model.evaluate(test_padded, test_labels, verbose=1)
#     results[f"Adam (lr={lr})"] = test_loss
#
#     print(f"Adam (lr={lr}) - Test Accuracy: {test_accuracy:.4f}")
#     print(f"Batch {batch_size}, Epochs {epochs} - Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
#
# # Display results
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Convert results to a DataFrame
# df_results = pd.DataFrame(list(results.items()), columns=['Learning Rate', 'Test Loss'])
#
# # Print table format
# print("\nAdam Learning Rate Comparison:\n")
# print(df_results.to_string(index=False))
#
# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(df_results['Learning Rate'], df_results['Test Accuracy'], marker='o', linestyle='-', color='b')
# plt.xscale('log')  # Log scale for better visualization
# plt.xlabel("Learning Rate (log scale)")
# plt.ylabel("Test Accuracy")
# plt.title("Adam Optimizer: Learning Rate vs Test Accuracy")
# plt.grid(True)
# plt.show()