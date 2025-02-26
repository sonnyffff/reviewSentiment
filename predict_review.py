import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.utils import plot_model
import pydot
import graphviz
import pydotplus


# Load the trained MLP model
model_path = "mlp_word_embeddings.keras"
model_path2 = "mlp_baseline.keras"
mlp_model = tf.keras.models.load_model(model_path)

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define constants
max_length = 200  # Same as used in training

# Function to preprocess and predict sentiment
def predict_sentiment(review):
    # Tokenize input text
    sequence = tokenizer.texts_to_sequences([review])

    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Predict sentiment
    prediction = mlp_model.predict(padded_sequence)[0][0]

    # Convert probability to label
    sentiment = "Positive" if prediction >= 0.5 else "Negative"

    return sentiment, prediction


# Run interactive loop
if __name__ == "__main__":
    plot_model(mlp_model, to_file="mlp_model.png", show_shapes=True, show_layer_names=True)
    print("Movie Review Sentiment Predictor (Type 'exit' to quit)")
    while True:
        user_input = input("\nEnter a movie review: ")
        if user_input.lower() == "exit":
            print("Exiting program...")
            break

        sentiment, confidence = predict_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")
