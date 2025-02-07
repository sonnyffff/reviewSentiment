import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

"""
Preprocess the IMDB datasets
"""

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset (Assuming the CSV has 'review' and 'sentiment' columns)
file_path = "datasets/IMDB Dataset.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Convert sentiment labels to binary (0 = negative, 1 = positive)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Function to clean and preprocess text"""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    words = word_tokenize(text)  # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize & remove stopwords
    return ' '.join(words)

# Apply cleaning function to reviews
df['cleaned_review'] = df['review'].apply(clean_text)

# Display sample processed data
print(df[['cleaned_review', 'sentiment']].head())

# Save the processed data
df.to_csv("datasets/imdb_reviews_cleaned.csv", index=False)
