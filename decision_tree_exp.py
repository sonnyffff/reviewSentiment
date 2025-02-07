import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import time

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.
        """
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        # Stopping conditions
        if len(np.unique(y)) == 1:  # Only one class left
            return {'label': np.unique(y)[0]}
        if len(X) == 0:  # No data left
            return {'label': np.random.choice(y)}
        if self.max_depth and depth >= self.max_depth:  # Max depth reached
            return {'label': self._majority_class(y)}

        # Find the best split
        best_split = self._get_best_split(X, y)
        left_tree = self._build_tree(*best_split['left'], depth + 1)
        right_tree = self._build_tree(*best_split['right'], depth + 1)

        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _get_best_split(self, X, y):
        """
        Find the best feature and threshold to split the data.
        """
        best_gini = float('inf')
        best_split = None

        # Loop over all features
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_y, right_y = y[left_mask], y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:  # No data to split
                    continue

                gini = self._gini_impurity(left_y, right_y)

                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left': (X[left_mask], left_y),
                        'right': (X[right_mask], right_y)
                    }

        return best_split

    def _gini_impurity(self, left_y, right_y):
        """
        Compute the Gini impurity of a split.
        """
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size

        # Compute Gini for left and right splits
        left_gini = 1 - sum((np.sum(left_y == c) / left_size) ** 2 for c in np.unique(left_y))
        right_gini = 1 - sum((np.sum(right_y == c) / right_size) ** 2 for c in np.unique(right_y))

        # Weighted Gini for both splits
        gini = (left_size / total_size) * left_gini + (right_size / total_size) * right_gini
        return gini

    def _majority_class(self, y):
        """
        Return the majority class label for a set of labels.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        """
        Predict the class label for each sample in X.
        """
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        """
        Predict the class label for a single sample.
        """
        if 'label' in tree:
            return tree['label']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

# Example usage
if __name__ == "__main__":
    start = time.time()
    # Load the preprocessed dataset
    file_path = "datasets/imdb_reviews_cleaned.csv"  # Ensure the correct path

    df = pd.read_csv(file_path)

    df = df.sample(frac=0.5, random_state=42)

    # Convert text data into numerical format using TF-IDF
    vectorizer = TfidfVectorizer(max_features=2000)  # Limit vocab size for efficiency
    X = vectorizer.fit_transform(df['cleaned_review']).toarray()  # Convert to dense array

    # Target variable (sentiment: 1 = positive, 0 = negative)
    y = df['sentiment']

    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the decision tree
    tree = DecisionTree(max_depth=7)
    tree.fit(X_train, y_train)

    # Make predictions
    y_pred = tree.predict(X_test)

    # Use training accuracy for comparison
    y_train_pred = tree.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    print(f"Training Accuracy: {train_accuracy}")

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    end = time.time()
    print(end - start)
