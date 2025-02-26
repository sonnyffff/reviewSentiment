import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# Load preprocessed dataset
file_path = "datasets/imdb_reviews_cleaned.csv"
df = pd.read_csv(file_path)

# Convert text data into numerical format using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)  # Limit vocab size for efficiency
X = vectorizer.fit_transform(df['cleaned_review']) # Sparse Matrix

# Target variable (sentiment: 1 = positive, 0 = negative)
y = df['sentiment']

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_decision_tree(X_train, X_test, y_train, y_test):
    """
    Train a Decision Tree classifier WITHOUT hyperparameter tuning.
    """
    print("\nTraining Decision Tree without Fine-tuning...")
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)  # Fixed depth
    clf.fit(X_train, y_train)  # Train model

    # Make predictions
    y_pred = clf.predict(X_test)

    plt.figure(figsize=(20, 10))
    feature_names = vectorizer.get_feature_names_out()
    class_names = ['negative', 'positive']
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy}")

    # tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    # plt.show()

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def train_decision_tree_finetuned(X_train, X_test, y_train, y_test):
    """
    Train a Decision Tree classifier WITH hyperparameter tuning using GridSearchCV.
    """
    print("\nTuning Hyperparameters for Decision Tree...")

    # Define hyperparameter grid
    param_grid = {
        'max_depth': [5, 10, 20, None],  # Different tree depths
        'min_samples_split': [2, 5, 10],  # Minimum samples for a split
        'min_samples_leaf': [1, 2, 5],  # Minimum samples per leaf
        'criterion': ['gini', 'entropy']  # Split criteria
    }

    # Initialize Decision Tree classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("\nBest Hyperparameters Found:", best_params)

    # Make predictions with the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Optimized Decision Tree Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

train_decision_tree(X_train, X_test, y_train, y_test)
