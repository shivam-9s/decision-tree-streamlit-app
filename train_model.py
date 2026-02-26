import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model():

    # Load dataset
    df = pd.read_csv("dummy_data.csv")

    # Features
    X = df[["F1", "F2", "F3", "F4"]]

    # Target
    y = df["Target"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    model = DecisionTreeClassifier()

    # Train
    model.fit(X_train, y_train)

    # Test accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("✅ Model trained successfully!")
    print("Accuracy:", accuracy)

    # Save model
    joblib.dump(model, "model.pkl")

    print("✅ Model saved as model.pkl")

if __name__ == "__main__":
    train_model()