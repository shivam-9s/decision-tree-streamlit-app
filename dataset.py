import pandas as pd
from sklearn.datasets import make_classification

def create_data():

    # Generate dummy binary classification data
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_classes=2,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    # Convert to DataFrame
    df = pd.DataFrame(X, columns=["F1", "F2", "F3", "F4"])
    df["Target"] = y

    # Save dataset
    df.to_csv("dummy_data.csv", index=False)

    print("✅ Dummy dataset created successfully!")

if __name__ == "__main__":
    create_data()