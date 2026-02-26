import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load model
model = joblib.load("model.pkl")

# Load dataset
df = pd.read_csv("dummy_data.csv")

X = df[["F1","F2","F3","F4"]]

# Plot tree
plt.figure(figsize=(12,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Class 0","Class 1"],
    filled=True
)

plt.title("Decision Tree Visualization")

plt.savefig("decision_tree.png")

print("✅ Decision tree saved as decision_tree.png")

plt.show()