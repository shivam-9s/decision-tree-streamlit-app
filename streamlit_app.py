import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load model
model = joblib.load("model.pkl")

st.title("🌳 Decision Tree Binary Classifier")

st.write("Enter feature values to predict class and confidence score.")

# Input sliders
f1 = st.slider("Feature 1", -5.0, 5.0, 0.0)
f2 = st.slider("Feature 2", -5.0, 5.0, 0.0)
f3 = st.slider("Feature 3", -5.0, 5.0, 0.0)
f4 = st.slider("Feature 4", -5.0, 5.0, 0.0)

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[f1, f2, f3, f4]],
        columns=["F1","F2","F3","F4"]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)

    confidence = max(probability[0]) * 100

    if prediction == 1:
        st.success(f"✅ Positive Class (1)")
    else:
        st.error(f"❌ Negative Class (0)")

    st.write(f"Confidence: {confidence:.2f}%")

# Tree visualization
st.subheader("🌳 Decision Tree Visualization")

if st.button("Show Decision Tree"):

    fig, ax = plt.subplots(figsize=(12,8))

    plot_tree(
        model,
        feature_names=["F1","F2","F3","F4"],
        class_names=["Class 0","Class 1"],
        filled=True
    )

    st.pyplot(fig)