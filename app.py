import gradio as gr
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load model
model = joblib.load("model.pkl")


def predict(f1, f2, f3, f4):

    input_data = pd.DataFrame(
        [[f1, f2, f3, f4]],
        columns=["F1","F2","F3","F4"]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)

    confidence = max(probability[0]) * 100

    if prediction == 1:
        result = f"✅ Positive Class (1)\nConfidence: {confidence:.2f}%"
    else:
        result = f"❌ Negative Class (0)\nConfidence: {confidence:.2f}%"

    return result


def show_tree():

    plt.figure(figsize=(12,8))

    plot_tree(
        model,
        feature_names=["F1","F2","F3","F4"],
        class_names=["Class 0","Class 1"],
        filled=True
    )

    plt.title("Decision Tree Visualization")

    plt.savefig("tree.png")

    return "tree.png"


with gr.Blocks() as app:

    gr.Markdown("# 🌳 Decision Tree Binary Classifier")

    gr.Markdown("Enter feature values to predict class and see confidence score.")

    with gr.Row():

        with gr.Column():

            f1 = gr.Slider(-5,5,label="Feature 1")
            f2 = gr.Slider(-5,5,label="Feature 2")
            f3 = gr.Slider(-5,5,label="Feature 3")
            f4 = gr.Slider(-5,5,label="Feature 4")

            predict_btn = gr.Button("Predict")

        with gr.Column():

            output = gr.Textbox(label="Prediction Result")

    predict_btn.click(
        fn=predict,
        inputs=[f1,f2,f3,f4],
        outputs=output
    )

    gr.Markdown("## 🌳 Model Visualization")

    tree_btn = gr.Button("Show Decision Tree")

    tree_image = gr.Image()

    tree_btn.click(
        fn=show_tree,
        outputs=tree_image
    )

app.launch(share=True)