# 🌳 Decision Tree Binary Classification Web App

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Scikit Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end **Machine Learning web application** that predicts binary outcomes using a **Decision Tree Classifier** and an interactive dashboard built with Streamlit.

The application allows users to input feature values, generate predictions, view model confidence, and visualize the trained decision tree.

---

# 🚀 Live Demo

🔗 **Try the app here**

https://decision-tree-app-shivam.streamlit.app/

No installation required — open the link and interact with the ML model directly.

---

# 📸 Application Preview

*(Add screenshots here after capturing your running app)*

Example:

```
/screenshots
   dashboard.png
   prediction.png
   tree_visualization.png
```

Then embed them like:

```markdown
![Dashboard](screenshots/dashboard.png)
```

---

# 📌 Project Overview

This project demonstrates the **complete lifecycle of a machine learning system**, from data generation to deployment.

The system includes:

* Synthetic dataset creation
* Model training with Decision Tree
* Model evaluation
* Model serialization
* Interactive Streamlit dashboard
* Online deployment

The goal is to showcase how machine learning models can be **integrated into real-world web applications**.

---

# 🧠 Machine Learning Model

The model used is a **Decision Tree Classifier** from the Scikit-Learn library.

Decision Trees work by splitting the dataset based on feature thresholds to create a tree structure that predicts class labels.

### Binary Classes

Class 0 → Negative
Class 1 → Positive

---

# ⚙️ Tech Stack

| Technology   | Purpose                |
| ------------ | ---------------------- |
| Python       | Programming Language   |
| Streamlit    | Web App Framework      |
| Scikit-Learn | Machine Learning       |
| Pandas       | Data Processing        |
| Matplotlib   | Visualization          |
| Joblib       | Model Saving & Loading |

---

# 📂 Project Structure

```
decision-tree-streamlit-app
│
├── dataset.py
├── train_model.py
├── streamlit_app.py
├── dummy_data.csv
├── model.pkl
├── requirements.txt
└── README.md
```

---

# 📊 Application Features

✔ Interactive feature sliders
✔ Real-time prediction results
✔ Confidence score display
✔ Decision Tree visualization
✔ Fully deployed ML application
✔ User-friendly dashboard

---

# 🔄 Machine Learning Pipeline

```
Dataset Generation
        ↓
Data Preprocessing
        ↓
Model Training (Decision Tree)
        ↓
Model Evaluation
        ↓
Model Saved (Joblib)
        ↓
Streamlit Web Interface
        ↓
User Input → Prediction → Output
```

---

# 🧪 Dataset Generation

The dataset is generated using:

```python
from sklearn.datasets import make_classification
```

Dataset properties:

* 500 samples
* 4 input features
* Binary target variable

Example dataset:

| F1   | F2  | F3  | F4   | Target |
| ---- | --- | --- | ---- | ------ |
| -1.3 | 0.8 | 2.1 | -0.4 | 1      |

---

# 🧠 Model Training

The model is trained using:

```python
from sklearn.tree import DecisionTreeClassifier
```

Training steps:

1. Load dataset
2. Split dataset into train and test sets
3. Train Decision Tree classifier
4. Evaluate accuracy
5. Save trained model using Joblib

---

# 🌳 Decision Tree Visualization

The application also displays the learned decision tree using:

```python
from sklearn.tree import plot_tree
```

This visualization helps understand how the model splits features to make predictions.

---

# 💻 Running the Project Locally

### 1️⃣ Clone Repository

```
git clone https://github.com/shivam-9s/decision-tree-streamlit-app.git
cd decision-tree-streamlit-app
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Generate Dataset

```
python dataset.py
```

---

### 5️⃣ Train Model

```
python train_model.py
```

---

### 6️⃣ Run Application

```
streamlit run streamlit_app.py
```

Open in browser

```
http://localhost:8501
```

---

# 📈 Example Prediction

Input values:

```
Feature1 = -1
Feature2 = 1
Feature3 = 0
Feature4 = 2
```

Output:

```
Positive Class (1)
Confidence: 100%
```

---

# 🎯 Learning Outcomes

This project demonstrates:

* Binary Classification
* Decision Tree Algorithm
* Model Deployment
* Streamlit Application Development
* ML Model Serialization
* Data Visualization

---

# 🔮 Future Improvements

Possible improvements:

* Add confusion matrix visualization
* Feature importance chart
* Upload CSV for batch prediction
* Compare multiple ML models
* Add dataset explorer

---

# 👨‍💻 Author

**Shivam Kumar**

Machine Learning | Data Science | AI Projects

GitHub
https://github.com/shivam-9s

---

# ⭐ Support

If you like this project, consider **starring the repository**.

It helps others discover the project.
