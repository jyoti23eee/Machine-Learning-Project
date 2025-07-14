# Email Spam Detection using Machine Learning
This project focuses on building a machine learning model to detect spam vs ham (non-spam) emails using Natural Language Processing (NLP) techniques. It also incorporates model explainability using SHAP (SHapley Additive exPlanations).
# 🧠 Project Objectives
Classify email messages as Spam or Ham

Apply NLP for text cleaning and vectorization

Train multiple models (Logistic Regression, Random Forest, SVC)

Use a Voting Classifier to ensemble predictions

Visualize model performance and class distribution

Add explainability using SHAP to interpret model predictions
# 🔍 Dataset Used
The dataset contains two columns:

Category: Target label (spam or ham)

Message: The email text content

You can use the popular SMS Spam Collection Dataset or your own labeled email data.
# 🛠️ Key Features
Preprocessing:

Lowercasing

Removing punctuation and stopwords

Tokenization and lemmatization

TF-IDF vectorization

Model Training:

Individual models: LogisticRegression, RandomForestClassifier, SVC

Combined with VotingClassifier (hard or soft voting)

Evaluation Metrics:

Accuracy, Precision, Recall, F1-Score

Confusion Matrix

ROC Curve and AUC

Visualization:

Class distribution (Pie chart)

Most common spam/ham words (WordCloud)

Model evaluation charts

Explainability with SHAP:

Use shap.TreeExplainer on Random Forest to understand feature impact

Waterfall plot to explain individual predictions

# Example SHAP Waterfall Plot
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer(X_test_array[:1])
shap.plots.waterfall(shap_values[0])
# 📦 Requirements
Python ≥ 3.8

pandas, numpy, matplotlib, seaborn

sklearn

nltk

shap

wordcloud
pip install -r requirements.txt
# 📁 Folder Structure
├── data/
│   └── spam.csv
├── notebooks/
│   └── spam_detection.ipynb
├── README.md

