# Author Classification Using Text Mining and Machine Learning

This repository contains a collection of Jupyter notebooks implementing various machine learning and deep learning techniques for **author classification**. The project explores traditional ML algorithms and transformer-based models to determine the author of a given text based on its linguistic features.

---
Folder implementation :change the folder resource with your path 
# Project Overview

The task of **author classification** involves identifying the most likely author of a piece of text using statistical and deep learning techniques. This is a common problem in **stylometry**, **natural language processing (NLP)**, and **forensic linguistics**.

# Key Objectives

- Extract meaningful features from raw text using **TF-IDF** and **BERT** embeddings.
- Train and evaluate multiple classification models.
- Compare the performance of classical ML algorithms with transformer-based approaches.
- Analyze the results using evaluation metrics like accuracy and classification report.

---

# Models and Approaches

The following models and techniques were applied:

| Notebook         | Approach                          | Description |
|------------------|-----------------------------------|-------------|
| `BERT.ipynb`     | BERT + Classifiers                | Uses `transformers` library to extract embeddings from a pre-trained BERT model. |
| `Desicion_Tree.ipynb` | TF-IDF + Decision Tree          | A tree-based method for classification using sparse TF-IDF features. |
| `MLP_CSM.ipynb`  | TF-IDF + MLP                      | Applies a Multi-Layer Perceptron for text classification. |
| `Naive_Bayes.ipynb` | TF-IDF + Multinomial Naive Bayes | Probabilistic approach using TF-IDF transformed inputs. |
| `Random_Forest.ipynb` | TF-IDF + Random Forest            | Ensemble method with multiple decision trees. |
| `Xgboost.ipynb`  | TF-IDF + XGBoost                  | Gradient-boosted decision tree classifier optimized with `xgboost`. |

---

# Evaluation Metrics

Each model is evaluated using the following metrics:

- **Accuracy**
- **Classification Report (Precision, Recall, F1-Score)**
- **Confusion Matrix (in some notebooks)**

---

## 📂 Folder Structure
