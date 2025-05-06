from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

def train_and_evaluate_models(texts, labels):
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    vectorizers = {
        "2-gram": TfidfVectorizer(ngram_range=(2, 2), max_features=10000, min_df=2),
        "3-gram": TfidfVectorizer(ngram_range=(3, 3), max_features=10000, min_df=2),
    }

    models = {
        "SVM": SVC(kernel="linear"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        "Naive Bayes": MultinomialNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    for vec_name, vectorizer in vectorizers.items():
        X_train = vectorizer.fit_transform(X_train_texts)
        X_test = vectorizer.transform(X_test_texts)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"\n📌 {model_name} | Feature: {vec_name}")
            print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
            print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
            print(f"Recall:    {recall_score(y_test, y_pred, average='macro'):.4f}")
            print(f"F1-score:  {f1_score(y_test, y_pred, average='macro'):.4f}\n")
            print(classification_report(y_test, y_pred, target_names=le.classes_))