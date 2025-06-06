import streamlit as st
import pandas as pd
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from sklearn.metrics import confusion_matrix
import seaborn as sns

with mlflow.start_run():
    # Load the dataset (assuming it's in the same directory)
    customer = pd.read_csv("Customer Churn.csv")
    
    # Preprocessing
    X = customer.drop("Churn", axis=1)
    y = customer.Churn
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # 3. Train the Model (Random Forest)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 4. Evaluate the Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 5. Log Parameters, Metrics, and the Model
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')


    # Save and log the plot
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

print(f"Model trained and logged with accuracy: {accuracy}")
