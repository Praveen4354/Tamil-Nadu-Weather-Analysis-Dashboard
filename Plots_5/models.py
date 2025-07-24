import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, learning_curve
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

def plot_model_evaluation(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=model.classes_, y=model.classes_, title=f'Confusion Matrix - {model_name}')
    fig.update_xaxes(side="top")
    fig.update_traces(hoverinfo='all')
    for i in range(len(cm)):
        for j in range(len(cm)):
            fig.add_annotation(text=str(cm[i][j]), x=model.classes_[j], y=model.classes_[i], showarrow=False, font=dict(color="white" if cm[i][j] < np.max(cm) / 2 else "black"))
    fig.update_layout(width=1000, height=1000)
    st.plotly_chart(fig, use_container_width=True)

    report = classification_report(y_test, y_pred, labels=model.classes_, target_names=[str(cls) for cls in model.classes_])
    st.write(f'Classification Report - {model_name}')
    st.code(report)

    train_sizes, train_scores, test_scores = learning_curve(model, X_test, y_test, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean - train_scores_std,
                        mode='lines',
                        name='Train score - std',
                        line=dict(color='rgb(255, 127, 14)'),
                        fill=None))
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean + train_scores_std,
                        mode='lines',
                        name='Train score + std',
                        line=dict(color='rgb(255, 127, 14)'),
                        fill='tonexty'))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean - test_scores_std,
                        mode='lines',
                        name='Test score - std',
                        line=dict(color='rgb(31, 119, 180)'),
                        fill=None))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean + test_scores_std,
                        mode='lines',
                        name='Test score + std',
                        line=dict(color='rgb(31, 119, 180)'),
                        fill='tonexty'))
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean,
                        mode='lines+markers',
                        name='Train score',
                        line=dict(color='rgb(255, 127, 14)')))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean,
                        mode='lines+markers',
                        name='Test score',
                        line=dict(color='rgb(31, 119, 180)')))
    fig.update_layout(title=f'Learning Curve - {model_name}',
                      xaxis_title='Training examples',
                      yaxis_title='Score')
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve Plot (for multiclass classification)
    n_classes = len(model.classes_)
    y_test_bin = label_binarize(y_test, classes=model.classes_)
    y_pred_prob = model.predict_proba(X_test)

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, color in zip(range(n_classes), colors):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc_i = auc(fpr_i, tpr_i)
        fig.add_trace(go.Scatter(x=fpr_i, y=tpr_i,
                            mode='lines',
                            name=f'ROC curve of class {i} (area = {roc_auc_i:0.2f})',
                            line=dict(color=color)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(color='black', dash='dash')))
    fig.update_layout(title=f'ROC Curve - {model_name}',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def show_images_based_on_selection(target):
    if target == "SPI":
        image_path_1 = "spi_accuracy.jpg"
        image_path_2 = "spi_loss.jpg"
    elif target == "WPI":
        image_path_1 = "wpi_accuracy.jpg"
        image_path_2 = "wpi_loss.jpg"
    
    st.image(image_path_1, caption="Image 1", use_column_width=True)
    st.image(image_path_2, caption="Image 2", use_column_width=True)



def train_and_evaluate_model(data, target, model_name):
    spiF = ['solarradiation', 'solarenergy', 'uvindex', 'cloudcover', 'precipcover', 'humidity', 'precip']
    wpiF = ['windspeed', 'windspeedmax', 'windgust', 'temp', 'tempmax', 'tempmin']

    if target == 'SPI':
        X = data[spiF]
        y = data[target]
    else:
        X = data[wpiF]
        y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    model_filename = f'{model_name.replace(" ", "_")}_{target}.pkl'

    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        st.write(f"Loaded {model_name} model from {model_filename}")
    else:
        model = models[model_name]
        model.fit(X_train, y_train)
        joblib.dump(model, model_filename)
        st.write(f"Trained and saved {model_name} model to {model_filename}")

    st.write(f"Model: {model_name}")
    plot_model_evaluation(model, X_test, y_test, model_name)

def model_prediction_and_evaluation(data):
    # st.title("Model Prediction and Evaluation")

    # data = pd.read_csv("Master3.csv")

    st.write("Data Preview:")
    st.write(data.head())

    target_vars = ['SPI', 'WPI']
    col1, col2 = st.columns([1,5])
    with col1:
        target = st.selectbox("Select the target variable", target_vars)
    with col2:
        model_name = st.selectbox("Choose a model", ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Random Forest", "Neural Networks"])

    
    if st.button("Train and Evaluate"):
        if(model_name == "Neural Networks"):
            show_images_based_on_selection(target)
        else:
            train_and_evaluate_model(data, target, model_name)

