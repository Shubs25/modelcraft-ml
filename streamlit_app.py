import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# Page Configuration
st.set_page_config(page_title="ModelCraft-ML Benchmark", page_icon="🧊", layout="centered")

st.title("ModelCraft-ML")
st.caption("Just like Minecraft, but with ML Models!")
st.markdown("---")

TARGET_COL = "default.payment.next.month"
SCALED_MODELS = {"Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"}


@st.cache_resource
def load_artifacts():
    """Load pre-fitted scaler and saved model binaries from disk."""
    scaler = joblib.load(os.path.join("model", "scaler.joblib"))
    models = {
        "Logistic Regression": joblib.load(os.path.join("model", "logistic_regression.joblib")),
        "Decision Tree": joblib.load(os.path.join("model", "decision_tree.joblib")),
        "K-Nearest Neighbors": joblib.load(os.path.join("model", "knn.joblib")),
        "Naive Bayes": joblib.load(os.path.join("model", "naive_bayes.joblib")),
        "Random Forest": joblib.load(os.path.join("model", "random_forest.joblib")),
    }
    return scaler, models


# Data Preview & Upload
st.header("1. Dataset Input")

if st.button("Peek into sample training data"):
    if os.path.exists("test_data.csv"):
        df_sample = pd.read_csv("test_data.csv")
        st.dataframe(df_sample.head(5), use_container_width=True)

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])

st.markdown("---")

# Model Evaluation
st.header("2. Model Evaluation & Diagnostics")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

        # Split features and target labels
        if TARGET_COL in df.columns:
            y_test = df[TARGET_COL]
            X_test_df = df.drop(columns=[TARGET_COL])
        else:
            X_test_df = df.iloc[:, :-1]
            y_test = df.iloc[:, -1]

        scaler, models = load_artifacts()
        X_test_vals = X_test_df.values
        X_test_scaled_vals = scaler.transform(X_test_vals)

        # Model Selection Dropdown
        selected_model_name = st.selectbox(
            "Select Model to Evaluate:",
            options=list(models.keys())
        )

        model = models[selected_model_name]
        X_eval = X_test_scaled_vals if selected_model_name in SCALED_MODELS else X_test_vals

        # Predictions
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Metric Cards
        st.subheader(f"Metrics for {selected_model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC Score", f"{roc_auc_score(y_test, y_prob):.4f}")
        col3.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        col5.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
        col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        st.markdown("---")

        # Classification Report & Confusion Matrix
        st.subheader("Model Diagnostics")
        tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])

        with tab1:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["No Default (0)", "Default (1)"],
                        yticklabels=["No Default (0)", "Default (1)"])
            ax.set_ylabel("Actual Label")
            ax.set_xlabel("Predicted Label")
            ax.set_title(f"Confusion Matrix ({selected_model_name})")
            st.pyplot(fig)

        with tab2:
            report_str = classification_report(y_test, y_pred, zero_division=0)
            st.code(report_str, language="text")

        # Overall Multi-Model Comparison Table
        with st.expander("View Comparison Across All 5 Models"):
            comparison_results = []
            for name, m in models.items():
                Xe = X_test_scaled_vals if name in SCALED_MODELS else X_test_vals
                yp = m.predict(Xe)
                ypr = m.predict_proba(Xe)[:, 1] if hasattr(m, "predict_proba") else yp
                comparison_results.append({
                    "Model": name,
                    "Accuracy": f"{accuracy_score(y_test, yp):.4f}",
                    "AUC": f"{roc_auc_score(y_test, ypr):.4f}",
                    "Precision": f"{precision_score(y_test, yp, zero_division=0):.4f}",
                    "Recall": f"{recall_score(y_test, yp, zero_division=0):.4f}",
                    "F1": f"{f1_score(y_test, yp, zero_division=0):.4f}",
                    "MCC": f"{matthews_corrcoef(y_test, yp):.4f}"
                })
            st.dataframe(pd.DataFrame(comparison_results), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")
else:
    st.info("Upload `test_data.csv` above to display evaluation results.")