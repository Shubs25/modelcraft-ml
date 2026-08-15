import os
import joblib
import numpy as np
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
st.set_page_config(page_title="ModelCraft-ML Benchmark", page_icon="🟣", layout="centered")

st.title("ModelCraft-ML")
st.caption("Is a customer is likely to default on their next credit-card payment? Let's find out!")
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
st.header("Dataset Input")

if st.button("Peek into sample training data"):
    if os.path.exists("test_data.csv"):
        df_sample = pd.read_csv("test_data.csv")
        st.dataframe(df_sample.head(5), use_container_width=True)

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])

st.markdown("---")

# Model Evaluation
st.header("Model Evaluation & Diagnostics")

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

st.markdown("---")

#----------------------
# Custom Prediction
st.header("Custom Prediction")
st.write("Want to see what each model says about your own input? Expand this form and fill!")

with st.expander("Expand me!", expanded=False):
    tab_demo, tab_pay_stat, tab_bill, tab_pay_amt = st.tabs([
        "Demographics", "Repayment Status", "Bill Statements", "Payment Amounts"
    ])

    with tab_demo:
        c1, c2, c3 = st.columns(3)
        limit_bal = c1.number_input("Credit Limit (LIMIT_BAL)", min_value=1000, max_value=1000000, value=50000,
                                    step=5000)
        sex = c2.selectbox("Gender (SEX)", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = c3.selectbox("Education (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x:
        {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}[x])

        c4, c5 = st.columns(2)
        marriage = c4.selectbox("Marital Status (MARRIAGE)", options=[1, 2, 3],
                                format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
        age = c5.number_input("Age (AGE)", min_value=18, max_value=100, value=30, step=1)

    with tab_pay_stat:
        st.caption("Repayment status: -1 = Pay duly, 0 = Revolving credit, 1-8 = Payment delay (in months)")
        c1, c2, c3 = st.columns(3)
        pay_0 = c1.number_input("Last Month (PAY_0)", min_value=-2, max_value=8, value=0)
        pay_2 = c2.number_input("2 Months Ago (PAY_2)", min_value=-2, max_value=8, value=0)
        pay_3 = c3.number_input("3 Months Ago (PAY_3)", min_value=-2, max_value=8, value=0)

        c4, c5, c6 = st.columns(3)
        pay_4 = c4.number_input("4 Months Ago (PAY_4)", min_value=-2, max_value=8, value=0)
        pay_5 = c5.number_input("5 Months Ago (PAY_5)", min_value=-2, max_value=8, value=0)
        pay_6 = c6.number_input("6 Months Ago (PAY_6)", min_value=-2, max_value=8, value=0)

    with tab_bill:
        c1, c2, c3 = st.columns(3)
        bill_amt1 = c1.number_input("Bill Statement 1 (BILL_AMT1)", value=3000, step=500)
        bill_amt2 = c2.number_input("Bill Statement 2 (BILL_AMT2)", value=2500, step=500)
        bill_amt3 = c3.number_input("Bill Statement 3 (BILL_AMT3)", value=2000, step=500)

        c4, c5, c6 = st.columns(3)
        bill_amt4 = c4.number_input("Bill Statement 4 (BILL_AMT4)", value=1500, step=500)
        bill_amt5 = c5.number_input("Bill Statement 5 (BILL_AMT5)", value=1000, step=500)
        bill_amt6 = c6.number_input("Bill Statement 6 (BILL_AMT6)", value=500, step=500)

    with tab_pay_amt:
        c1, c2, c3 = st.columns(3)
        pay_amt1 = c1.number_input("Previous Payment 1 (PAY_AMT1)", min_value=0, value=1000, step=200)
        pay_amt2 = c2.number_input("Previous Payment 2 (PAY_AMT2)", min_value=0, value=1000, step=200)
        pay_amt3 = c3.number_input("Previous Payment 3 (PAY_AMT3)", min_value=0, value=1000, step=200)

        c4, c5, c6 = st.columns(3)
        pay_amt4 = c4.number_input("Previous Payment 4 (PAY_AMT4)", min_value=0, value=1000, step=200)
        pay_amt5 = c5.number_input("Previous Payment 5 (PAY_AMT5)", min_value=0, value=1000, step=200)
        pay_amt6 = c6.number_input("Previous Payment 6 (PAY_AMT6)", min_value=0, value=1000, step=200)

    predict_custom_btn = st.button("Predict Default Risk for this Input", type="primary")

if 'predict_custom_btn' in locals() and predict_custom_btn:
    scaler, models = load_artifacts()

    # Construct the input vector
    input_values = np.array([[
        limit_bal, sex, education, marriage, age,
        pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
        bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
        pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
    ]])

    input_scaled = scaler.transform(input_values)

    custom_predictions = []
    for name, model in models.items():
        inp = input_scaled if name in SCALED_MODELS else input_values
        pred = int(model.predict(inp)[0])
        prob = model.predict_proba(inp)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)

        custom_predictions.append({
            "Model": name,
            "Prediction": "Default (1)" if pred == 1 else "No Default (0)",
            "Default Probability": f"{prob * 100:.2f}%",
            "Risk Assessment": "High Risk" if prob >= 0.50 else "Low Risk"
        })

    st.subheader("Prediction Results for Entered Input")
    st.dataframe(pd.DataFrame(custom_predictions), use_container_width=True, hide_index=True)