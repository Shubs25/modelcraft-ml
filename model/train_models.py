import os
import joblib
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Map raw UCI column codes (X1..X23) to readable feature names
COLUMN_MAPPING = {
    'X1': 'LIMIT_BAL', 'X2': 'SEX', 'X3': 'EDUCATION', 'X4': 'MARRIAGE', 'X5': 'AGE',
    'X6': 'PAY_0', 'X7': 'PAY_2', 'X8': 'PAY_3', 'X9': 'PAY_4', 'X10': 'PAY_5', 'X11': 'PAY_6',
    'X12': 'BILL_AMT1', 'X13': 'BILL_AMT2', 'X14': 'BILL_AMT3', 'X15': 'BILL_AMT4',
    'X16': 'BILL_AMT5', 'X17': 'BILL_AMT6', 'X18': 'PAY_AMT1', 'X19': 'PAY_AMT2',
    'X20': 'PAY_AMT3', 'X21': 'PAY_AMT4', 'X22': 'PAY_AMT5', 'X23': 'PAY_AMT6'
}

def check_cwd():
    if Path.cwd().name != 'model':
        raise RuntimeError("Current working directory is not the 'model' directory.")
    else:
        print('Current working directory is the model directory. Check succeeded.')

def train_and_export():
    check_cwd()

    print("Fetching dataset from UCI Repository...")
    credit_card = fetch_ucirepo(id=350)

    # Rename features BEFORE fitting models
    X = credit_card.data.features.rename(columns=COLUMN_MAPPING)
    y = credit_card.data.targets.iloc[:, 0]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit and Save Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, "./scaler.joblib")


    # Train and Save Models
    models = {
        "logistic_regression.joblib": LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train),
        "decision_tree.joblib": DecisionTreeClassifier(random_state=42).fit(X_train, y_train),
        "knn.joblib": KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train),
        "naive_bayes.joblib": GaussianNB().fit(X_train_scaled, y_train),
        "random_forest.joblib": RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    }

    for filename, model in models.items():
        joblib.dump(model, os.path.join(".", filename))

    print("Successfully retrained and updated model binaries in the cwd (model)")

def generate_test_csv(sample_size: int = None, filename: str = "test_data.csv"):
    """
        Fetch dataset from UCI repository and export the test set to CSV.
        Essentially generating a fresh test set
        This function is a little heavy, so use it wisely :)
    """
    check_cwd()
    print("Fetching dataset from UCI Repository...")
    credit_card = fetch_ucirepo(id=350)

    X = credit_card.data.features.rename(columns=COLUMN_MAPPING)
    y = credit_card.data.targets.iloc[:, 0]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    test_df = X_test.copy()
    test_df['default.payment.next.month'] = y_test.values

    if sample_size:
        test_df = test_df.head(sample_size)

    test_df.to_csv(os.path.join('..', filename), index=False)
    print(f"Successfully exported {len(test_df)} rows to '{filename}'.")

TEST_DATA_SIZE = 6000
TEST_DATA_FILE_NAME = "test_data.csv"

if __name__ == "__main__":
    print("1. Train and Export Models")
    print("2. Generate Test CSV")
    choice = input("Enter choice: ")

    if choice == "1":
        train_and_export()
    elif choice == "2":
        generate_test_csv(TEST_DATA_SIZE, TEST_DATA_FILE_NAME)
    else:
        print("What was that?")