# ModelCraft-ML: Credit Card Default Risk

---
## Metadata
| Author        | BITS ID     | Subject                           |
|---------------|-------------|-----------------------------------|
| Shubham Ghosh | 2025AD05055 | S2-25_AIMLCZG565 Machine Learning |

### Live Link: https://bits-modelcraft-ml.streamlit.app

---

## A. Problem Statement

* **Objective:** Predict whether a credit card client will default on their upcoming monthly payment (`default.payment.next.month` = `1` vs `0`) using their demographic profile, credit limit, and six months of historical billing, repayment status, and payment activity.
* **Business Impact:** Credit card defaults lead to significant financial losses for issuing institutions. Identifying high-risk borrowers early allows risk management teams to take preventative actions, such as adjusting credit limits, initiating structured payment plans, or conducting manual account reviews.
* **Key Challenges:**
  * **Class Imbalance:** Non-defaulting cardholders outnumber defaulting cardholders significantly (~78% non-default vs. ~22% default). Traditional accuracy alone is misleading, requiring evaluation across balanced metrics including Precision, Recall, F1 Score, ROC-AUC, and the Matthews Correlation Coefficient (MCC).
  * **Cost Asymmetry:** False Negatives (granting credit to clients who fail to repay) carry much higher financial risk than False Positives (flagging a safe client for verification), prioritizing higher Recall in real-world scenarios.

---

## B. Dataset Description

* **Dataset:** Default of Credit Card Clients Dataset
* **Source:** UCI Machine Learning Repository (Dataset ID: `350`)
  * **URL:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
* **Target Variable:** `default.payment.next.month` (`1` = Default, `0` = No Default / Paid Duly)
* **Total Features:** 23 numerical and categorical variables

### Feature Grouping

| Feature Category      | Column Names                                         | Description                                                                                                                                                                                              |
|:----------------------|:-----------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Demographics**      | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`   | Amount of credit limit (NT dollar), gender (1=Male, 2=Female), education level (1=Grad School, 2=University, 3=High School, 4=Others), marital status (1=Married, 2=Single, 3=Others), and age in years. |
| **Repayment Status**  | `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` | Monthly repayment history from April to September. Values represent: `-2` (No consumption), `-1` (Paid in full), `0` (Revolving credit / minimum paid), and `1`–`8` (Payment delay in months).           |
| **Bill Statements**   | `BILL_AMT1` to `BILL_AMT6`                           | Monthly bill statement balance (NT dollar) from April to September.                                                                                                                                      |
| **Previous Payments** | `PAY_AMT1` to `PAY_AMT6`                             | Historical payment amount (NT dollar) made during each preceding month from April to September.                                                                                                          |

---
## C. GitHub Repository Link
### https://github.com/Shubs25/modelcraft-ml

---

## D. Models Used

Five classical supervised classification algorithms representing distinct mathematical paradigms are benchmarked under identical train/test splits:

#### i. Logistic Regression (`LogisticRegression`)
* **Model Type:** Probabilistic Linear Model
* **Description:** Maps a linear combination of input features to a probability between 0 and 1 using the standard sigmoid function.
* **Data Preprocessing:** Requires feature scaling via `StandardScaler` to ensure numerical stability and balanced gradient descent updates.

#### ii. Decision Tree Classifier (`DecisionTreeClassifier`)
* **Model Type:** Non-Parametric Rule-Based Model
* **Description:** Recursively partitions the feature space into rectangular regions using Information Gain to maximize class separation.
* **Data Preprocessing:** Operates directly on raw feature values without standard scaling.

#### iii. K-Nearest Neighbors (`KNeighborsClassifier`)
* **Model Type:** Instance-Based / Distance-Based Classifier
* **Description:** Evaluates the Euclidean distance across the 23-dimensional feature space to locate the $k=5$ nearest neighbors and assigns the majority class.
* **Data Preprocessing:** This is sensitive to feature magnitudes, hence it requires input normalization via `StandardScaler`.

#### iv. Gaussian Naive Bayes (`GaussianNB`)
* **Model Type:** Probabilistic Bayesian Classifier
* **Description:** Applies Bayes' Theorem assuming that all 23 input features follow Gaussian distributions and are conditionally independent given the class label.
* **Data Preprocessing:** Evaluated on standardized continuous features.

#### v. Random Forest Classifier (`RandomForestClassifier`)
* **Model Type:** Ensemble Learning (Bagging / Bootstrap Aggregation)
* **Description:** Constructs an ensemble of 100 de-correlated decision trees using random feature subsets and outputs the majority class prediction across all trees to minimize model variance.
* **Data Preprocessing:** Accepts raw numerical features directly.

## Evaluation Metrics Across Models

| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.8077   | 0.7076 | 0.6868    | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree       | 0.7145   | 0.6075 | 0.3694    | 0.4115 | 0.3893 | 0.2042 |
| K-Nearest Neighbors | 0.7928   | 0.7015 | 0.5487    | 0.3564 | 0.4322 | 0.3233 |
| Naive Bayes         | 0.7525   | 0.7249 | 0.4515    | 0.5539 | 0.4975 | 0.3386 |
| Random Forest       | 0.8120   | 0.7506 | 0.6325    | 0.3580 | 0.4572 | 0.3749 |

## Model Performance Observations

| ML Model Name        | Observaton about Model Performance                                                                                                                                                                                                                      |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Random Forest        | Best overall classifier, achieving the highest Accuracy (81.20%), AUC (0.7506), and MCC (0.3749). The ensemble of 100 trees captured the non-linear relationships and provided the strongest overall classification power.                              |
| Naive Bayes          | Best balanced F1 score, with the highest F1 score (0.4975) and Recall (55.39%). The conditional independence assumption makes it more willing to predict that someone will default, rather than being overly cautious, compared with tree-based models. |
| Logistic Regression  | Achieves high Accuracy (80.77%) and Precision (68.68%) but has low Recall (23.96%), meaning it identifies relatively few actual defaults. Without class re-weighting, the model tends to favor the majority class.                                      |
| K-Nearest Neighbors  | Balanced results, with balanced Precision (54.87%) and a reasonable AUC (0.7015), while achieving moderate Recall (35.64%).                                                                                                                             |
| Decision Tree        | Weakest overall performer, with the lowest AUC (0.6075) and MCC (0.2042). The unpruned single tree is more vulnerable to high variance, limiting its generalization and classification performance.                                                     |
| ***Overall Winner*** | ***Random Forest (Highest ROC-AUC, MCC, Accuracy & a solid Precision)***                                                                                                                                                                                |

---
