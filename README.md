# Customer Churn Prediction

A machine learning project that uses XGBoost to predict customer churn. The repository includes a Jupyter Notebook for data exploration and model development, a Python dashboard, and a serialized model for prediction workflows.

## Overview

Customer churn occurs when customers stop using a company's products or services. Predicting churn can help businesses identify customers who may leave and make better-informed customer retention decisions.

This project focuses on applying machine learning to customer churn prediction, with supporting components for data analysis, model usage, and visualization.

## Project Objectives

- Explore customer data and identify patterns associated with churn.
- Prepare data for machine learning.
- Develop an XGBoost-based churn prediction model.
- Evaluate model performance using appropriate classification metrics.
- Present prediction-related information through a dashboard.
- Explore how a churn prediction workflow can be integrated with backend and data streaming technologies.

## Repository Structure

All project files are maintained directly in the repository root.

| File | Purpose |
|---|---|
| `Customer_Churn_Prediction (1).ipynb` | Jupyter Notebook for data exploration, preprocessing, and model development. |
| `dashboard.py` | Python dashboard component for presenting the project's functionality. |
| `xgb_churn_model.pkl` | Serialized XGBoost model artifact. |
| `requirements.txt` | Python dependencies required by the project. |
| `__init__.py` | Python package initialization file. |
| `.gitignore` | Specifies files Git should ignore. |
| `README.md` | Project documentation and setup instructions. |

## Technology Stack

- **Programming language:** Python
- **Notebook environment:** Google Colab / Jupyter Notebook
- **Machine learning:** XGBoost
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Backend and data infrastructure:** FastAPI, Kafka, PostgreSQL, as described in the repository overview

## How to Run

### Prerequisites

Install Python, Git, and pip. Use a compatible Python environment for the dependencies and serialized model.

### 1. Clone the repository

```bash
git clone https://github.com/Revanth6699/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Open the notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `Customer_Churn_Prediction (1).ipynb` and execute the cells in sequence.

Alternatively, open the notebook in Google Colab and configure any required dataset access.

### 5. Launch the dashboard

If `dashboard.py` is implemented as a Streamlit application, run:

```bash
streamlit run dashboard.py
```

The dashboard's actual input requirements, model-loading path, and prediction workflow depend on the implementation in `dashboard.py`.

## Machine Learning Approach

The project uses XGBoost, a gradient-boosted decision tree algorithm, for customer churn classification.

A typical churn prediction workflow involves:

1. **Data preparation:** Inspect the dataset, handle missing values, and prepare the input features.
2. **Feature processing:** Convert categorical variables and prepare numerical features as required by the model.
3. **Model development:** Train an XGBoost classifier using labeled customer data.
4. **Model evaluation:** Assess predictions using suitable classification metrics.
5. **Prediction:** Apply the trained model to compatible customer records.
6. **Visualization:** Present relevant results through the dashboard.

The exact preprocessing steps, evaluation metrics, and training configuration should be confirmed from the notebook before treating them as implemented features.

## Model Artifact

The repository includes `xgb_churn_model.pkl`, a serialized model file.

To reuse this artifact, the prediction code must provide inputs in the same feature order and format expected by the trained model. The compatible XGBoost version and any preprocessing requirements should also be verified.

Only load serialized model files from trusted sources.

## Potential Applications

Customer churn prediction can support:

- Identifying customers who may be at risk of leaving.
- Understanding patterns associated with customer attrition.
- Supporting customer retention analysis.
- Helping business teams prioritize further investigation.

Predictions indicate estimated risk, not certainty that a customer will leave.

## Limitations

- Model performance depends on the quality and representativeness of the training data.
- Changes in customer behavior can reduce predictive performance over time.
- A prediction model alone does not establish the reasons a customer will leave.
- Real-time processing and deployment require the corresponding infrastructure to be configured and operational.
- No accuracy or other performance figures are claimed here because verified evaluation results have not been documented in this README.

## Future Scope

Potential extensions include:

- Comparing XGBoost with baseline classification models.
- Evaluating precision, recall, F1-score, and ROC-AUC on held-out data.
- Improving model interpretability and identifying influential features.
- Integrating validated predictions into a customer retention workflow.
- Adding monitoring to track changes in data and model performance.

These are potential extensions rather than claims about existing functionality.

## Author

**Revanth Kumar**

GitHub: [@Revanth6699](https://github.com/Revanth6699)

## Disclaimer

This project is intended for educational and analytical purposes. Its predictions should not be treated as guarantees of customer behavior or as a substitute for business judgment.
