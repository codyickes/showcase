# Data Science Showcase: Interactive ML Applications

A collection of end-to-end Machine Learning appplications built with Python, Scikit-Learn, and Streamlit. These projects focus on interactive feature engineering and real-time model evaluation.

## Projects Overview

### Titanic Survival Predictor

This application explores the classic Titanic dataset through an interactive lens. Instead of a static notebook, users can dynamically select features and toggle between different classification algorithms to see how hyperparameters affect accuracy.

- Key Features:
    - Dynamic Feature Selection: Choose which columns (Age, Fare, Class, etc.) to include in the training set.
    - Algorithm Comparison: Switch between Logistic Regression, Random Forest, and SVM.
    - Performance Metrics: Real-time accuracy score output.
- Tech Stack: Streamlit, Scikit-Learn, Pandas, Jupyter.

---

### Ames Housing Price Regressor

A comprehensive data cleaning and regression tool using the Ames, Iowa housing dataset. This project demonstrates how to reformat raw data into a format suitable for Linear Regression and provides a "Live Appraisal" tool for users.

- Key Features:
    - Data Reformatting: Handles categorical encoding and missing value imputation for linear model compatibility.
    - Interactive Prediction: Users input property details (Square footage, Year Built. Neighborhood, etc.) via the UI to receive a predicted Sale Price.
- Tech Stack: Streamlit, Scikit-learn, Pandas, Jupyter.

---

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/codyickes/showcase.git
cd showcase
```

2. Install dependencies:

```bash
# To install dependencies for the Titanic App
pip install -r titanic-ml-streamlit/requirements.txt

# To install dependecies for the Ames App
pip install -r ames-housing-streamlit/requirements.txt
```

3. Run the apps:

```bash
# To run the Titanic App
streamlit run titanic-ml-streamlit/streamlit_app.py

# To run the Ames App
streamlit run ames-housing-streamlit/streamlit_app.py
```

---

## Repository Structure

```
├── ames-housing-streamlit/
│   ├── data/                    # Original and cleaned Ames dataset
│   ├── README.md
│   ├── data_processing.ipynb    # Jupyter Notebook documenting data conversion
│   ├── requirements.txt
│   └── streamlit_app.py         # Streamlit UI & Logic
├── titanic-ml-streamlit/
│   ├── data/                    # Original and cleaned Titanic dataset
│   ├── README.md
│   ├── streamlit_app.py         # Streamlit UI & Logic
│   └── requirements.txt
└── README.md
```

## About the Developer

I am a Data Analyst and Software Engineer specializing in full-stack Python development and containerized ML workflows. My work focuses on making complex data models accessible through intuitive user interfaces.
