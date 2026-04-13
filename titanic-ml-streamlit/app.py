import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Information about the application
st.title("Titanic ML Testing Tool")
st.markdown("""
The Titanic ML Testing Tool is a Streamlit application that allows users to
try different supervised learning algorithms from scikit-learn on Kaggle's
[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
datasets.
""")

st.divider()

# Read the dataset into a dataframe
data = pd.read_csv("data/titanic_data.csv")

# Process the data and add more features
# Get title from Name (Mrs, Miss, Mr, Master)
data["Title"] = data["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
# Get deck from Cabin
data["Deck"] = data["Cabin"].fillna("Unknown").apply(lambda x: x[0])
# Get number of passengers under each ticket
data["TicketGroupSize"] = (
    data
    .groupby(by="Ticket")["Ticket"]
    .transform("count")
)
# Drop columns that won't be possible features
data.drop(
    columns=[
        "Unnamed: 0",
        "Name",
        "Sex",
        "Ticket",
        "Cabin",
        "Pclass_1",
        "Pclass_2",
        "Pclass_3",
    ],
    inplace=True,
)

# Display the head of the training dataset in a dataframe
st.markdown("#### Dataset sample")
st.dataframe(data.head(), hide_index=True)

# Split into training and testing features and labels
X = data.drop(columns="Survived")
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    train_size=0.8,
)

# Get selected features from user
features = st.segmented_control(
    "Select features to train on",
    X_train.columns,
    selection_mode="multi",
    default=X_train.columns,
)

# Expander containing information about the features
with st.expander("Show feature dictionary"):
    st.dataframe(
        pd.DataFrame(data={
            "Variable": [
                "PassengerId",
                "Survived",
                "Pclass",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
                "Sex_binary",
                "Title",
                "Deck",
                "TicketGroupSize",
            ],
            "Definition": [
                "Unique identifier for passengers",
                "Survival",
                "Ticket class",
                "Age in years",
                "# of siblings / spouses aboard the Titanic",
                "# of parents / children aboard the Titanic",
                "Passenger fare",
                "Port of Embarkation",
                "Sex",
                "Courtesy title",
                "Deck level",
                "Number of passengers under one ticket",
            ],
            "Key": [
                None,
                "0 = No, 1 = Yes",
                "1 = 1st, 2 = 2nd, 3 = 3rd",
                None,
                None,
                None,
                None,
                "C = Cherbourg, Q = Queenstown, S = Southampton",
                "0 = Male, 1 = Female",
                None,
                None,
                None,
            ],
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption("""
##### Variable Notes

**Pclass**: A proxy for socio-economic status (SES)

- 1st = Upper
- 2nd = Middle
- 3rd = Lower

**Age**: Age is fractional if less than 1. If the age is estimated, it is in
the form of xx.5

**SibSp**: The dataset defines family relations in this way...

- Sibling = brother, sister, stepbrother, stepsister
- Spouse = husband, wife (mistresses and fiancés were ignored)

**Parch**: The dataset defines family relations in this way...

- Parent = mother, father
- Child = daughter, son, stepdaughter, stepson

Some children traveled only with a nanny, therefore Parch=0 for them.
    """)

# Set up columns for radio and metric
left_col, right_col = st.columns(2, vertical_alignment="center")

# Create dict of model names -> model objects
model_dict = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}
# Get model instance from user selection
model = left_col.radio(
    "Select a model",
    model_dict.keys(),
)
model = model_dict[model]

# Set up One-Hot encoding for necessary features
one_hot_columns = [f for f in ("Embarked", "Title", "Deck") if f in features]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), one_hot_columns),
    ],
    remainder="passthrough",
)

if st.button("Train"):
    # Ensure one or more features are selected
    if not features:
        st.error("You must select at least one feature")
    else:
        with st.spinner("Training and making predictions..."):
            # Transform X_train and X_test with One-Hot encoding
            X_train = preprocessor.fit_transform(X_train[features])
            X_test = preprocessor.transform(X_test[features])
            # Train model on training data
            model.fit(X_train, y_train)
            # Use model to make predictions on testing data
            y_pred = model.predict(X_test)
        # Calculate and display model accuracy
        accuracy = accuracy_score(y_test, y_pred)
        right_col.metric("Model Accuracy", accuracy, format="percent")
