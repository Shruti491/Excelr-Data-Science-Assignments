import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load Data
data = pd.read_excel("Churn (1) (2).xlsx", sheet_name="Churn (1)")

# Drop unnecessary columns
data = data.drop(columns=["Unnamed: 0"], errors='ignore')

# Define Features & Target
selected_features = [
    "intl.plan", "voice.plan", "customer.calls", "day.charge", "intl.charge",
    "eve.charge", "night.charge", "day.mins", "eve.mins", "night.mins", "state", "area.code"
]
X = data[selected_features]
y = data["churn"].map({"no": 0, "yes": 1})  # Convert churn labels to 0 & 1

# Encode Categorical Columns
categorical_cols = ["intl.plan", "voice.plan", "state", "area.code"]
label_encoders = {}

for col in categorical_cols:
    X[col] = X[col].astype(str)  # Convert to string before encoding
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Ensure all numerical columns are properly converted
numeric_cols = X.select_dtypes(include=['number']).columns

# Handle Missing Values
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Handle Imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Build Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train Model
pipeline.fit(X_train, y_train)

# Streamlit App
st.title("📊 Churn Prediction App")
st.write("Enter customer details to predict churn probability.")

# User Inputs
user_inputs = {}
for col in selected_features:
    if col in categorical_cols:
        user_inputs[col] = st.selectbox(f"{col}", label_encoders[col].classes_)
    else:
        # Ensure only numeric columns are processed for mean calculation
        default_value = float(X[col].mean()) if col in numeric_cols else None
        user_inputs[col] = st.number_input(f"{col}", value=default_value)

if st.button("Predict Churn"):
    input_df = pd.DataFrame([user_inputs])

    # Convert categorical values using encoders
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])[0]

    # Apply imputation to handle any missing values
    input_df = pd.DataFrame(imputer.transform(input_df), columns=selected_features)

    # Make Prediction
    prediction = pipeline.predict(input_df)[0]
    st.write(f"### 🚀 Predicted Churn: {'Yes' if prediction == 1 else 'No'}")
