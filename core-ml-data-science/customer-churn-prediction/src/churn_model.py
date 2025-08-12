import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def load_data(filepath):
    """Load CSV data into DataFrame."""
    data = pd.read_csv(filepath)
    print(f"Data shape: {data.shape}")
    return data

def preprocess_data(df):
    """Basic preprocessing: handle categorical, missing data."""
    # Example: drop customer ID column if exists
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Convert 'Yes'/'No' to 1/0 for binary columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Fill missing values for TotalCharges (if any)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # One-hot encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def main():
    data_path = '../data/churn_data.csv'  # Adjust to your dataset path
    df = load_data(data_path)
    df_processed = preprocess_data(df)

    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save the model
    joblib.dump(model, '../models/churn_model.pkl')
    print("Model saved to ../models/churn_model.pkl")

if __name__ == '__main__':
    main()
