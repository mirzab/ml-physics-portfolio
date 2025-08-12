import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(filepath):
    data = pd.read_csv(filepath)
    print(f"Data shape: {data.shape}")
    return data

def preprocess_data(df):
    # Example preprocessing: fill missing values and encode categoricals
    df = df.fillna(df.median())
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

def main():
    data_path = '../data/house_prices.csv'  # Adjust path to dataset
    df = load_data(data_path)

    df = preprocess_data(df)

    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, '../models/house_price_model.pkl')
    print("Model saved to ../models/house_price_model.pkl")

if __name__ == '__main__':
    main()
