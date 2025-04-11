import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

class DemandAgent:
    def __init__(self, data_path):
        self.data_path = data_path

    def preprocess(self):
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Day'] = df['Date'].dt.dayofyear
        df['Store ID'] = df['Store ID'].astype(str)
        df['Product ID'] = df['Product ID'].astype(str)
        df = pd.get_dummies(df, columns=['Store ID', 'Product ID', 'Promotions', 'Seasonality Factors'])
        self.X = df.drop(columns=['Sales Quantity', 'Date', 'Demand Trend', 'External Factors', 'Customer Segments'])
        self.y = df['Sales Quantity']

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = XGBRegressor(objective='reg:squarederror')
        self.model.fit(X_train, y_train)
        self.mse = mean_squared_error(y_test, self.model.predict(X_test))

    def save_model(self, path="models/demand_model.pkl"):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path="models/demand_model.pkl"):
        self.model = joblib.load(path)

    def forecast(self, future_X):
        return self.model.predict(future_X)
