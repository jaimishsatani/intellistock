import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

class PricingAgent:
    def __init__(self, df):
        self.df = df
        self.model_path = "models/pricing_model.pkl"

    def train_model(self):
        # ✅ Updated feature list
        X = self.df[[ 
            "Price", 
            "Competitor Prices", 
            "Discounts",
            "Storage Cost", 
            "Elasticity Index",
            "Customer Reviews", 
            "Return Rate (%)"
        ]]
        y = self.df["Sales Volume"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model = HistGradientBoostingRegressor()
        self.model.fit(X_train, y_train)

    def save_model(self):
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def recommend_price(self, input_df):
        return self.model.predict(input_df)
