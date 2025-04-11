import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class InventoryAgent:
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        # Target: Stockout Risk (1 if stock is below reorder point)
        self.df["Stockout Risk"] = (self.df["Stock Levels"] < self.df["Reorder Point"]).astype(int)

        # Convert date to days remaining (optional)
        self.df["Days Until Expiry"] = pd.to_datetime(self.df["Expiry Date"]) - pd.Timestamp.today()
        self.df["Days Until Expiry"] = self.df["Days Until Expiry"].dt.days.clip(lower=0)

        # Features for model
        self.features = [
            "Stock Levels",
            "Supplier Lead Time (days)",
            "Stockout Frequency",
            "Reorder Point",
            "Warehouse Capacity",
            "Order Fulfillment Time (days)",
            "Days Until Expiry"
        ]

        self.X = self.df[self.features]
        self.y = self.df["Stockout Risk"]

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        self.model = model

        # Evaluate accuracy
        y_pred = model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print("📦 Inventory Agent Accuracy:", round(self.accuracy * 100, 2), "%")
        print(classification_report(y_test, y_pred))

    def save_model(self, path="models/inventory_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path="models/inventory_model.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def predict_risk(self, new_data_df):
        return self.model.predict(new_data_df[self.features])
