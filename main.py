# ================================
# main.py with emojis
# ================================

from agents.demand_agent import DemandAgent
from agents.inventory_agent import InventoryAgent
from agents.pricing_agent import PricingAgent
from agents.supplier_agent import SupplierAgent
from agents.customer_feedback_agent import CustomerFeedbackAgent
import pandas as pd
import os

# 📂 Ensure model directory exists
os.makedirs("models", exist_ok=True)

# ================================
# 📦 Load CSVs
# ================================
demand_data = 'data/demand_forecasting.csv'
inventory_data = pd.read_csv('data/inventory_monitoring.csv')
pricing_data = pd.read_csv('data/pricing_optimization.csv')

# ================================
# 🧰 Demand Agent
# ================================
print("\U0001f4c8 Training Demand Forecasting Agent...")
d_agent = DemandAgent(demand_data)
d_agent.preprocess()
d_agent.train_model()
d_agent.save_model()

# ✅ Save MSE to file for dashboard
with open("models/demand_mse.txt", "w") as f:
    f.write(str(d_agent.mse))
print("\u2705 Demand Forecast MSE:", round(d_agent.mse, 2))

# ================================
# 📦 Inventory Agent
# ================================
print("\U0001f4e6 Training Inventory Agent...")

# Prepare features
inventory_data["Stockout Risk"] = (inventory_data["Stock Levels"] < inventory_data["Reorder Point"]).astype(int)
inventory_data["Days Until Expiry"] = pd.to_datetime(inventory_data["Expiry Date"]) - pd.Timestamp.today()
inventory_data["Days Until Expiry"] = inventory_data["Days Until Expiry"].dt.days.clip(lower=0)

i_agent = InventoryAgent(inventory_data)
i_agent.features = [
    "Stock Levels",
    "Supplier Lead Time (days)",
    "Stockout Frequency",
    "Reorder Point",
    "Warehouse Capacity",
    "Order Fulfillment Time (days)",
    "Days Until Expiry"
]
i_agent.X = inventory_data[i_agent.features]
i_agent.y = inventory_data["Stockout Risk"]
i_agent.train_model()
i_agent.save_model()
print(f"✅ Inventory Agent Accuracy: {round(i_agent.accuracy * 100, 2)}%")

# ================================
# 💰 Pricing Agent
# ================================
print("\U0001f4b0 Training Pricing Agent...")
p_agent = PricingAgent(pricing_data)
p_agent.train_model()
p_agent.save_model()
print("\u2705 Pricing Agent Trained.")

# ================================
# 🚚 Supplier Agent (Example Run)
# ================================
s_agent = SupplierAgent()
msg = s_agent.evaluate_order(101, 40, 100, 5)
print("\U0001f69a Supplier Message:", msg)

# ================================
# 💬 Customer Feedback Agent (Example Run)
# ================================
c_agent = CustomerFeedbackAgent()
feedback = "Service was great but the item was delayed."
sentiment = c_agent.analyze_sentiment(feedback)
print("\U0001f4ac Customer Sentiment:", sentiment)