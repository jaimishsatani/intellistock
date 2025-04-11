# agents/supplier_agent.py

import datetime

class SupplierAgent:
    def __init__(self):
        self.restock_log = []

    def evaluate_order(self, product_id, stock_level, reorder_point, lead_time, forecasted_demand=None):
        restock_required = False
        suggested_qty = 0

        # 1. Calculate restock based on stock vs forecast
        if forecasted_demand is not None:
            if stock_level < forecasted_demand:
                restock_required = True
                suggested_qty = forecasted_demand - stock_level
        # 2. Fallback to traditional threshold logic
        elif stock_level < reorder_point:
            restock_required = True
            suggested_qty = reorder_point - stock_level

        if restock_required:
            restock_entry = {
                "Product ID": product_id,
                "Restock Quantity": suggested_qty,
                "ETA (days)": lead_time,
                "Restock Date": datetime.date.today().isoformat()
            }
            self.restock_log.append(restock_entry)
            return f"🚚 Restock triggered for Product {product_id}, Quantity: {suggested_qty}, ETA {lead_time} days."

        else:
            return f"✅ Stock sufficient for Product {product_id}. No restock needed."

    def get_supplier_info(self, product_id):
        return {
            "Supplier": "ABC Supplies",
            "On-time Delivery Rate": "92%",
            "Average Lead Time": "5 days",
            "Last Restock Date": (datetime.date.today() - datetime.timedelta(days=4)).isoformat()
        }
