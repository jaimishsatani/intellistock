from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import datetime
from agents.demand_agent import DemandAgent
from agents.inventory_agent import InventoryAgent
from agents.pricing_agent import PricingAgent
from agents.supplier_agent import SupplierAgent
from agents.customer_feedback_agent import CustomerFeedbackAgent
import plotly.express as px

st.set_page_config(page_title="📊 IntelliStock Dashboard", layout="wide")
st.title("📊 IntelliStock Multi-Agent Inventory System")

# Load data
demand_path = 'data/demand_forecasting.csv'
inventory_df = pd.read_csv('data/inventory_monitoring.csv')
pricing_df = pd.read_csv('data/pricing_optimization.csv')

# ✅ Demand Agent
d_agent = DemandAgent(demand_path)
d_agent.preprocess()
try:
    d_agent.load_model()
except Exception as e:
    st.warning("📊 Demand model not found or incompatible. Retraining...")
    d_agent.train_model()
    d_agent.save_model()

# ✅ Inventory Agent
i_agent = InventoryAgent(inventory_df)
i_agent.preprocess()
try:
    i_agent.load_model()
except Exception as e:
    st.warning("📦 Inventory model not found or incompatible. Retraining...")
    i_agent.train_model()
    i_agent.save_model()
i_agent.features = [
    "Stock Levels",
    "Supplier Lead Time (days)",
    "Stockout Frequency",
    "Reorder Point",
    "Warehouse Capacity",
    "Order Fulfillment Time (days)",
    "Days Until Expiry"
]

# ✅ Pricing Agent
p_agent = PricingAgent(pricing_df)
try:
    p_agent.load_model()
except Exception as e:
    st.warning("💰 Pricing model not found or incompatible. Retraining...")
    p_agent.train_model()
    p_agent.save_model()

# Other agents
s_agent = SupplierAgent()
c_agent = CustomerFeedbackAgent()

# === Sidebar Navigation ===
page = st.sidebar.radio("Choose a Section", [
    "📈 Demand Forecasting",
    "📦 Inventory Monitoring",
    "💰 Pricing Suggestions",
    "🚚 Supplier Insights",
    "💬 Customer Sentiment",
    "📋 Overview Dashboard"
])

# === Load the rest of the page logic ===
# ⬇️ Paste the rest of your working interface below this line (you already have this!)
# To keep the message short, I won't re-paste all 1000+ lines unless you want.




# 📈 Demand Forecasting Page
if page == "📈 Demand Forecasting":
    st.subheader("📈 Demand Forecasting (XGBoost)")
    raw_df = pd.read_csv(demand_path)
    store_ids = sorted(raw_df["Store ID"].astype(str).unique())
    product_ids = sorted(raw_df["Product ID"].astype(str).unique())
    promo_options = ["Yes", "No"]
    season_options = ["High", "Low", "None"]
    external_options = sorted(raw_df["External Factors"].dropna().unique())
    trend_options = sorted(raw_df["Demand Trend"].dropna().unique())
    segment_options = sorted(raw_df["Customer Segments"].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        store_id = st.selectbox("🏪 Store ID", store_ids)
        product_id = st.selectbox("📦 Product ID", product_ids)
        promo = st.selectbox("🎯 Promotion Active", promo_options)
        price = st.number_input("💰 Product Price", min_value=1.0, value=100.0, step=1.0)
        ext = st.selectbox("🌐 External Factors", external_options)
    with col2:
        season = st.selectbox("📅 Seasonality", season_options)
        trend = st.selectbox("📊 Demand Trend", trend_options)
        segment = st.selectbox("👥 Customer Segment", segment_options)
        selected_date = st.date_input("📆 Forecast Date", datetime.date.today())
        day = selected_date.timetuple().tm_yday

    def build_input_dataframe():
        base = pd.DataFrame([0], columns=["dummy"]).drop(columns=["dummy"])
        base[f"Store ID_{store_id}"] = 1
        base[f"Product ID_{product_id}"] = 1
        base[f"Promotions_{promo}"] = 1
        base[f"Seasonality Factors_{season}"] = 1
        base[f"External Factors_{ext}"] = 1
        base[f"Demand Trend_{trend}"] = 1
        base[f"Customer Segments_{segment}"] = 1
        base["Day"] = day
        base["Price"] = price
        for col in d_agent.X.columns:
            if col not in base.columns:
                base[col] = 0
        return base[d_agent.X.columns]

    if st.button("🚀 Predict Demand"):
        st.markdown("---")
        custom_input = build_input_dataframe()
        pred = d_agent.forecast(custom_input)
        st.metric("📦 Forecasted Sales Quantity", f"{int(pred[0])} units")

        try:
            with open("models/demand_mse.txt") as f:
                mse_val = float(f.read())
                st.caption(f"📉 Model MSE: `{round(mse_val, 2)}`")
        except:
            st.warning("ℹ️ MSE not available. Please run main.py.")

# 📦 Inventory Monitoring Page
elif page == "📦 Inventory Monitoring":
    st.subheader("📦 Inventory Stockout Risk (GradientBoostingClassifier)")
    st.markdown("🔍 Enter inventory details to predict the **risk of stockout**.")

    col1, col2 = st.columns(2)
    with col1:
        stock_level = st.number_input("📦 Current Stock Level", 0, 1000, 50)
        reorder_point = st.number_input("📉 Reorder Point", 0, 1000, 100)
        stockout_freq = st.slider("🔁 Stockout Frequency", 0, 30, 5)
        lead_time = st.slider("⏱️ Supplier Lead Time (days)", 1, 30, 7)
    with col2:
        capacity = st.slider("🏢 Warehouse Capacity", 100, 5000, 1000)
        fulfillment_time = st.slider("🚚 Order Fulfillment Time (days)", 1, 30, 4)
        expiry_date = st.date_input("📆 Expiry Date", datetime.date.today() + datetime.timedelta(days=60))

    days_until_expiry = (expiry_date - datetime.date.today()).days

    user_input = pd.DataFrame([{
        "Stock Levels": stock_level,
        "Supplier Lead Time (days)": lead_time,
        "Stockout Frequency": stockout_freq,
        "Reorder Point": reorder_point,
        "Warehouse Capacity": capacity,
        "Order Fulfillment Time (days)": fulfillment_time,
        "Days Until Expiry": days_until_expiry
    }])

    if st.button("🚨 Predict Stockout Risk"):
        risk = i_agent.predict_risk(user_input)
        label = "⚠️ High Stockout Risk" if risk[0] else "✅ Stock Level is Safe"
        st.success(f"Prediction: {label}")

elif page == "💰 Pricing Suggestions":
    st.subheader("💰 Pricing Impact on Sales (HistGradientBoosting)")
    st.markdown("Enter pricing-related features to predict **expected sales volume**.")

    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("💰 Product Price", min_value=1.0, value=100.0, step=1.0)
        competitor_price = st.number_input("🏷️ Competitor Prices", min_value=1.0, value=95.0, step=1.0)
        discount = st.slider("🔻 Discounts (%)", 0, 100, 10)
        storage_cost = st.number_input("📦 Storage Cost", min_value=0.0, value=5.0, step=0.5)
    with col2:
        elasticity = st.slider("📉 Elasticity Index", 0.0, 5.0, 1.0, step=0.1)
        customer_reviews = st.slider("⭐ Avg Customer Review (1-5)", 1, 5, 4)
        return_rate = st.slider("📦 Return Rate (%)", 0.0, 100.0, 10.0, step=0.5)

    if st.button("📊 Predict Sales Volume"):
        input_data = pd.DataFrame([{
            "Price": price,
            "Competitor Prices": competitor_price,
            "Discounts": discount,
            "Storage Cost": storage_cost,
            "Elasticity Index": elasticity,
            "Customer Reviews": customer_reviews,
            "Return Rate (%)": return_rate
        }])
        sales_pred = p_agent.recommend_price(input_data)
        st.success(f"📈 Predicted Sales Volume: {round(sales_pred[0])} units")

# 🚚 Supplier Insights Page
elif page == "🚚 Supplier Insights":
    st.subheader("🚚 Restock Evaluation Agent")

    product_id = st.number_input("📦 Product ID", value=1001, key="supplier_product_id")
    stock = st.slider("📊 Current Stock Level", 0, 200, 45, key="supplier_stock")
    reorder = st.slider("📉 Reorder Point", 10, 200, 100, key="supplier_reorder")
    lead_time = st.slider("⏱️ Supplier Lead Time (Days)", 1, 30, 5, key="supplier_lead")
    forecasted_demand = st.number_input("📈 Forecasted Demand (optional)", min_value=0, value=80, key="supplier_demand")

    if st.button("📬 Evaluate Restock"):
        # Core evaluation
        msg = s_agent.evaluate_order(product_id, stock, reorder, lead_time, forecasted_demand)
        st.success(msg)

        # Calculate and show suggested reorder and ETA
        suggested_order_qty = max(0, forecasted_demand - stock)
        eta_date = datetime.date.today() + datetime.timedelta(days=lead_time)

        # Display
        st.markdown(f"### 📦 Suggested Reorder Quantity: `{suggested_order_qty}` units")
        st.markdown(f"### 📅 Expected Restock Date: `{eta_date.strftime('%B %d, %Y')}`")

        # Supplier Info
        st.markdown("### 🧾 Supplier Info")
        st.json(s_agent.get_supplier_info(product_id))

# 💬 Customer Sentiment Page
elif page == "💬 Customer Sentiment":
    st.subheader("💬 Customer Feedback Analyzer (TextBlob)")
    st.markdown("Use this tool to detect customer satisfaction from review text.")

    review = st.text_area("✍️ Enter Customer Review:")

    if st.button("🧠 Analyze Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a valid review before analyzing.")
        else:
            sentiment = c_agent.analyze_sentiment(review)

            if sentiment == "Positive":
                st.success(f"🟢 Sentiment Detected: {sentiment} – Thank you for your feedback!")
            elif sentiment == "Negative":
                st.error(f"🔴 Sentiment Detected: {sentiment} – We'll work to improve!")
            else:
                st.info(f"🟡 Sentiment Detected: {sentiment} – Thanks for sharing.")

# 📋 Overview Dashboard Page
elif page == "📋 Overview Dashboard":
    st.subheader("📋 IntelliStock Summary Dashboard")

    col1, col2 = st.columns(2)

    # 🔢 Total Products
    total_products = inventory_df["Product ID"].nunique() if "Product ID" in inventory_df else len(inventory_df)
    col1.metric("🔢 Total Products", total_products)

    # ⚠️ High Stockout Risk Items
    try:
        inventory_df["Days Until Expiry"] = pd.to_datetime(inventory_df["Expiry Date"]) - pd.Timestamp.today()
        inventory_df["Days Until Expiry"] = inventory_df["Days Until Expiry"].dt.days.clip(lower=0)
        sample_X = inventory_df[i_agent.features]
        risk_preds = i_agent.predict_risk(sample_X)
        high_risk_count = sum(risk_preds)
        col2.metric("⚠️ High Stockout Risk Items", high_risk_count)
    except Exception as e:
        col2.error("❌ Inventory risk prediction failed.")
        st.exception(e)

    st.markdown("---")

    # 💰 Predicted Sales Volume (sample)
    try:
        sample_input = pd.DataFrame([{
            "Price": 100,
            "Competitor Prices": 90,
            "Discounts": 10,
            "Storage Cost": 5,
            "Elasticity Index": 1.5,
            "Customer Reviews": 4,
            "Return Rate (%)": 5.0
        }])
        sales_volume = p_agent.recommend_price(sample_input)[0]
        st.metric("💰 Predicted Sales Volume", f"{round(sales_volume)} units")
    except Exception as e:
        st.warning("⚠️ Could not calculate predicted sales volume.")
        st.exception(e)

    st.markdown("---")

    # 😊 Average Sentiment Score (Optional)
    try:
        feedbacks = pricing_df["Customer Reviews"] if "Customer Reviews" in pricing_df else []
        if len(feedbacks) > 0:
            avg_score = sum(feedbacks) / len(feedbacks)
            st.metric("😊 Avg. Customer Rating", f"{round(avg_score, 2)} / 5")
            st.progress(min(int(avg_score * 20), 100))
        else:
            st.info("No customer reviews available in dataset.")
    except Exception as e:
        st.warning("⚠️ Could not calculate sentiment score.")
        st.exception(e)

    st.markdown("---")

    # 📤 Export to PDF
    def generate_inventory_pdf(df):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        text = c.beginText(40, height - 40)
        text.setFont("Helvetica", 12)
        text.textLine("📋 IntelliStock - Inventory Report")
        text.textLine(" ")

        cols = list(df.columns)
        text.textLine(" | ".join(cols))
        text.textLine("-" * 110)

        for i, row in df.iterrows():
            line = " | ".join(str(row[col]) for col in cols)
            text.textLine(line)
            if i >= 25:
                text.textLine("... (truncated)")
                break

        c.drawText(text)
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    if st.button("📤 Download PDF Inventory Report"):
        pdf_bytes = generate_inventory_pdf(inventory_df)
        st.download_button(
            label="📄 Click to Download PDF",
            data=pdf_bytes,
            file_name="Inventory_Report.pdf",
            mime="application/pdf"
        )

    st.markdown("---")

    # 📊 Stock Health Pie Chart
    try:
        labels = ['High Risk', 'Safe']
        values = [high_risk_count, total_products - high_risk_count]
        fig = px.pie(names=labels, values=values, title="📦 Stock Health Overview")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("📊 Could not generate stock health chart.")
        st.exception(e)

    st.markdown("---")

    # 📊 Sales Volume by Discount Level (Grouped by bins for clarity)
    try:
        if "Discounts" in pricing_df.columns and "Sales Volume" in pricing_df.columns:
            bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%',
                      '25-30%', '30-35%', '35-40%', '40-45%', '45-50%']
            pricing_df['Discount Range'] = pd.cut(pricing_df['Discounts'], bins=bins, labels=labels, right=False)
            grouped = pricing_df.groupby('Discount Range', as_index=False)['Sales Volume'].mean()

            fig = px.bar(
                grouped,
                x='Discount Range',
                y='Sales Volume',
                title='📊 Avg Sales Volume by Discount Range',
                labels={'Sales Volume': 'Avg Sales', 'Discount Range': 'Discount Range (%)'},
                color='Sales Volume',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("Could not create grouped discount chart.")
        st.exception(e)
