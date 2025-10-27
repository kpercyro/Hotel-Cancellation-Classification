import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Load model and feature list
rf_model = joblib.load("random_forest_pipeline.pkl")
top_features = joblib.load("top_10_raw_features.pkl")

# Custom CSS
st.markdown("""
    <style>
        .stButton > button {
            background-color: #3475E0;
            color: white;
            border-radius: 6px;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #1f5fcc;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Hotel Booking Cancellation Predictor")
st.caption("Simulate a booking and assess its cancellation risk based on customer and booking attributes.")

# User input or bulk upload
st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio("Choose Input Type", ["Single Booking", "Bulk Upload"])

if input_mode == "Single Booking":
    user_input = {}
    col1, col2 = st.columns(2)

    with col1:
        if 'lead_time' in top_features:
            user_input['lead_time'] = st.slider("Lead Time (days)", 0, 500, 100)
        if 'total_nights' in top_features:
            user_input['total_nights'] = st.slider("Total Nights", 1, 30, 3)
        if 'booking_changes' in top_features:
            user_input['booking_changes'] = st.slider("Booking Changes", 0, 10, 0)
        if 'previous_cancellations' in top_features:
            user_input['previous_cancellations'] = st.slider("Previous Cancellations", 0, 10, 0)
        if 'adr' in top_features:
            user_input['adr'] = st.number_input("Average Daily Rate (ADR)", min_value=0.0, value=100.0)

    with col2:
        if 'total_of_special_requests' in top_features:
            user_input['total_of_special_requests'] = st.slider("Total Special Requests", 0, 5, 0)
        if 'required_car_parking_spaces' in top_features:
            user_input['required_car_parking_spaces'] = st.slider("Required Parking Spaces", 0, 3, 0)
        if 'market_segment' in top_features:
            user_input['market_segment'] = st.selectbox("Market Segment", ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Groups"])
        if 'deposit_type' in top_features:
            user_input['deposit_type'] = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
        if 'customer_type' in top_features:
            user_input['customer_type'] = st.selectbox("Customer Type", ["Transient", "Contract", "Group", "Transient-Party"])

    input_df = pd.DataFrame([user_input])

else:
    uploaded_file = st.file_uploader("Upload Booking CSV File", type=["csv"])
    input_df = None
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)

# Prediction and output
if input_df is not None and st.button("Predict Cancellation"):
    predictions = rf_model.predict(input_df)
    probs = rf_model.predict_proba(input_df)

    input_df["Prediction"] = ["Canceled" if pred == 1 else "Not Canceled" for pred in predictions]
    input_df["Cancellation Probability"] = [f"{p[1]*100:.2f}%" for p in probs]

    # Granular risk levels
    def categorize_risk(prob):
        if prob > 0.9:
            return "Very High"
        elif prob > 0.75:
            return "High"
        elif prob > 0.5:
            return "Moderate"
        elif prob > 0.25:
            return "Low"
        else:
            return "Very Low"

    risk_scores = [categorize_risk(p[1]) for p in probs]
    input_df["Risk Level"] = risk_scores

    # Estimated revenue at risk
    if "adr" in input_df.columns and "total_nights" in input_df.columns:
        input_df["Estimated Revenue"] = input_df["adr"] * input_df["total_nights"]

    st.markdown("---")
    st.subheader("Prediction Results")
    st.dataframe(input_df)

    # Show summary metrics for bulk
    if len(input_df) > 1:
        st.metric("Total Bookings", len(input_df))
        st.metric("Predicted Cancellations", sum(predictions))
    else:
        prob_canceled = probs[0][1]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_canceled * 100,
            title={'text': "Cancellation Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "crimson" if prob_canceled > 0.5 else "seagreen"}
            }
        ))
        st.plotly_chart(fig)

        # Threshold-based cancellation guidance
        # Thresholds reflect real-world decision cutoffs:
        if prob_canceled > 0.9:
            st.error("Very high cancellation risk. Take action immediately.")
        elif prob_canceled > 0.75:
            st.warning("High risk. Follow up or require confirmation.")
        elif prob_canceled > 0.5:
            st.info("Moderate risk. Consider monitoring.")
        elif prob_canceled > 0.25:
            st.success("Low risk. Booking likely to go through.")
        else:
            st.success("Very low risk. Booking is stable.")

        # Contextual feature-based insights
        if input_df.get("Estimated Revenue", 0).iloc[0] > 1000:
            st.warning("This is a high-value booking. Monitor it closely.")

        if input_df.get("lead_time", 0).iloc[0] > 300:
            st.info("Bookings with high lead time are more likely to cancel.")

        if input_df.get("total_of_special_requests", 0).iloc[0] == 0:
            st.info("No special requests could indicate low customer commitment.")

    # Download results
    csv = input_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results as CSV", csv, "bulk_prediction_results.csv", "text/csv")

# Sidebar information
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
This dashboard predicts the likelihood of a hotel booking being canceled based on lead time, customer type, and other key features.

**Model**: Random Forest  
**Data**: Hotel Booking Dataset  
**Features Used**: Top 10 predictive features  
""")
    st.markdown("[Made with Streamlit](https://streamlit.io)")


