import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go  # For gauge chart

# Load trained model and encoder
model = joblib.load("sales_predictor.pkl")
encoder = joblib.load("encoder.pkl")

# Streamlit UI
st.title("📊 Sales Prediction for Marketing Campaigns")
st.write("Enter the store details to predict sales.")

# User Inputs
marketsize = st.selectbox("Market Size", ["Small", "Medium", "Large"])
promotion = st.number_input("Promotion Level", min_value=1, max_value=3, step=1)
ageofstore = st.number_input("Age of Store", min_value=1,max_value=28, step=1)
week = st.number_input("Week", min_value=1, max_value=4, step=1)

# Predict Button
if st.button("Predict Sales"):
    # Prepare Input Data
    input_df = pd.DataFrame([[marketsize, promotion, ageofstore, week]], columns=["marketsize", "promotion", "ageofstore", "week"])

    # Encode Categorical Data
    encoded_input = encoder.transform(input_df[['marketsize', 'promotion', 'week']])
    encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out())

    # Merge with Other Features
    input_df = input_df.drop(['marketsize', 'promotion', 'week'], axis=1)
    input_df = pd.concat([input_df, encoded_df], axis=1)

    # Predict Sales
    prediction = model.predict(input_df)

    # Display Prediction
    st.success(f"💰 Predicted Sales: **{prediction[0]:,.2f} Thousands**")

    # 📟 Gauge Chart
    st.subheader("📟 Predicted Sales Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction[0],
        title={'text': "Predicted Sales (Thousands)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge)
