import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI बाइनरी सिग्नल", page_icon="📈")
st.title("🤖 EUR/USD AI सिग्नल जनरेटर")
st.write("अशोक कुमार खिमानी के लिए निर्मित")

@st.cache_data(ttl=3600)
def train_model():
    ticker = "EURUSD=X"
    data = yf.download(ticker, start="2022-01-01", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Price_Change'] = data['Close'].diff()

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    model_data = data.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'SMA_5', 'SMA_20', 'Price_Change', 'RSI']
    X = model_data[features]
    y = model_data['Target']

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, features

model, features = train_model()

if st.button("🔄 लाइव मार्केट एनालाइज करें", type="primary"):
    live_data = yf.download("EURUSD=X", period="30d", interval="1d", progress=False)
    if isinstance(live_data.columns, pd.MultiIndex):
        live_data.columns = live_data.columns.droplevel(1)

    live_data['SMA_5'] = live_data['Close'].rolling(window=5).mean()
    live_data['SMA_20'] = live_data['Close'].rolling(window=20).mean()
    live_data['Price_Change'] = live_data['Close'].diff()

    delta_l = live_data['Close'].diff()
    gain_l = (delta_l.where(delta_l > 0, 0)).rolling(window=14).mean()
    loss_l = (-delta_l.where(delta_l < 0, 0)).rolling(window=14).mean()
    rs_l = gain_l / loss_l
    live_data['RSI'] = 100 - (100 / (1 + rs_l))

    latest_row = live_data[features].tail(1)
    prediction = model.predict(latest_row)[0]
    probability = model.predict_proba(latest_row)[0]
    
    confidence = probability[prediction] * 100
    current_close = live_data['Close'].iloc[-1]

    st.subheader(f"अंतिम भाव (Close): {current_close:.5f}")
    st.write(f"मजबूती (Confidence): {confidence:.2f}%")

    if confidence >= 58.0:
        if prediction == 1:
            st.success("🚀 STRONG CALL / UP (कैंडल ऊपर जाने की संभावना है)")
        else:
            st.error("📉 STRONG PUT / DOWN (कैंडल नीचे जाने की संभावना है)")
    else:
        st.warning("⏳ MARKET UNCLEAR - NO TRADE (बाजार स्पष्ट नहीं है, रुकें)")
