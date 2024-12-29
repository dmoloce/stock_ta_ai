import streamlit as st
import yfinance as yf
import pandas as pd 
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os

st.set_page_config(layout="wide", page_title="AI Technical Analysis")
st.title("AI Technical Analysis Dashboard")
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Enter stock ticker: ", "AAPL").upper()
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2023-01-01"))
end_date  = st.sidebar.date_input("End date", value=pd.to_datetime("2024-12-27"))

# Fetch stock data
# if st.sidebar.button("Fetch Data"):
#     st.session_state["stock_data"] = yf.download(ticker, start=start_date, end=end_date)
#     st.success("Stock data loaded successfully!")

if st.sidebar.button("Fetch Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    st.write(data)

    # Handle potential multi-index structure in newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        # Extract the specific stock ticker data if multi-index is present
        data = data.xs(key=ticker, axis=1, level=1)  # Use `xs` to select the correct level
    
    data.index = pd.to_datetime(data.index)  # Ensure index is datetime
    data = data.dropna()  # Drop rows with missing values
    
    # Save to session state
    st.session_state["stock_data"] = data
    st.success("Stock data loaded successfully!")

# Check if data is available
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    # Plot candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"  # Replace "trace 0" with "Candlestick"
        )
    ])

    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
        default=["20-Day SMA"]
    )

    def add_indicator(indicator):
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

    for indicator in indicators:
        add_indicator(indicator)

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    #llama vvv
    st.subheader("AI Analysis")
    if st.button("Run AI Analysis"):
        with st.spinner("analyzing chart, please wait..."):
            #save chart to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.write_image(tmpfile.name)
                tmpfile_path = tmpfile.name
            #read image 
            with open(tmpfile_path,"rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            #prepare the AI
            messages = [{
                'role': 'user',
                'content': """you are a stock trader specialized in technical analysis at a top financial institution.
                              Analyze the stock chart's technical indicators and provide a buy/sell/hold recommendation.
                              Base your recommendation only on the candlestick chart and the displayed technical indicators.
                              First provide the recommendation and then the detailed reasoning.""",
                'images': [image_data]
            }]
            response = ollama.chat(model='llama3.2-vision',messages=messages)
            
            #print the response
            st.write("**AI analysis results:**")
            st.write(response["message"]["content"])

            #cleanup
            os.remove(tmpfile_path)
