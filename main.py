# streamlit_app.py

import streamlit as st
import yfinance as yf
from openai import OpenAI
import plotly.graph_objects as go
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# File name for storing user preferences
PREFERENCES_FILE = 'user_preferences.json'

# Function to save user preferences to a file
def save_preferences(ticker):
    preferences = {'ticker': ticker}
    with open(PREFERENCES_FILE, 'w') as f:
        json.dump(preferences, f)

# Function to load user preferences from a file
def load_preferences():
    if os.path.exists(PREFERENCES_FILE):
        with open(PREFERENCES_FILE, 'r') as f:
            preferences = json.load(f)
        return preferences.get('ticker', '')
    else:
        return ''

# Function to fetch stock data for a given ticker symbol
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        if data.empty:
            raise ValueError("No data found. Please check the ticker symbol.")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Base class for technical indicators
class TechnicalIndicator:
    def __init__(self, data):
        self.data = data

# Class for calculating Moving Average
class MovingAverage(TechnicalIndicator):
    def calculate(self, window=14):
        return self.data['Close'].rolling(window=window).mean()

# Class for calculating Relative Strength Index (RSI)
class RSI(TechnicalIndicator):
    def calculate(self, window=14):
        delta = self.data['Close'].diff(1)
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# Function to add Moving Average to the data
def add_moving_average(data, window=14):
    ma = MovingAverage(data)
    data[f'MA_{window}'] = ma.calculate(window)
    return data

# Function to add RSI to the data
def add_rsi(data, window=14):
    rsi_indicator = RSI(data)
    data[f'RSI_{window}'] = rsi_indicator.calculate(window)
    return data

# Function to get explanations from the AI model
def get_indicator_explanation(indicator_name):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains financial indicators."},
                {"role": "user", "content": f"Explain what the {indicator_name} indicates in stock trading."}
            ],
            max_tokens=150
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        st.error(f"Error getting AI explanation: {e}")
        return None

# Function to get insights from the AI model based on indicator values
def get_indicator_insights(ticker, indicator_name, value):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides insights on stock indicators."},
                {"role": "user", "content": f"The current {indicator_name} for {ticker} is {value:.2f}. What does this suggest about the stock's potential movement?"}
            ],
            max_tokens=150
        )
        insight = response.choices[0].message.content.strip()
        return insight
    except Exception as e:
        st.error(f"Error getting AI insights: {e}")
        return None

# Function to plot stock data and indicators
def plot_data(data, ticker):
    fig = go.Figure()
    # Add candlestick chart for stock prices
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    # Add Moving Average to the chart
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[f'MA_14'],
        line=dict(color='blue', width=1),
        name='MA 14'
    ))
    # Update chart layout
    fig.update_layout(
        title=f"{ticker} Price Chart with Moving Average",
        xaxis_title='Date',
        yaxis_title='Price'
    )
    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Plot RSI using Streamlit's line_chart
    st.subheader("RSI Over Time")
    st.line_chart(data[f'RSI_14'])

def main():
    st.title("AI-Powered Stock Technical Indicator Analyzer")

    # Load the saved ticker symbol
    saved_ticker = load_preferences()

    # Create a text input for the user to enter a ticker symbol
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", value=saved_ticker)

    # When the user clicks the "Analyze" button
    if st.button("Analyze"):
        # Save the user's ticker preference
        save_preferences(ticker)

        # Fetch stock data
        data = fetch_stock_data(ticker)
        if data is not None:
            # Calculate technical indicators
            data = add_moving_average(data)
            data = add_rsi(data)

            # Plot the stock data and indicators
            plot_data(data, ticker)

            # Get explanations for the indicators using AI
            ma_explanation = get_indicator_explanation("Moving Average")
            rsi_explanation = get_indicator_explanation("RSI")

            # Display the explanations
            st.subheader("Indicator Explanations")
            if ma_explanation:
                st.write(f"**Moving Average:** {ma_explanation}")
            if rsi_explanation:
                st.write(f"**RSI:** {rsi_explanation}")

            # Get the latest values of the indicators
            latest_ma = data[f'MA_14'].iloc[-1]
            latest_rsi = data[f'RSI_14'].iloc[-1]

            # Get AI-generated insights based on the latest values
            ma_insight = get_indicator_insights(ticker, "Moving Average", latest_ma)
            rsi_insight = get_indicator_insights(ticker, "RSI", latest_rsi)

            # Display the insights
            st.subheader("AI Insights")
            if ma_insight:
                st.write(f"**Moving Average Insight:** {ma_insight}")
            if rsi_insight:
                st.write(f"**RSI Insight:** {rsi_insight}")

if __name__ == "__main__":
    main()