import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch Bitcoin data
def fetch_bitcoin_data(interval):
    btc = yf.Ticker("BTC-USD")
    data = btc.history(period="max", interval=interval)
    data['Change'] = data['Close'].pct_change()
    return data

# Function to convert percentage changes to a pattern of 'U' and 'D'
def convert_to_pattern(series):
    pattern = ''.join(['U' if x > 0 else 'D' for x in series])
    return pattern

# Function to find matching patterns in historical data
def find_matching_patterns(data, current_pattern):
    pattern_length = len(current_pattern)
    data['Pattern'] = np.nan

    for i in range(pattern_length, len(data)):
        pattern = convert_to_pattern(data['Change'].iloc[i-pattern_length:i])
        if pattern == current_pattern:
            data.loc[data.index[i], 'Pattern'] = pattern

    matches = data.dropna(subset=['Pattern'])
    return matches

# Function to plot the chart in black and white with horizontal overlays
def plot_with_overlays(data, matches, current_pattern_length):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data['Close'], label='Current Price', color='black')

    # Ensure there are enough data points before attempting to plot horizontal lines
    for match_date in matches.index:
        if match_date in data.index:  # Ensure the match date is within the range of data
            future_price = data['Close'].loc[match_date]
            ax.axhline(y=future_price, linestyle='--', color='gray', linewidth=0.8)

    ax.set_title('Bitcoin Price Chart with Matching Patterns (Horizontal Lines)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    return fig

# Main function to control Streamlit app
def main():
    st.title("Bitcoin Price Pattern Matcher")
    
    # Restrict interval options
    interval = st.selectbox("Select Interval", ["1h", "1d"])
    data = fetch_bitcoin_data(interval)

    # Extract recent pattern
    current_pattern_length = 5
    recent_data = data.iloc[-current_pattern_length:]
    current_pattern = convert_to_pattern(recent_data['Change'])
    
    st.write("### Current Pattern and Prices")
    st.write(f"Pattern: {current_pattern}")
    st.table(recent_data[['Close', 'Change']])

    # Find matching patterns
    matches = find_matching_patterns(data, current_pattern)

    if not matches.empty:
        st.write("### Matched Patterns")
        st.write(matches[['Close', 'Pattern']])

        # Plot matching patterns with horizontal lines
        st.write("### Chart with Matching Patterns")
        fig = plot_with_overlays(data.tail(200), matches, current_pattern_length)
        st.pyplot(fig)
    else:
        st.write("No matching patterns found.")

if __name__ == "__main__":
    main()
