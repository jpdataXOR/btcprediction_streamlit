import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_bitcoin_data(interval):
    btc = yf.Ticker("BTC-USD")
    
    if interval == "1h":
        period = "730d"
    elif interval == "1d":
        period = "max"
    
    data = btc.history(period=period, interval=interval)
    data.index = data.index.tz_localize(None)
    data['Change'] = data['Close'].pct_change()
    return data

def convert_to_pattern(series):
    return ''.join(['U' if x > 0 else 'D' for x in series])

def find_matching_patterns(data, current_pattern, lookforward=5):
    pattern_length = len(current_pattern)
    matches = pd.DataFrame()
    
    for i in range(pattern_length, len(data) - lookforward):
        window_pattern = convert_to_pattern(data['Close'].pct_change().iloc[i-pattern_length:i])
        
        if window_pattern == current_pattern:
            future_changes = data['Close'].iloc[i:i+lookforward].pct_change()
            future_pattern = convert_to_pattern(future_changes)
            
            match_info = {
                'match_date': data.index[i-1],
                'pattern': window_pattern,
                'future_pattern': future_pattern,
                'base_price': data['Close'].iloc[i-1],
                'future_changes': [x for x in future_changes.iloc[1:]]
            }
            matches = pd.concat([matches, pd.DataFrame([match_info])], ignore_index=True)
    
    return matches

def calculate_future_prices(recent_price, future_changes):
    future_prices = []
    current_price = recent_price
    
    for change in future_changes:
        current_price = current_price * (1 + change)
        future_prices.append(current_price)
    
    return future_prices

def plot_with_predictions(data, matches, current_pattern_length, recent_price, lookforward=5):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot main price line
    ax.plot(data['Close'], label='Bitcoin Price', color='black', linewidth=1)
    
    # Calculate and plot future price levels for each match
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for idx, match in matches.iterrows():
        future_prices = calculate_future_prices(recent_price, match['future_changes'])
        
        for i, price in enumerate(future_prices):
            color = colors[idx % len(colors)]
            ax.axhline(y=price, 
                      color=color, 
                      linestyle='--', 
                      alpha=0.5,
                      label=f'Match {idx+1} Day {i+1}' if i == 0 else "")
    
    ax.set_title('Bitcoin Price with Pattern-Based Predictions', pad=20)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig

def main():
    st.title("Bitcoin Pattern Matcher")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    interval = st.sidebar.selectbox("Select Interval", ["1h", "1d"])
    pattern_length = st.sidebar.slider("Pattern Length", 3, 10, 5)
    lookforward = st.sidebar.slider("Prediction Days", 1, 10, 5)
    
    # Fetch data
    data = fetch_bitcoin_data(interval)
    recent_data = data.tail(pattern_length)
    current_pattern = convert_to_pattern(recent_data['Change'])
    
    # Create tabs
    tab1, tab2 = st.tabs(["Chart View", "Detailed Data"])
    
    with tab1:
        # Display current pattern
        st.markdown(f"**Current Pattern:** `{'→'.join(current_pattern)}`  (U = Up, D = Down)")
        
        # Find matches and display chart
        matches = find_matching_patterns(data, current_pattern, lookforward)
        
        if not matches.empty:
            st.markdown(f"**Found {len(matches)} matching patterns**")
            
            # Show patterns only
            for idx, match in matches.iterrows():
                st.markdown(f"Match {idx+1}: Future moves: `{'→'.join(match['future_pattern'])}`")
            
            # Plot with predictions
            fig = plot_with_predictions(data.tail(200), matches, pattern_length, 
                                      recent_data['Close'].iloc[-1], lookforward)
            st.pyplot(fig)
        else:
            st.warning("No matching patterns found in historical data.")
    
    with tab2:
        st.write("### Recent Prices")
        st.dataframe(recent_data[['Close', 'Change']].round(4))
        
        if not matches.empty:
            st.write("### Pattern Matches Details")
            for idx, match in matches.iterrows():
                st.write(f"**Match {idx+1}** (Found at: {match['match_date'].strftime('%Y-%m-%d')})")
                future_prices = calculate_future_prices(recent_data['Close'].iloc[-1], match['future_changes'])
                price_df = pd.DataFrame({
                    'Day': range(1, len(future_prices) + 1),
                    'Predicted Price': future_prices,
                    'Change %': [x * 100 for x in match['future_changes']]
                }).round(2)
                st.dataframe(price_df)

if __name__ == "__main__":
    main()