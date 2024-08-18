import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

def fetch_bitcoin_data(interval):
    btc = yf.Ticker("BTC-USD")
    data = btc.history(period="max", interval=interval)
    data['Change'] = data['Close'].pct_change()
    return data

def convert_to_pattern(series):
    pattern = ''.join(['U' if x > 0 else 'D' for x in series])
    return pattern

def find_matching_patterns(data, current_pattern):
    pattern_length = len(current_pattern)
    data['Pattern'] = np.nan

    for i in range(pattern_length, len(data)):
        pattern = convert_to_pattern(data['Change'].iloc[i-pattern_length:i])
        if pattern == current_pattern:
            data.loc[data.index[i], 'Pattern'] = pattern

    matches = data.dropna(subset=['Pattern'])
    return matches

def plot_with_overlays(data, matches, current_pattern_length, zoom_periods):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data['Close'], label='Current Price', color='blue')

    colors = plt.cm.get_cmap('tab10', len(matches))
    
    for i, match_date in enumerate(matches.index):
        matched_series = data['Close'].iloc[
            (matches.index.get_loc(match_date) - current_pattern_length):matches.index.get_loc(match_date)]
        
        if len(matched_series) == current_pattern_length:
            matched_series = matched_series / matched_series.iloc[0] * data['Close'].iloc[-1]
            ax.plot(data.index[-current_pattern_length:], matched_series.values, '--', color=colors(i))

    last_date = data.index[-1]
    ax.axvline(x=last_date, color='red', linestyle='--', label='Future Prediction Start')

    # Adjust x-axis limits based on available data and zoom_periods
    start_index = max(0, len(data) - zoom_periods)
    future_end_date = last_date + DateOffset(days=5)  # Adjust depending on the period of the data
    ax.set_xlim([data.index[start_index], future_end_date])

    min_close = min(data['Close'].min(), matched_series.min())
    max_close = max(data['Close'].max(), matched_series.max())
    ax.set_ylim(min_close * 0.95, max_close * 1.05)

    return fig

def predict_future_prices(data, matches, current_pattern_length, future_periods):
    future_values = pd.DataFrame(index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_periods), columns=['Close'])
    future_values.index.name = 'Date'
    
    if not matches.empty:
        # Calculate the median price change for each period after the matching patterns
        median_changes = []
        for match_date in matches.index:
            match_end_index = data.index.get_loc(match_date)
            if match_end_index + future_periods < len(data):
                future_slice = data.iloc[match_end_index:match_end_index+future_periods]
                changes = future_slice['Close'].pct_change().fillna(0)
                median_changes.append(changes)
        
        if median_changes:
            median_future_changes = pd.concat(median_changes, axis=1).median(axis=1)
            
            # Ensure we have exactly future_periods number of changes
            if len(median_future_changes) > future_periods:
                median_future_changes = median_future_changes[:future_periods]
            elif len(median_future_changes) < future_periods:
                # Pad with zeros if we have fewer changes than future_periods
                median_future_changes = median_future_changes.reindex(range(future_periods), fill_value=0)
            
            last_price = data['Close'].iloc[-1]
            future_prices = [last_price]
            for change in median_future_changes:
                future_prices.append(future_prices[-1] * (1 + change))
            
            future_values['Close'] = future_prices[1:]  # Exclude the last known price
        else:
            future_values['Close'] = data['Close'].iloc[-1]  # If no valid changes, assume flat price
    else:
        future_values['Close'] = data['Close'].iloc[-1]  # If no matches, assume flat price
    
    print(f"Future values shape: {future_values.shape}")
    print(f"Future values index: {future_values.index}")
    print(f"Future values data: {future_values['Close'].values}")
    
    return future_values

def plot_future_values(data, future_values):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot historical data
    ax.plot(data.index[-30:], data['Close'].iloc[-30:], label='Historical Prices', color='blue')
    
    # Plot future predictions
    ax.plot(future_values.index, future_values['Close'], label='Future Predictions', color='green', linestyle='--')
    
    # Add confidence interval (assuming 10% deviation)
    upper_bound = future_values['Close'] * 1.1
    lower_bound = future_values['Close'] * 0.9
    ax.fill_between(future_values.index, lower_bound, upper_bound, color='green', alpha=0.2)
    
    ax.set_title('Bitcoin Price: Historical and Future Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    
    # Set x-axis limits
    ax.set_xlim([data.index[-30], future_values.index[-1]])
    
    # Set y-axis limits
    all_prices = pd.concat([data['Close'].iloc[-30:], future_values['Close'], upper_bound, lower_bound])
    ax.set_ylim(all_prices.min() * 0.95, all_prices.max() * 1.05)
    
    return fig

def main():
    st.title("Bitcoin Price Pattern Matcher")
    interval = st.selectbox("Select Interval", ["15m", "1h", "1d"])
    data = fetch_bitcoin_data(interval)

    current_pattern_length = 5
    recent_data = data.iloc[-current_pattern_length:]
    current_pattern = convert_to_pattern(recent_data['Change'])
    
    st.write("### Current Pattern and Prices")
    st.write(f"Pattern: {current_pattern}")
    st.table(recent_data[['Close', 'Change']])

    display_data = data.tail(200)
    min_close = display_data['Close'].min()
    max_close = display_data['Close'].max()
    st.write(f"### Current Bitcoin Price Chart (Last 200 {interval} periods)")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(display_data.index, display_data['Close'], label='Current Price', color='blue')
    ax.set_ylim(min_close * 0.95, max_close * 1.05)
    st.pyplot(fig)

    matches = find_matching_patterns(data, current_pattern)

    if not matches.empty:
        st.write("### Past Patterns Found")
        st.write(matches)

        overlay_zoom = st.slider("Zoom in on the overlay chart (number of periods)", min_value=5, max_value=min(200, len(display_data)), value=50, step=5)
        zoomed_data = display_data.tail(overlay_zoom)

        st.write("### Price Chart with Overlayed Matching Patterns")
        fig = plot_with_overlays(zoomed_data, matches, current_pattern_length, overlay_zoom)
        st.pyplot(fig)

        # Generate future predictions
        future_periods = 10
        print(f"Data shape: {data.shape}")
        print(f"Matches shape: {matches.shape}")
        print(f"Current pattern length: {current_pattern_length}")
        print(f"Future periods: {future_periods}")
        
        future_values = predict_future_prices(data, matches, current_pattern_length, future_periods)

        # Plot future values
        st.write("### Future Price Predictions")
        fig = plot_future_values(data, future_values)
        st.pyplot(fig)

        # Display future values in a table
        st.write(future_values)
    else:
        st.write("No matching patterns found.")

if __name__ == "__main__":
    main()