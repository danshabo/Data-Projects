from indices import * 
import yfinance as yf
import pandas as pd
from tqdm.notebook import tqdm
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from scipy.signal import argrelextrema
import os
import time
import random
import string
import requests
import datetime
import importlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore")



def yahoo_data():
    global yahoo_df 
    tickers = indices + global_indices
    # tickers = test
    dfs = []
    with tqdm(tickers, desc="Downloading data") as pbar:
        for ticker in pbar:
            clear_output(wait=True)
            df = yf.download(ticker, period="4mo", interval="1d")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df['Ticker'] = ticker
            dfs.append(df)
            # time.sleep(1)
    yahoo_df = pd.concat(dfs).reset_index()
    duplicate_tickers = yahoo_df['Ticker'].value_counts()
    duplicate_tickers = duplicate_tickers[duplicate_tickers > 1].index.tolist()
    yahoo_df = yahoo_df[yahoo_df['Ticker'].isin(duplicate_tickers)]
    return yahoo_df

def aggregate_ticker_stats(df):
    # Aggregate stats: min, max, median, mean, std
    aggregated = df.groupby('Ticker')['Close'].agg(
        min_close='min',
        max_close='max',
        median_close='median',
        mean_close='mean',
        std_close='std'
    ).reset_index()

    # Calculate normalized std (std / mean)
    aggregated['normalized_std'] = aggregated['std_close'] / aggregated['mean_close']

    # Get the last close and last date
    last_entries = df.sort_values('Date').groupby('Ticker').tail(1)
    last_entries = last_entries[['Ticker', 'Close', 'Date']].rename(
        columns={'Close': 'last_close', 'Date': 'last_date'}
    )

    # Merge the aggregated stats with the last entries
    result = pd.merge(aggregated, last_entries, on='Ticker')

    return result

def analyze_ticker_intervals(df, interval_days):
    """
    Splits each ticker's data into intervals of a given number of days, and computes summary statistics for each interval:
    - Local min
    - Local max
    - Normalized standard deviation
    - Variance

    Parameters:
    df (pd.DataFrame): DataFrame with columns 'Ticker', 'Date', 'Close'
    interval_days (int): Number of days per interval

    Returns:
    dict: A dictionary with tickers as keys and summary DataFrames as values
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    results = {}

    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Date').reset_index(drop=True)

        # Assign an interval number to each row
        group['Interval'] = group.index // interval_days

        # Aggregate statistics by interval
        summary = group.groupby('Interval').agg(
            start_date=('Date', 'first'),
            end_date=('Date', 'last'),
            min_close=('Close', 'min'),
            max_close=('Close', 'max'),
            std_close=('Close', 'std'),
            var_close=('Close', 'var'),
            mean_close=('Close', 'mean')
        ).reset_index()

        # Calculate normalized std
        summary['normalized_std'] = summary['std_close'] / summary['mean_close']

        results[ticker] = summary

    return results

def stats(ticker):
    """
    Plots a candlestick chart, a line plot of closing prices with local/global minima and maxima,
    a boxplot, and a volume plot for the selected ticker.
    Uses the global yahoo_df as the data source.
    """
    # נשתמש תמיד ב-yahoo_df הגלובלי
    try:
        df = yahoo_df
    except NameError:
        print("yahoo_df is not defined. Please run f.yahoo_data() first.")
        return

    ticker_df = df[df['Ticker'] == ticker].copy()
    if ticker_df.empty:
        print(f"No data available for {ticker}.")
        return

    # Convert 'Date' column to datetime and set as index
    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
    ticker_df = ticker_df.sort_values('Date')
    ticker_df.set_index('Date', inplace=True)

    # Label minima/maxima
    df_labeled = ticker_df.reset_index()
    df_labeled['min_type'] = np.nan
    local_min_idx = argrelextrema(df_labeled['Close'].values, np.less, order=1)[0]
    df_labeled.iloc[local_min_idx, df_labeled.columns.get_loc('min_type')] = 'local_min'
    global_min_idx = df_labeled['Close'].idxmin()
    df_labeled.loc[global_min_idx, 'min_type'] = 'global_min'
    global_max_idx = df_labeled['Close'].idxmax()
    df_labeled.loc[global_max_idx, 'min_type'] = 'global_max'

    # --- Plot 1: Candlestick chart ---
    mpf.plot(ticker_df, type='candle', style='yahoo', title=f'Candlestick Chart for {ticker}', ylabel='Price',block=False)

    # --- Plot 2: Closing price with minima/maxima ---
    plt.figure(figsize=(14, 6))
    plt.plot(df_labeled['Date'], df_labeled['Close'], label='Close Price', color='blue')
    local_min = df_labeled[df_labeled['min_type'] == 'local_min']
    plt.scatter(local_min['Date'], local_min['Close'], color='orange', label='Local Min', marker='v', s=100)
    global_min = df_labeled[df_labeled['min_type'] == 'global_min']
    plt.scatter(global_min['Date'], global_min['Close'], color='red', label='Global Min', marker='X', s=150)
    global_max = df_labeled[df_labeled['min_type'] == 'global_max']
    plt.scatter(global_max['Date'], global_max['Close'], color='green', label='Global Max', marker='^', s=150)
    plt.title('Close Price with Local Minima, Global Minima and Global Maximum')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    # --- Plot 3: Volume over time ---
    if 'Volume' in ticker_df.columns:
        plt.figure(figsize=(10, 3))
        plt.bar(ticker_df.index, ticker_df['Volume'], color='purple', alpha=0.6)
        plt.title(f'Volume Over Time for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show(block=False)

    # --- Print percentiles ---
    percentiles = np.percentile(ticker_df['Close'], np.arange(10, 100, 10))
    print(f"Percentiles for closing prices of {ticker}:")
    for i, perc in enumerate(range(10, 100, 10)):
        print(f"{perc}th percentile: {percentiles[i]:.2f}")

    # --- Plot 4: Boxplot of closing prices ---
    last_close = ticker_df['Close'].iloc[-1]
    plt.figure(figsize=(6, 4))
    plt.boxplot(ticker_df['Close'], vert=False)
    plt.axvline(x=last_close, color='red', linestyle='--', linewidth=2, label=f'Last Close: {last_close:.2f}')
    plt.title(f'Boxplot of Closing Prices for {ticker}')
    plt.xlabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def label_minima(df):
    df = df.copy()
    df['min_type'] = np.nan
    # זיהוי מינימום מקומי
    local_min_idx = argrelextrema(df['Close'].values, np.less, order=1)[0]
    df.iloc[local_min_idx, df.columns.get_loc('min_type')] = 'local_min'
    global_min_idx = df['Close'].idxmin()
    df.loc[global_min_idx, 'min_type'] = 'global_min'
    global_max_idx = df['Close'].idxmax()
    df.loc[global_max_idx, 'min_type'] = 'global_max'
    return df

def get_weekly_earnings_calendar(api_key, days_ahead=7, save_to_csv=False, csv_path='earnings_calendar.csv'):
    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=days_ahead)
    url = f'https://finnhub.io/api/v1/calendar/earnings?from={today}&to={end_date}&token={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        earnings = data.get("earningsCalendar", [])
        df = pd.DataFrame(earnings)
        return df
    else:
        raise Exception(f"API request failed ({response.status_code}): {response.text}")
    
def calculate_trade_profit(
    num_shares,
    buy_price,
    market,  # 'ISR' or 'USA'
    currency,  # 'ILS' or 'USD'
    sell_price=None,  # Optional sell price,
    ticker = None  # Optional ticker for identification
):
    """
    Calculates and plots net profit across a range of sell prices.
    """


    # Constants
    USD_TO_ILS = 3.7
    TAX_RATE = 0.25

    # Validate market
    if market not in ['ISR', 'USA']:
        raise ValueError("Market must be 'ISR' or 'USA'.")

    # Fee settings
    if market == 'ISR':
        fee_calc = lambda value: max(value * 0.0007, 3.0)
    else:
        fee_calc = lambda shares: max(shares * 0.01, 6.0)

    # Generate sell prices
    sell_prices = np.linspace(buy_price * 0.85, buy_price * 1.35, 100)
    net_profits = []

    for sp in sell_prices:
        total_buy = buy_price * num_shares
        total_sell = sp * num_shares

        buy_fee = fee_calc(total_buy if market == 'ISR' else num_shares)
        sell_fee = fee_calc(total_sell if market == 'ISR' else num_shares)

        if currency == 'USD':
            conversion = USD_TO_ILS
            buy_fee *= conversion
            sell_fee *= conversion
            total_buy *= conversion
            total_sell *= conversion

        gross_profit = total_sell - total_buy
        tax = gross_profit * TAX_RATE if gross_profit > 0 else 0
        net_profit = gross_profit - buy_fee - sell_fee - tax
        net_profits.append(net_profit)

    # Find break-even point
    net_profits_array = np.array(net_profits)
    zero_crossings = np.where(np.diff(np.signbit(net_profits_array)))[0]
    breakeven_price = None

    if len(zero_crossings) > 0:
        i = zero_crossings[0]
        x1, x2 = sell_prices[i], sell_prices[i + 1]
        y1, y2 = net_profits_array[i], net_profits_array[i + 1]
        breakeven_price = x1 - y1 * (x2 - x1) / (y2 - y1)

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(sell_prices, net_profits, label='Net Profit (ILS)', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', label='Break-even', linewidth=2)

    if breakeven_price is not None:
        plt.plot(breakeven_price, 0, 'go', markersize=8, label='Break-even Point')
        plt.annotate(f'Break-even\nPrice: {breakeven_price:.2f}',
                     xy=(breakeven_price, 0),
                     xytext=(breakeven_price + (buy_price * 0.05), buy_price * 0.1),
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    # Mark buy price point
    if sell_prices[0] <= buy_price <= sell_prices[-1]:
        buy_price_profit = np.interp(buy_price, sell_prices, net_profits)
        plt.plot(buy_price, buy_price_profit, 'ro', markersize=8, label='Buy Price Point')
        plt.annotate(f'Buy Price\nPrice: {buy_price:.2f}',
                     xy=(buy_price, buy_price_profit),
                     xytext=(buy_price - (buy_price * 0.05), buy_price_profit + abs(buy_price_profit * 0.5)),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

        # Draw line between buy and break-even
        if breakeven_price is not None:
            percentage_gain = ((breakeven_price - buy_price) / buy_price) * 100
            plt.plot([buy_price, breakeven_price], [buy_price_profit, 0],
                     'k--', alpha=0.6, linewidth=1.5, label='Break-even Distance')
            plt.annotate(f'{percentage_gain:.1f}%',
                         xy=((buy_price + breakeven_price) / 2, buy_price_profit / 2),
                         xytext=(buy_price, buy_price_profit * 0.75),
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

    plt.text(0.02, 0.98, f'Market: {market}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.title('Net Profit vs. Sell Price', fontsize=16)
    plt.xlabel('Sell Price', fontsize=12)
    plt.ylabel('Net Profit (ILS)', fontsize=12)
    plt.legend(fontsize=11)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)

    # Data summary
    print("\n" + "=" * 80)
    print("DETAILED TRADE DATA SUMMARY")
    print("=" * 80)

    analysis_price = sell_price or breakeven_price
    if analysis_price is None:
        print("No valid sell price or break-even point.")
        return None

    total_buy = buy_price * num_shares
    total_sell = analysis_price * num_shares

    buy_fee = fee_calc(total_buy if market == 'ISR' else num_shares)
    sell_fee = fee_calc(total_sell if market == 'ISR' else num_shares)

    if currency == 'USD':
        buy_fee *= USD_TO_ILS
        sell_fee *= USD_TO_ILS
        total_buy *= USD_TO_ILS
        total_sell *= USD_TO_ILS

    gross_profit = total_sell - total_buy
    tax = gross_profit * TAX_RATE if gross_profit > 0 else 0
    net_profit = gross_profit - buy_fee - sell_fee - tax
    revenue_pct = (gross_profit / total_buy) * 100
    net_per_share = net_profit / num_shares

    print(f"\nSTATISTICAL SUMMARY:")
    print(f"Ticker: {ticker}")
    print(f"{'-'*50}")
    print(f"Purchase Price:       {buy_price:.2f}")
    print(f"Number of Shares:     {num_shares}")
    print(f"Total Paid:           {total_buy}")
    print(f"Currency:             {currency}")
    print(f"Market:               {market}")
    print(f"Break Even Price:     {breakeven_price:.2f}" if breakeven_price else "Break Even Price:    N/A")
    print(f"Target Sell Price:    {analysis_price:.2f}")
    print(f"Gross Amount:         {gross_profit:.2f} ILS")
    print(f"Tax Paid:             {tax:.2f} ILS")
    print(f"Revenue Percentage:   {revenue_pct:.2f}%")
    print(f"Net Amount:           {net_profit:.2f} ILS")
    print(f"Normalized Amount:    {net_per_share:.2f} ILS per share")
    print(f"Total Net Profit:     {net_profit:.2f} ILS")

    # Return summary DataFrame
    return pd.DataFrame([{
        "Buy Price": buy_price,
        "Sell Price": analysis_price,
        "Total Paid (ILS)": total_buy,
        "Num Shares": num_shares,
        "Currency": currency,
        "Market": market,
        "Gross Profit (ILS)": gross_profit,
        "Tax (ILS)": tax,
        "Net Profit (ILS)": net_profit,
        "Net Profit per Share (ILS)": net_per_share,
        "Total Net Profit (ILS)": net_profit,
        "Revenue %": revenue_pct
    }])

def moving_average(series, window, method='sma'):
    """Calculate moving average (SMA or EMA) for a pandas Series."""
    if method == 'ema':
        return series.ewm(span=window, adjust=False).mean()
    else:
        return series.rolling(window=window, min_periods=1).mean()

def calc_rsi(series, window=14):
    """Calculate RSI for a pandas Series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

