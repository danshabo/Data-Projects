import pandas as pd
import numpy as np
import functions as f
import warnings
warnings.filterwarnings("ignore")
def logic_low_close_vs_p25(agg_df, raw_df):
    """
    Logic 1: Flaag tickers where the last close is less than the 25th percentile of historical closes.
    """
    p25_series = raw_df.groupby('Ticker')['Close'].quantile(0.10).rename('p25')
    df = agg_df.merge(p25_series, on='Ticker', how='left')
    df['logic_1'] = df['last_close'] < df['p25']
    return df[['Ticker', 'logic_1']]

def logic_high_volatility(agg_df, threshold=0.1):
    """
    Logic 2: Flag tickers where normalized_std is above a given threshold.
    Default is 0.1 (can be replaced with quantile-based threshold if needed).
    """
    df = agg_df.copy()
    df['logic_2'] = df['normalized_std'] >= threshold
    return df[['Ticker', 'logic_2']]


def logic_global_min_in_last_interval(raw_df, interval_days=10):
    """
    Logic 3: Flag tickers where the global minimum close price occurred in the last interval,
    but not on the last day of data.
    
    Parameters:
    raw_df (pd.DataFrame): DataFrame with columns 'Ticker', 'Date', 'Close'
    interval_days (int): Number of days per interval
    
    Returns:
    pd.DataFrame: DataFrame with columns 'Ticker', 'logic_3'
    """
    # Convert Date to datetime if it's not already
    raw_df = raw_df.copy()
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    
    # Initialize results dictionary
    results = {}
    
    for ticker, group in raw_df.groupby('Ticker'):
        # Sort by date
        group = group.sort_values('Date').reset_index(drop=True)
        
        # Calculate intervals
        group['Interval'] = group.index // interval_days
        
        # Get the last interval
        last_interval = group['Interval'].max()
        
        # Get global minimum
        global_min_close = group['Close'].min()
        
        # Find where the global minimum occurred
        min_close_rows = group[group['Close'] == global_min_close]
        
        # Check if any global minimum is in the last interval but not on the last day
        last_interval_mins = min_close_rows[min_close_rows['Interval'] == last_interval]
        
        if not last_interval_mins.empty:
            # Get the last date in the dataset
            last_date = group['Date'].max()
            
            # Check if all minimum instances in the last interval are not on the last day
            min_not_on_last_day = not any(last_interval_mins['Date'] == last_date)
            
            # Logic 3 is True if global min is in last interval but not on last day
            results[ticker] = min_not_on_last_day
        else:
            # Global minimum is not in the last interval
            results[ticker] = False
    
    # Convert results to DataFrame
    result_df = pd.DataFrame({
        'Ticker': list(results.keys()),
        'logic_3': list(results.values())
    })
    
    return result_df

def logic_min_below_median_by_std(agg_df):
    """
    Logic 4: Flag tickers where the global min_close is at least 2 standard deviations below the median.
    """
    df = agg_df.copy()
    df['logic_4'] = df['min_close'] < (df['median_close'] - 3 * df['std_close'])
    return df[['Ticker', 'logic_4']]

def logic5_support_bounce_reversal(raw_df):
    results = []
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    four_months_ago = raw_df['Date'].max() - pd.DateOffset(months=4)
    for ticker, df in raw_df.groupby('Ticker'):
        df = df[df['Date'] >= four_months_ago].sort_values('Date')
        supports = df.loc[(df['Close'].shift(1) > df['Close']) & (df['Close'].shift(-1) > df['Close'])]
        if len(supports) < 2:
            results.append({'Ticker': ticker, 'logic_5': False})
            continue
        support_level = supports['Close'].median()
        recent = df.tail(2)
        hammer_mask = (
            (recent['High'] - recent['Low'] > 2 * (recent['Open'] - recent['Close'])) &
            ((recent['Close'] - recent['Low']) / (0.001 + recent['High'] - recent['Low']) < 0.25)
        )
        green_mask = recent['Close'] > recent['Open']

        # Specifically: previous candle is hammer, last candle is green (and not a hammer)
        hammer_found = bool(hammer_mask.iloc[0])
        green_found = bool(green_mask.iloc[1])
        valid_pattern = hammer_found and green_found and not hammer_mask.iloc[1]
        volume_increasing = recent['Volume'].iloc[-1] > recent['Volume'].iloc[-2]
        near_support = abs(recent['Close'].iloc[-1] - support_level) / support_level < 0.02
        flag = (valid_pattern and volume_increasing and near_support)
        results.append({'Ticker': ticker, 'logic_5': flag})
        # print("results",results)
        # print("support_level", support_level)
        # print("green_found", green_found)
        # print("hammer_found")
        # print("near_support", near_support)
        # print(volume_increasing,volume_increasing)
        # print(f"Ticker: {ticker}, Logic 5: {flag}")
        # print(f"Ticker: {ticker}, Support Level: {support_level}")
    return pd.DataFrame(results)

def is_bullish_engulfing(raw_df):
    results = []
    for ticker, df in raw_df.groupby('Ticker'):
        df = df.sort_values('Date').reset_index(drop=True)
        if len(df) < 2:
            flag = False
        else:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            prev_red = prev['Close'] < prev['Open']
            last_green = last['Close'] > last['Open']
            engulf = (last['Open'] < prev['Close']) and (last['Close'] > prev['Open'])
            flag = prev_red and last_green and engulf
        results.append({'Ticker': ticker, 'logic_5': flag})
    return pd.DataFrame(results)

def is_hammer(raw_df):
    results = []
    for ticker, df in raw_df.groupby('Ticker'):
        df = df.sort_values('Date').reset_index(drop=True)
        if len(df) < 1:
            flag = False
        else:
            last = df.iloc[-1]
            body = abs(last['Close'] - last['Open'])
            range_ = last['High'] - last['Low']
            lower_shadow = min(last['Close'], last['Open']) - last['Low']
            flag = (body < range_ * 0.3) and (lower_shadow > body * 2) and (lower_shadow > (range_ * 0.4))
        results.append({'Ticker': ticker, 'logic_6': flag})
    return pd.DataFrame(results)

def is_doji_open_higher(raw_df):
    results = []
    for ticker, df in raw_df.groupby('Ticker'):
        df = df.sort_values('Date').reset_index(drop=True)
        if len(df) < 2:
            flag = False
        else:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            body = abs(last['Close'] - last['Open'])
            range_ = last['High'] - last['Low']
            doji = (body < range_ * 0.1)
            flag = doji and (last['Open'] > prev['Open'])
        results.append({'Ticker': ticker, 'logic_7': flag})
    return pd.DataFrame(results)
def logic_earnings_positive_estimate(agg_df, api_key, days_ahead=7):
    earnings_df = f.get_weekly_earnings_calendar(api_key=api_key, days_ahead=days_ahead)
    merged = agg_df.merge(earnings_df, left_on='Ticker', right_on='symbol', how='left')
    merged['logic_8'] = merged['epsEstimate'].apply(lambda x: pd.notnull(x) and x > 0)
    return merged[['Ticker', 'logic_8']]

API_KEY ='d0f1obhr01qssausu6igd0f1obhr01qssausu6j0' #finhub


def logic_ma_cross(raw_df, short_window=7, long_window=21, method='sma'):
    """
    Flags tickers where the short moving average crosses above the long moving average on the last day.
    """
    results = []
    for ticker, df in raw_df.groupby('Ticker'):
        df = df.sort_values('Date').copy()
        short_ma = f.moving_average(df['Close'], short_window, method)
        long_ma = f.moving_average(df['Close'], long_window, method)
        # Cross up: yesterday short_ma < long_ma, today short_ma >= long_ma
        cross = (short_ma.iloc[-2] < long_ma.iloc[-2]) and (short_ma.iloc[-1] >= long_ma.iloc[-1])
        results.append({'Ticker': ticker, 'logic_9': cross})
    return pd.DataFrame(results)

def logic_rsi(raw_df, window=14):
    """
    Flags tickers where RSI is below 30 (oversold) or crossed up from below 30 to above 40.
    """
    results = []
    for ticker, df in raw_df.groupby('Ticker'):
        df = df.sort_values('Date').copy()
        rsi = f.calc_rsi(df['Close'], window)
        # Case 1: RSI < 30 on last day
        case1 = rsi.iloc[-1] < 30
        # Case 2: RSI crossed from below 30 to above 40
        case2 = (rsi.iloc[-2] < 30) and (rsi.iloc[-1] >= 40)
        results.append({'Ticker': ticker, 'logic_rsi': case1 or case2})
    return pd.DataFrame(results)

def logic_mean_reversion(raw_df, window=10, z_thresh=-2):
    """
    Flags tickers where the z-score of the last close vs moving average (window) is less than z_thresh.
    z_score = (close - MA_window) / std_window
    """
    results = []
    for ticker, df in raw_df.groupby('Ticker'):
        df = df.sort_values('Date').copy()
        ma = df['Close'].rolling(window=window, min_periods=1).mean()
        std = df['Close'].rolling(window=window, min_periods=1).std()
        z_score = (df['Close'] - ma) / (std + 1e-9)
        flag = z_score.iloc[-1] < z_thresh
        results.append({'Ticker': ticker, 'logic_10': flag})
    return pd.DataFrame(results)


def logic_ma_strict_uptrend(raw_df, window=3, intervals=4):
    """
    Flags tickers where the moving average (window) is strictly increasing at each interval split.
    The data is split into X equal intervals (intervals param), and the MA at the end of each interval
    must be higher than the previous one.
    """
    results = []
    for ticker, df in raw_df.groupby('Ticker'):
        df = df.sort_values('Date').copy()
        ma = df['Close'].rolling(window=window, min_periods=1).mean()
        n = len(ma)
        if n < intervals + 1:
            results.append({'Ticker': ticker, 'logic_ma_strict_uptrend': False})
            continue
        # Find split points for intervals (including last point)
        idxs = np.linspace(0, n - 1, intervals + 1, dtype=int)
        ma_points = ma.iloc[idxs].values
        is_strict_uptrend = np.all(np.diff(ma_points) > 0)
        results.append({'Ticker': ticker, 'logic_11': is_strict_uptrend})
    return pd.DataFrame(results)

def logic_volume_spike_zscore(raw_df, intervals=5, z_thresh=2.5):
    """
    Flags tickers where there is a sudden spike in trading volume, measured by z-score.
    For each ticker, computes the z-score of volume for each day.
    Flags True if the max z-score in the last interval exceeds z_thresh.

    Parameters:
        raw_df (pd.DataFrame): DataFrame with columns ['Ticker', 'Date', 'Volume', ...]
        intervals (int): Number of equal time intervals to split the data
        z_thresh (float): Z-score threshold to flag a spike

    Returns:
        pd.DataFrame: DataFrame with columns ['Ticker', 'logic_volume_spike']
    """
    results = []
    for ticker, df in raw_df.groupby('Ticker'):
        df = df.sort_values('Date').copy()
        vols = df['Volume']
        mean = vols.mean()
        std = vols.std(ddof=0) + 1e-9  # avoid division by zero
        zscores = (vols - mean) / std
        n = len(df)
        if n < intervals:
            results.append({'Ticker': ticker, 'logic_volume_spike': False})
            continue
        # Find interval split points
        idxs = np.linspace(0, n, intervals + 1, dtype=int)
        # Focus on last interval
        last_start, last_end = idxs[-2], idxs[-1]
        last_interval_z = zscores.iloc[last_start:last_end]
        # Check if any z-score in last interval exceeds threshold
        spike = (last_interval_z > z_thresh).any()
        results.append({'Ticker': ticker, 'logic_12': spike})
    return pd.DataFrame(results)

def rank_tickers(agg_df, raw_df, interval_days=10):
    df = agg_df.copy()
    # Run logic functions
    logic_results = {
        # 'logic_1': l.logic_low_close_vs_p25(df, raw_df),
        'logic_3': logic_global_min_in_last_interval(raw_df, interval_days),
        'logic_5': is_bullish_engulfing(raw_df),
        'logic_6': is_hammer(raw_df),
        'logic_7': is_doji_open_higher(raw_df),
        'logic_8': logic_earnings_positive_estimate(agg_df, api_key=API_KEY, days_ahead=7),
        'logic_9': logic_ma_cross(raw_df, short_window=7, long_window=21, method='sma'),
        'logic_10': logic_mean_reversion(raw_df, window=21, z_thresh=-2),
        'logic_11':  logic_ma_strict_uptrend(raw_df, window=17, intervals=5),
        'logic_12':  logic_volume_spike_zscore(raw_df, intervals=15, z_thresh=2.5),
    }

    # Merge all logic results
    for logic_name, logic_df in logic_results.items():
        df = df.merge(logic_df, on='Ticker', how='left')

    logic_groups = {
        'value':     {
            'logic_10': 2.0  # העלאת משקל ל-mean reversion
        },
        'momentum':  {
            'logic_3': 1.0,
            'logic_12': 1.0,
            'logic_8': 1.0
        },
        'candlestick': {
            'logic_5': 1.0,
            'logic_6': 0.8,
            'logic_7': 0.8
        },
        'trend':     {
            'logic_9': 2.0,   # העלאת משקל לחציית ממוצעים
            'logic_11': 2.0   # העלאת משקל למגמת עלייה
        }
    }
    group_scores = {}
    group_caps = {
        'value': 2.5,
        'momentum': 2.5,
        'candlestick': 2.0,
        'trend': 2.5
    }

    # Compute capped group scores
    for group, logic_weight_map in logic_groups.items():
        score_col = f"{group}_score"
        df[score_col] = 0
        for logic_name, weight in logic_weight_map.items():
            df[score_col] += df[logic_name].fillna(0) * weight
        df[score_col] = df[score_col].clip(upper=group_caps[group])
        group_scores[group] = score_col

    # Count number of groups with any hit for diversity bonus
    df['active_groups'] = df[[col for col in group_scores.values()]].gt(0).sum(axis=1)
    df['diversity_bonus'] = np.where(df['active_groups'] >= 2, 1, 0)

    # Final score
    df['final_rank'] = df[[col for col in group_scores.values()]].sum(axis=1) + df['diversity_bonus']

    df.sort_values(by='final_rank', ascending=False, inplace=True)

    return df[
    (df['active_groups'] == 3) |
    (df['final_rank'] >= 4)
]