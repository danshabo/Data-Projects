
import os
import sys
import subprocess
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

required = [
    "yfinance",
    "pandas",
    "tqdm",
    "matplotlib",
    "mplfinance",
    "numpy",
    "scipy",
    "requests",
    "seaborn",
    "upsetplot",
    "ipython",
    "tabulate"
]

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in required:
    mod = pkg if pkg != "mplfinance" else "mplfinance"
    install_and_import(mod)



import functions as f
import logics as l
yahoo_df = None
yahoo_df_agg = None
rank_df = None


def calculate_trade_profit(*args, **kwargs):
    f.calculate_trade_profit(*args, **kwargs)

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Welcome to Theta")
    GREEN = "\033[92m"
    RESET = "\033[0m"
    logo = f"""{GREEN}
 _   _          _        
| | | |        | |       
| |_| |__   ___| |_ __ _ 
| __| '_ \ / _ \ __/ _` |
| |_| | | |  __/ || (_| |
 \__|_| |_|\___|\__\__,_|
                         
{RESET}"""
    print(logo)
    print("Then use stats('TICKER') or calculate_trade_profit(...) as needed.")


def collect_data():
    global yahoo_df, yahoo_df_agg, rank_df
    print("Downloading data...")
    yahoo_df = f.yahoo_data()
    print("Aggregating stats...")
    yahoo_df_agg = f.aggregate_ticker_stats(yahoo_df)
    print("Ranking tickers...")
    rank_df = l.rank_tickers(yahoo_df_agg, raw_df=yahoo_df, interval_days=10)
    print("\n=== Rank Tickers Output ===")
    clear_console()
    return rank_df

def stats(ticker):
    if 'yahoo_df' not in globals() or yahoo_df is None:
        print("Please run collect_data() first.")
        return
    f.stats(ticker)



if __name__ == "__main__":
    os.system('cls')
    print("Welcome to Theta")
    GREEN = "\033[92m"
    RESET = "\033[0m"
    logo = f"""{GREEN}
 _   _          _        
| | | |        | |       
| |_| |__   ___| |_ __ _ 
| __| '_ \ / _ \ __/ _` |
| |_| | | |  __/ || (_| |
 \__|_| |_|\___|\__\__,_|
                         
{RESET}"""
    print(logo)
    print("Type collect_data() to download and process data.")
    import code
    code.interact(local=globals())