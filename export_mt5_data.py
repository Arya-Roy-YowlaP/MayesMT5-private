import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

mt5.initialize()

def export_data(symbol, timeframe, filename, count=2000):
    rates = mt5.copy_rates_from(symbol, timeframe, datetime.now(), count)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

export_data("EURUSD", mt5.TIMEFRAME_M5, "bars_5m.csv")
export_data("EURUSD", mt5.TIMEFRAME_H1, "bars_1h.csv")
export_data("EURUSD", mt5.TIMEFRAME_D1, "bars_1d.csv")

mt5.shutdown()
