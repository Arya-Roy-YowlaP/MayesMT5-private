import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Export historical data from MetaTrader 5')
    
    # Required arguments
    parser.add_argument('--symbol', type=str, default='EURUSD',
                      help='Trading symbol (default: EURUSD)')
    parser.add_argument('--timeframes', type=str, nargs='+', 
                      default=['M5', 'H1', 'D1'],
                      help='Timeframes to download (M1, M5, M15, M30, H1, H4, D1) (default: M5 H1 D1)')
    
    # Date range arguments
    parser.add_argument('--start-date', type=str,
                      help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str,
                      help='End date in YYYY-MM-DD format (default: current time)')
    
    # Optional arguments
    parser.add_argument('--count', type=int,
                      help='Number of candles to download (if specified, overrides date range)')
    parser.add_argument('--output-dir', type=str, default='data',
                      help='Directory to save the CSV files (default: data)')
    
    return parser.parse_args()

def get_timeframe_enum(timeframe_str):
    # Remove any non-alphanumeric characters and convert to uppercase
    timeframe_str = ''.join(c for c in timeframe_str.upper() if c.isalnum())
    
    timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    
    if timeframe_str not in timeframes:
        print(f"Warning: Invalid timeframe {timeframe_str}. Using M1 as default.")
        return mt5.TIMEFRAME_M1
        
    return timeframes[timeframe_str]

def export_data(symbol, timeframe, filename, start_date=None, end_date=None, count=None):
    try:
        if count is not None:
            # Download specific number of candles
            rates = mt5.copy_rates_from(symbol, timeframe, start_date or datetime.now(), count)
        else:
            # Download entire date range
            if not start_date:
                start_date = datetime(2020, 1, 1)  # Default start date if none provided
            if not end_date:
                end_date = datetime.now()
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
        if rates is None or len(rates) == 0:
            print(f"Error: No data received for {symbol} timeframe {timeframe}")
            return
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(df)} candles from {df['time'].min()} to {df['time'].max()}")
        
    except Exception as e:
        print(f"Error downloading data for {symbol} timeframe {timeframe}: {e}")
        print(f"Error details: {str(e)}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize MT5
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert dates if provided
    start_date = None
    end_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print("Error: Invalid start date format. Use YYYY-MM-DD")
            mt5.shutdown()
            return
            
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print("Error: Invalid end date format. Use YYYY-MM-DD")
            mt5.shutdown()
            return
    
    # Export data for each timeframe
    for tf in args.timeframes:
        timeframe = get_timeframe_enum(tf)
        filename = os.path.join(args.output_dir, f"{args.symbol}_{tf.lower()}.csv")
        export_data(args.symbol, timeframe, filename, start_date, end_date, args.count)
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
