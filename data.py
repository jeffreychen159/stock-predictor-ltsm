import yfinance as yf
import pandas as pd
from datetime import datetime

def generate_csv(ticker, start_date, end_date): 
    data = yf.download(ticker, start_date, end_date)
    data.to_csv('data.csv')
    
    return exit(0)
    
if __name__ == "__main__":
    itw = 'ITW'
    start = datetime(1970, 1, 1)
    end = datetime(2024, 9, 21)

    generate_csv(itw, start, end)