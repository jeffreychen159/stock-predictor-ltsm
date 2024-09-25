import yfinance as yf
import pandas as pd
from datetime import datetime


ticker = 'ITW'
start_date = datetime(2010, 1, 1)
end_date = datetime(2024, 9, 21)

data = yf.download(ticker, start_date, end_date)

data.to_csv('data.csv')