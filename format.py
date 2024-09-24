import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import util

df = pd.read_csv('data.csv')

df = df[['Date', 'Close', 'Volume']]
df['Date'] = df['Date'].apply(util.str_to_datetime)

df.index = df.pop('Date')
# print(df)


plt.plot(df.index, df['Close'])

# plt.show()

# df = pd.read_csv('data.csv')

# df = df[['Datetime', 'Close']]
# df['Datetime'] = df['Datetime'].apply(util.time_to_datetime)

# df.index = df.pop('Datetime')
# print(df)


# plt.plot(df.index, df['Close'])



# Takes data and shapes sets the target, and the last three days to create a supervised training data set
def df_to_windowed_df(dataframe, first_date, last_date, n=9):

  target_date = first_date
  
  dates = []
  X, Y = [], []
  V = []

  last_time = False
  V = dataframe['Volume'][n:].to_numpy()
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)
    
    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    # print(next_date)
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Volume'] = V
  ret_df['Target'] = Y

  return ret_df

# Takes the data and reshapes it for training
def windowed_df_to_date_X_y(windowed_df): 
    df_as_np = windowed_df.to_numpy()
    
    dates = df_as_np[:, 0]
    middle = df_as_np[:, 1:-1]
    
    X = middle.reshape(len(dates), middle.shape[1], 1)
    Y = df_as_np[:, -1]
    
    return dates, X.astype(np.float32), Y.astype(np.float32)

# Make sure the last date is the last data of the data or it won't work
print("Windowing df ...")
windowed_df = df_to_windowed_df(df, datetime.datetime(1990, 1, 15), datetime.datetime(2024, 9, 20))
print(windowed_df)
print("Finished windowing")

print("Converting to date ...")
dates, X, y, = windowed_df_to_date_X_y(windowed_df)
print("Process finished")

print(dates.shape, X.shape, y.shape)


q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)
q_95 = int(len(dates) * .95)

# Data for training
dates_train, X_train, y_train = dates[:q_90], X[:q_90], y[:q_90]

# Data for validating
dates_val, X_val, y_val = dates[q_90:], X[q_90:], y[q_90:]

# Data for testing
dates_test, X_test, y_test = dates[q_95:], X[q_95:], y[q_95:]

plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

plt.legend(['Train', 'Validation', 'Test'])

# plt.show()

