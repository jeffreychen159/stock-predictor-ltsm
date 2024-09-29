# Stock Price and Volume Prediction Using LSTM

This project implements a Long Short-Term Memory (LSTM) neural network model to predict future stock prices. The input consists of past closing prices and it uses this information to predict the next day's closing price. 

## Project Overview

This project uses a time series forecasting technique to attempt to predict close prices of specific stocks. By utilizing a deep learning architecture, specifically LSTM layers, it was able to learn patterns in historical stock data and attempt to predict future stock prices. 

This model takes in the date, and the historical data over a window of days, consisting of price, and generates a prediction of the close price for the next day. 

## Data Preparation

1. **Raw Data:** Using yfinance, I was able to get the historical price data of a stock and use that as the input to my code. The date column and the close price were extracted, preparing it for windowing.
2. **Windowing:** Data is windowed, where a window of n days is used as the input for the model.
3. **Splitting the Data:** The data is split into three sets where about 80% is used for training, 15% is used for validating, and 5% is used for testing. This structure allows the model to learn from the training data and apply the pattern learned from the training data onto the validation and testing data. 

## Model Architecture

The neural network is composed of multiple LSTM layers which are connected with Dense layers that further process the sequential output. 

The key components of the model architecture are: 
- Input state: shape=(9, 1) is used to represent the 9 windowed dates and 1 element we are looking at, price.
- Dense Layers: Connecting dense layers after each LSTM layer allows the model to extract patterns from the LSTM outputs.
- Final Output: A final dense layer with 1 neuron predicts the close price.

While the model trained using these components, it was also important to ensure that the model didn't over-train. To ensure that the model was trained optimally, hyperparameters were used such as: 
- Early Stopping
- Learning Rate Schedulers
- Changing Epochs
- Batch Size manipulation

Using these hyperparameters ensures that the model is trained enough to be able to predict future patterns while ensuring that it doesn't overfit the training data. 

## Conclusion

In this project, I have developed a stock predictor model using LSTM. By leveraging the power of LSTMs and their ability to find patterns, the model could somewhat predict near-future stock prices but not far-out stock prices as the predictions always seemed to just converge. Given the inherent variables when it comes to stock predictions and the fact that price alone is not an indicator of future price, this model would need to include a lot more variables to better predict prices. In the future, I hope to include more variables where the model can not just learn from the price history itself, but many other market and human conditions that cause market changes.
