# PaddleX Time Series Forecasting Task Data Annotation Tutorial

Data for time series forecasting tasks does not require annotation. Simply collect real-world data and arrange it in a csv file in chronological order. During training, the data will be automatically segmented into multiple time slices to form training samples, as shown in the figure below. The historical time series data and future sequences represent the input data for training the model and its corresponding prediction targets, respectively. To ensure data quality and completeness, missing values can be filled based on expert experience or statistical methods.

![alt text](/tmp/images/data_prepare/time_series/01.png)