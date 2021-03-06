# Competition homepage

## This kaggle competition involves using forecasting methods to predict the sales for various items over a 28 day window for multiple store locations of Walmart in the USA.

## Evaluation
Weighted Root Mean Squared Scaled Error (RMSSE) of 28-day forecasting window

## Data

The data covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.

1. It uses hierarchical sales data, generously made available by Walmart, starting at the item level and aggregating to that of departments, product categories, stores in three geographical areas of the US: California, Texas, and Wisconsin.
1. Besides the time series data, it also includes explanatory variables such as price, promotions, day of the week, and special events (e.g. Super Bowl, Valentine’s Day, and Orthodox Easter) that affect sales which are used to improve forecasting accuracy.
1. The distribution of uncertainty is being assessed by asking participants to provide information on four indicative prediction intervals and the median.
1. The majority of the more than 42,840 time series display intermittency (sporadic sales including zeros).
1. Instead of having a single competition to estimate both the point forecasts and the uncertainty distribution, there will be two parallel tracks using the same dataset, the first requiring 28 days ahead point forecasts and the second 28 days ahead probabilistic forecasts for the median and four prediction intervals (50%, 67%, 95%, and 99%).
1. For the first time, it focuses on series that display intermittency, i.e., sporadic demand including zeros.

## Files

1. calendar.csv - Contains information about the dates on which the products are sold.
1. sales_train_validation.csv - Contains the historical daily unit sales data per product and store [d_1 - d_1913]
1. sample_submission.csv - The correct format for submissions. Reference the Evaluation tab for more info.
1. sell_prices.csv - Contains information about the price of the products sold per store and date.

## Research Notes (mostly from Forecasting with R book @ https://otexts.com/fpp2/data-methods.html)

Quantitative forecasting can be applied when two conditions are satisfied:

1. Numerical information about the past is available;
1. It is reasonable to assume that some aspects of the past patterns will continue into the future.

Time series models used for forecasting include:

1. Decomposition models
1. Exponential smoothing models
1. ARIMA models

Common categories of models include:

1. Explanatory models (incorporate features that try to **explain** what causes the variation in the target
1. Times series models that predict future values based on past values of the same variable, not external variables
1. Mixed models that incorporate both. Examples: 
   1. Dynamic regression
   1. Panel data
   1. Longitudinal
   1. Transfer function
   1. Linear system models
