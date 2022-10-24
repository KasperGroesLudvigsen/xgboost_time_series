# Repo for the article "Multi-step time series forecasting with XGBoost"
This is the repo for the Towards Data Science article titled "Multi-step time series forecasting with XGBoost"

The article shows how to use an XGBoost model wrapped in sklearn's MultiOutputRegressor to produce forecasts
on a forecast horizon larger than 1. 

The approach shown in the article generally follows the approach described in the paper ["Do we really need deep learning models for time series forecasting?"](https://arxiv.org/abs/2101.02118).

The main code is found in the notebook. Some util functions are implemented in utils.py.