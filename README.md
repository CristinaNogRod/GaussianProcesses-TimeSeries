# Modelling and Predicting Time Series using Gaussian Processes


We study how Gaussian Processes are able to model and predict time series data. We will use the birthday data set which contains records of the number of births per day in the U.S. We consider traditional Gaussian Process models as well as Deep Gaussian Process model. Within the former class of models we iteratively make the model more complex in order to obtain more descriptive fits of the data. Furthermore, we consider an alternative model in which we separate the data set into births that occurred on weekdays and weekends. The result of our study is a model comparison in which we find that complex models such as Deep Gaussian Processes do not always provide a better mean squared error than more traditional models. We also observe that modelling the weekend and weekday data separately produces a significant improvement in the fit. 

