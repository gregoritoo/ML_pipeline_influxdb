# ML_pipeline_influxdb
Exemple of real time linear prediction :

![Linear PRED](/Images/gif_linear.gif)

Exemple of LSTM + DECOMPOSITION prédiction :

![LSTM PRED](/Images/gif_lstm.gif)

Please refer to [https://www.influxdata.com/blog/how-supralog-built-an-online-incremental-machine-learning-pipeline-with-influxdb-for-capacity-planning/] for further informations.</br>

Why decomposition ?
-----------------
As describe in the article above, the aim of the project is to create a module able to forecast values of severals time series that could 
deferred in nature.</br>
One of the main problem in Deep Neural Network is to tune hyper-parameters (as for example the number of neurones ...) especially for multi-step ahead predictions. </br>
Decomposing the signal allow us to apply a single model for all the time series without spending time on hyper parameters tunning. </br>
Here below the results of this pre-processing process on differents signals : </br>

![First_page_1](/Images/res_1.PNG)

![First_page_1](/Images/res_2.PNG)

![First_page_1](/Images/res_3.PNG)

For the experiments above, the same LSTM model was applied on three differents signals with the same hyper parameters. For the first two signals the accuracy is almost the same (except a one point delay for the cpu signal that appears for the LSTM + DECOMPOSITION model after one weak ahead prediction (which explain the difference of accuracy on the table below)).</br>

But for the third signal, the model without decomposition seems to reach a local minimum during the training and then the forecated values converge to the mean value while the model with decomposition is way more accurate. </br>
(the dataset of the third experiment is the Minimum Daily Temperatures Dataset available here : [https://machinelearningmastery.com/time-series-datasets-for-machine-learning/])
</br>
Here the results of the three experiments :</br>
![First_page_1](/Images/table_res.PNG)
Note :  this method also seems to disminuish the variance of the predicted values.( ie for the same dataset, the LSTM with decomposition is more likely to give the same forecasted value)
