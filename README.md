# ML_pipeline_influxdb

Exemple of real time linear prediction :

![Linear PRED](/Images/gif_linear.gif)

Exemple of LSTM + DECOMPOSITION prédiction :

![LSTM PRED](/Images/gif_lstm.gif)

Please refer to [https://www.influxdata.com/blog/how-supralog-built-an-online-incremental-machine-learning-pipeline-with-influxdb-for-capacity-planning/] for further informations.</br>

Summary
-------
-[Why decomposition ?](#why-decomposition-?)</br>
-[GSFutur object](#gsfutur-object) </br>
-[Example use case ](#example-use-case ) </br>
-[Requirements](#quick-methods-explanation ) </br>
-[Quick methods explanation ](#requirements) </br>


GSFutur object
--------------

This object simplify prediction by hidding all the paramaters for the user.
Let's focus on the most important ones : </br>
-fit()  </br>
-predict() </br>
-retrain() </br>
-prediction_eval() </br>
-load_models() </br>

Example use case 
----------------
my dataframe (df) is like below and have a 200 points seasonal pattern :</br>
"time","y"</br>
"1749-01",58.0</br>
"1749-02",62.6</br>
"1749-03",70.0</br>
...</br>
**Code for prediction 200 steps ahead**
```
from GSFutur import GSFutur
model=GSFutur()
model.fit(df,look_back=400,freq_period=200,directory="My_directory_name/")
prediction,lower,upper=model.predict(steps=200)
```
**plot**
```
plt.plot(np.array(prediction),label='pred',color="red")
plt.legend()
plt.plot()
```

 Requirements 
------------
pandas </br>
numpy </br>
statsmodels</br>
tensorflow</br>
matplotlib</br>
scipy</br>
sklearn</br>
from statsmodels.tsa.seasonal import seasonal_decompose</br>
pickle</br>


Quick methods explanation 
----------------------
**fit needs only three inputs** </br>
  -df : dataframe </br>
     with a time columns (string) and y the columns with the values to forecast. </br>
  -look_back : int </br>
     size of inputs (generaly freq_period *2 but always more than freq_period). </br>
  -freq_period : int </br>
     size in point of the seasonal pattern (ex 24 if daily seasonality for a signal sampled at 1h frequency). </br>
  -directory : str </br>
     Directory where the models are going to be saved, by default at the root (r".").</br>
     
**Once the model fitted it can by used by applying the predict function which need only two inputs**: </br>
  -steps : int</br>
    number of points you want to forecast, by default 1.</br>
  -frame : Bool</br>
    *If frame == True , compute an 95% intervalle and retruns 3 arrays* | if frame == False return an array with the predicted values </br>
    
**Retrain allows your model to do incremental learning by retraining yours models with new data :**</br>
  -df : dataframe </br>
     with a time columns (string) and y the columns with the values to forecast. </br>
  -look_back : int </br>
     size of inputs (generaly freq_period *2 but always more than freq_period). </br>
  -freq_period : int </br>
     size in point of the seasonal pattern (ex 24 if daily seasonality for a signal sampled at 1h frequency). </br>
  
 **load_models allows to reuse saved model by loading it in the class** : </br>
   -directory : str
     name of directory contaning trend.h5,seasonal.h5,residual.h5 by default (r".") ie root of project</br>
      
**prediction_eval : once the prediction made**</br>
This function compute and print four differents metrics (mse ,mae ,r2 and median) to evaluate accuracy of the model prediction and real_data need to have the same size</br>
   -prediction : array</br>
        predicted values.</br>
   -real_data : array</br>
        real data.</br>
      
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
