import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.callbacks import EarlyStopping,Callback
import matplotlib.pyplot as plt
from Predictor import Predictor
import threading
import queue

class EarlyStoppingByUnderVal(Callback):
    '''
    Class to stop model's training earlier if the value we monitor(monitor) goes under a threshold (value)
    replace usual callbacks functions from keras 
    '''
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            #warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
            print("error")

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stop")
                      
class Thread_train_model(threading.Thread):
    
    def __init__(self,model,q,x_train,y_train,nb_epochs,nb_batch,name_ts,name=''):
        threading.Thread.__init__(self)
        self.name=name
        self.model=model
        self.x_train=x_train
        self.y_train=y_train
        self.nb_epochs=nb_epochs
        self.nb_batch=nb_batch
        self.name_ts=name_ts
        self._stopevent = threading.Event()
        self.q=q
        print("The new thread with the name : " + self.name + " start running")

    def run(self):
        model = self.train_model(self.model,self.x_train,self.y_train,self.nb_epochs,self.nb_batch,self.name_ts)
        self.q.put(model)
        self.stop()
        print("le thread " + self.name + " ended ")
        return 0

    def stop(self):
        self._stopevent.set()

    def train_model(self,model,x_train,y_train,nb_epochs,nb_batch,name):
        '''   
        Train the model and save it in the right file
        
        Parameters
        ----------
        model : Sequential object
            model.
        x_train : array
            training data inputs.
        y_train : array
            training data ouputs.
        nb_epochs : int.
            nb of training repetitions.
        nb_batch : int
            size of batch of element which gonna enter in the model before doing a back propagation.
        trend : bool
            Distinguish trend signal from others (more complicated to modelise).
        
        Returns
        -------
        model : Sequential object
            model.

        '''
        if name=="trend" :
            
            nb_epochs=nb_epochs*7
            x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])            	
            es = EarlyStopping(monitor='mse', mode='min', min_delta=0.01 ,patience = 200)
            hist=model.fit(x_train,y_train,epochs=nb_epochs,batch_size=nb_batch,verbose=2,callbacks=[es])
            i=0
            while hist.history["mse"][-1] > 2 and i <5:     #####################################§changer
                i=i+1
                epochs=50
                hist=model.fit(x_train,y_train,epochs=epochs,batch_size=100,verbose=0,callbacks=[es])
            print("model_trained")

        elif  name=="residual" : 
            nb_epochs=nb_epochs*2
            x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])         	
            es = EarlyStopping(monitor='loss', mode='min', min_delta=0.1 ,patience = 200)
            hist=model.fit(x_train,y_train,epochs=nb_epochs,batch_size=nb_batch,verbose=2,callbacks=[es])
            i=0

        else :
            es=EarlyStoppingByUnderVal(monitor="mse", value=0.0000005, verbose=1)
            x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])
            model.fit(x_train,y_train,epochs=nb_epochs,batch_size=nb_batch,verbose=2)#,callbacks=[es])
            print("model_trained")         
        return model

            
class New_Predictor(Predictor):
    
    def __init__(self,df,host,measurement,look_back,nb_layers,loss,metric,nb_features,optimizer,nb_epochs,nb_batch,form,freq_period) :
        Predictor.__init__(self)
        self.df=df
        self.host=host
        self.measurement=measurement
        self.form=form
        self.freq_period=freq_period
        trend_x, trend_y,seasonal_x,seasonal_y,residual_x,residual_y=self.prepare_data(df,look_back,self.freq_period)
        model_trend=self.make_models(nb_layers,loss,metric,nb_features,optimizer,True)
        model_seasonal=self.make_models(nb_layers,loss,metric,nb_features,optimizer,False)
        model_residual=self.make_models(nb_layers,loss,metric,nb_features,optimizer,False)
        que = queue.Queue()
        threads_list = list()
        thread = Thread_train_model(model_trend,que,trend_x,trend_y,nb_epochs,nb_batch,"trend","Trend Thread")
        thread.start()
        threads_list.append(thread)
        thread_1 = Thread_train_model(model_seasonal,que,seasonal_x,seasonal_y,nb_epochs,nb_batch,"seasonal","Seasonal Thread")
        thread_1.start()
        threads_list.append(thread_1)
        thread_2= Thread_train_model(model_residual,que,residual_x,residual_y,nb_epochs,nb_batch,"residual","Residual Thread")
        thread_2.start()
        threads_list.append(thread_2)
        for t in threads_list:
            t.join()
        self.model_trend=que.get(block=False)
        self.model_save(self.model_trend,"trend")
        self.model_seasonal=que.get(block=False)
        self.model_save(self.model_seasonal,"seasonal")
        self.model_residual=que.get(block=False)
        self.model_save(self.model_residual,"residual")

    
    def make_models(self,nb_layers,loss,metric,nb_features,optimizer,trend) :
        '''   
        Create an LSTM model depending on the parameters selected by the user 
        
        Parameters
        ----------
        nb_layers : int
            nb of layers of the lstm model.
        loss : str
            loss of the model.
        metric : str
            metric to evaluate the model.
        nb_features : int
            size of the ouput (one for regression).
        optimizer : str
            gradient descend's optimizer.
        trend : bool
              Distinguish trend signal from others (more complicated to modelise).
       
        Returns
        -------
        model : Sequential object
            model
        '''
        if trend : 
            nb_layers=int(nb_layers/1)
        model=Sequential()
        model.add(LSTM(nb_layers,return_sequences=True,activation='relu',input_shape=(nb_features,self.look_back)))
        model.add(Dropout(0.2))
        model.add(LSTM(nb_layers))
        model.add(Dropout(0.2))
        model.add(Dense(int(nb_layers/2),activation='relu'))
        model.add(Dense(1))
        model.compile(loss=loss,optimizer=optimizer,metrics=['mse'])
        print("model_made")
        return model
              
    def prediction_eval(prediction,real_data) :
        '''  
        This functino compute and print four differents metrics (mse ,mae ,r2 and median) to evaluate accuracy of the model
        prediction and real_data need to have the same size
        
        Parameters
        ----------
        prediction : array
            predicted values.
        real_data : array
            real data.
        
        Returns
        -------
        None.
        
        '''
        from sklearn.metrics import mean_absolute_error as mae 
        from sklearn.metrics import mean_squared_error as mse 
        from sklearn.metrics import median_absolute_error as medae
        from sklearn.metrics import r2_score as r2
        
        print("mean_absolute_error : ",mae(real_data,prediction))
        print("mean_squared_error : ",mse(real_data,prediction))
        print("median_absolute_error : ",medae(real_data,prediction))
        print("r2_score : ",r2(real_data,prediction))
        

    
    def plot_training(self,model,df_a):
        '''  
        Print one step ahead preidctidn during the training period ,  not working yet
        
        Parameters
        ----------
        model : Sequential object
            model
        df_a : dataframe
            dataframe of historical data before any processing.
            
        Returns
        -------
        None.

        '''
        if model=="trend":
            model=self.model_trend
            x_train,y_train= decoupe_dataframe(self.trend, self.look_back)
        elif model=="residual":
            model=self.model_residual
            x_train,y_train =decoupe_dataframe(self.residual, self.look_back)
        elif model=="seasonal":
            model=self.model_seasonal
            x_train,y_train =decoupe_dataframe(self.seasonal, self.look_back)
        
        x_train = np.reshape(x_train,(int(len(x_train)/self.look_back), 1,self.look_back))
        train_predict=model.predict(x_train)
        plt.plot(train_predict)
        plt.plot(df_a[: -self.look_back]) #plot tout df_a si on souhaite le mettre en prod 
        plt.show()
        
    

    
