import pandas as pd
import time
import numpy as np
import os
import sys 
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.layers import Dense,LSTM,Dropout,Activation,RepeatVector
from tensorflow.keras.callbacks import EarlyStopping,Callback
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate, Input 
from tensorflow.keras import Model
import shutil
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import save_model,load_model
from tensorflow.keras.callbacks import EarlyStopping,Callback
import sys
from sklearn.preprocessing import PowerTransformer
import pickle
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1
    def inverse_transform(self,input_array,y=None):
        return input_array*1



def decoupe_dataframe(df,look_back):
    dataX,dataY = [],[]
    for i in range(len(df) - look_back - 1):
        a = df[i:(i + look_back)]
        dataY=dataY+[df[i+look_back]]
        dataX.append(a)
    return (np.asarray(dataX),np.asarray(dataY).flatten())

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

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
            print("error")

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def attention_block(hidden_states):
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector
            
class GSFutur():
    
    def __init__(self,verbose=0) :
        if verbose==1:
            print("Model is being created")
        else :
            pass
    
    def fit(self,df,look_back,freq_period,verbose,nb_features=1,nb_epochs=300,nb_batch=100,nb_layers=50,attention=True,loss="mape",metric="mse",optimizer="Adamax",directory=r"."):
        '''
        This function take the data , make the preprocessing steps and train the models

        Parameters
        ----------
        df : dataframe
            with a time columns (string) and y the columns with the values to forecast.
        look_back : int
            size of inputs .
        freq_period : int
            size in point of the seasonal pattern (ex 24 if daily seasonality for a signal sampled at 1h frequency).
        verbose : 0 or 1
            If 0 no information printed during the creation and training of models.
        nb_features :int, optional
             output size. The default is 1.
        nb_epochs : int, optional
             The default is 50.
        nb_batch : int, optional
            DESCRIPTION. The default is 100.
        nb_layers : int, optional
             The default is 50.
        attention : Bool, optional
            Either use or not the attention layer. The default is True.
        loss : str, optional
            loss for evaluation error between input and output . The default is "mape".
        metric : str, optional
            metrics for evaluation the training predictions. The default is "mse".
        optimizer : str, optional
            Keras optimizers. The default is "Adamax".
        directory : str, optional
            directory path which need to end by "/". The default is r".".

        Returns
        -------
        None.

        '''
        self.loss=loss
        self.metric=metric
        self.optimizer=optimizer
        self.nb_features=nb_features
        self.nb_epochs=nb_epochs
        self.nb_batch=nb_batch
        self.df=df
        self.nb_layers=nb_layers
        self.freq_period=freq_period
        self.look_back=look_back
        self.verbose=verbose
        self.directory=directory
        trend_x, trend_y,seasonal_x,seasonal_y,residual_x,residual_y=self.prepare_data(df,look_back,self.freq_period)
        if attention :
            model_trend=self.make_models_with(True,self.df)
            model_seasonal=self.make_models_with(False,self.df)
            model_residual=self.make_models_with(False,self.df)
        else :
            model_trend=self.make_models(True,self.df)
            model_seasonal=self.make_models(False,self.df)
            model_residual=self.make_models(False,self.df)
        self.model_trend=self.train_model(model_trend,trend_x,trend_y,"trend")
        self.model_seasonal=self.train_model(model_seasonal,seasonal_x,seasonal_y,"seasonal")
        self.model_residual=self.train_model(model_residual,residual_x,residual_y,"residual")  
        print("Modele fitted")

    def predict(self,steps=1,frame=True):
        prediction,lower,upper=self.make_prediction(steps)
        if frame :
            return prediction[0][12 :],lower[0][12 :],upper[0][12 :]
        else :
            return prediction[0][12 :]
    

    
    def make_models_with(self,trend,df) :
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
        i=Input(shape=(self.nb_features,self.look_back))
        x=LSTM(self.nb_layers,return_sequences=True,activation='relu',input_shape=(self.nb_features,self.look_back))(i)
        x=Activation("relu")(x)
        x=Dropout(0.2)(x)
        x=LSTM(self.nb_layers,return_sequences=True)(x)
        x=Dropout(0.2)(x)
        x=attention_block(x)
        x=Dense(int(self.nb_layers/2),activation='relu')(x)
        output=Dense(1)(x)
        model = Model(inputs=[i], outputs=[output])
        model.compile(loss=self.loss,optimizer=self.optimizer,metrics=[self.metric])
        if self.verbose == 1 :
            print(model.summary())
        return model
    
    def make_models(self,trend,df):
        model=Sequential()
        model.add(LSTM(self.nb_layers,return_sequences=True,activation='relu',input_shape=(self.nb_features,self.look_back)))
        model.add(Dropout(0.2))
        model.add(LSTM(self.nb_layers))
        model.add(Dropout(0.2))
        if (df["y"].max()-df["y"].min()) > 100 :
           model.add(Activation('softmax'))
        model.add(Dense(int(self.nb_layers/2),activation='relu'))
        model.add(Dense(1))
        model.compile(loss=self.loss,optimizer=self.optimizer,metrics=[self.metric])
        if self.verbose == 1 :
            print(model.summary())
        return model
              
    def prediction_eval(self,prediction,real_data) :
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
        
     

    def prepare_data(self,df,look_back,freq_period):
        '''  
        Parameters
        ----------
        df : DataFrame
            datafrmae contening historical data .
        look_back : int
            length entry of the model .
        
        Decompose the signal in three sub signals, trend,seasonal and residual in order to work separetly on each signal

        Returns
        -------
        trend_x : array
             values of the trend of the signal, matrix of dimention X array of dimension (1,length entry of model) X= length(dataframe)/look_back.
        trend_y : array
            vaklues to be predicted during the training
        seasonal_x : array
            same as trend_x but with the seasonal part of the signal.
        seasonal_y : array
            same as trend_y but with the seasonal part of the signal.
        residual_x : array
            same as trend_x but with the residual part of the signal.
        residual_y : array
            same as trend_y but with the residual part of the signal.
        '''
        if (df["y"].max() - df["y"].min()) > 100:
            scaler = PowerTransformer()
        else :
            scaler= IdentityTransformer()
        self.scaler2=scaler.fit(np.reshape(np.array(df["y"]), (-1, 1)))
        Y =  self.scaler2.transform(np.reshape(np.array(df["y"]), (-1, 1)))
        decomposition = seasonal_decompose(Y, period = freq_period+1)
        df.loc[:,'trend'] = decomposition.trend
        df.loc[:,'seasonal'] = decomposition.seasonal
        df.loc[:,'residual'] = decomposition.resid
        df["trend"]= df['trend']
        df["seasonal"]=df['seasonal']
        df["residual"]=df['residual']
        df_a=df
        df=df.dropna()
        self.shift=len(df_a)-len(df)
        df=df.reset_index(drop=True)
        df["trend"]= df["trend"].fillna(method="bfill")        
        self.trend=np.asarray( df.loc[:,'trend'])
        self.seasonal=np.asarray( df.loc[:,'seasonal'])
        self.residual=np.asarray( df.loc[:,'residual'])
        trend_x,trend_y=decoupe_dataframe(df["trend"], look_back)
        seasonal_x,seasonal_y=decoupe_dataframe(df["seasonal"], look_back)
        residual_x,residual_y=decoupe_dataframe(df["residual"], look_back)
        if self.verbose == 1 :
            print("prepared")
        return trend_x, trend_y,seasonal_x,seasonal_y,residual_x,residual_y   
    
    def train_model(self,model,x_train,y_train,name):
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
            nb_epochs=self.nb_epochs*3
            try :
                x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])    
            except Exception as e :
                if e :
                    print("************Not enought data in the input compared to the look back size. Put more data as input******")       	
            es = EarlyStopping(monitor='mean_squared_error', mode='min', min_delta=0.1 ,patience = 200)
            hist=model.fit(x_train,y_train,epochs=nb_epochs,batch_size=self.nb_batch,verbose=self.verbose,callbacks=[es])
            i=0
            while hist.history["loss"][-1] > 10 and i <5:    
                i=i+1
                epochs=50
                hist=model.fit(x_train,y_train,epochs=epochs,batch_size=100,verbose=self.verbose,callbacks=[es])
            print("model_trained")
            self.model_save(model,name)
        elif  name=="residual" : 
            nb_epochs=self.nb_epochs*6
            x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])         	
            es = EarlyStopping(monitor='mean_squared_error', mode='min', min_delta=0.1 ,patience = 200)
            hist=model.fit(x_train,y_train,epochs=nb_epochs,batch_size=self.nb_batch,verbose=self.verbose,callbacks=[es])
            i=0
            self.model_save(model,name)
        else :
            es=EarlyStoppingByUnderVal(monitor="mean_squared_error", value=0.005, verbose=self.verbose)
            x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])
            model.fit(x_train,y_train,epochs=self.nb_epochs,batch_size=self.nb_batch,verbose=self.verbose)
            print("model_trained")
            self.model_save(model,name)
        return model
    
    def model_save(self,model,name) :
       path=self.directory+"/"+name+".h5"
       save_model(model,path)


    def make_prediction(self,len_prediction):
        ''' 
        This function make the prediction :
            reshape data to fit with the model
            make one prediction
            crush outdated data with prediction
            repeat len_prediction times 
            
        Parameters
        ----------
        len_prediction : int
            number of value we want to predict.
        
        Returns
        -------
        prediction : array
            values predicted.
        pred_trend : array
            values trend predicted (recherches prupose).
        pred_seasonal : array
            values seasonal predicted (recherches prupose).
        pred_residual : array
            values residual predicted (recherches prupose).
        '''
        data_residual=self.residual[-1*self.look_back : ]
        data_trend=self.trend[-1*self.look_back : ]
        data_seasonal=self.seasonal[-1*self.look_back : ]
        prediction=np.zeros((1,len_prediction))
        for i in progressbar(range(len_prediction), "Computing: ", 40):
            dataset = np.reshape(data_residual,(1, 1,self.look_back))
            prediction_residual=(self.model_residual.predict(dataset))
            data_residual=np.append(data_residual[1 :],[prediction_residual]).reshape(-1,1
                                                                                      )                               
            dataset = np.reshape(data_trend, (1, 1,self.look_back))
            prediction_trend=(self.model_trend.predict(dataset))
            data_trend = np.append(data_trend[1:], [prediction_trend]).reshape(-1, 1)
            
            dataset = np.reshape(data_seasonal, ( 1, 1,self.look_back))
            prediction_seasonal=(self.model_seasonal.predict(dataset))
            data_seasonal = np.append(data_seasonal[1:], [prediction_seasonal]).reshape(-1, 1)
            
            prediction[0,i]=prediction_residual+prediction_trend+prediction_seasonal
        prediction=self.scaler2.inverse_transform(np.reshape(prediction, (-1, 1)))
        lower,upper=self.frame_prediction(np.reshape(prediction,(1,-1)))
        return np.reshape(prediction,(1,-1)),lower,upper


    def frame_prediction(self,prediction):
        '''
        This function compute the 95% confident interval by calculating the standard deviation and the mean of the residual(Gaussian like distribution)
        and return yestimated +- 1.96*CI +mean (1.96 to have 95%)

        Parameters
        ----------
        prediction : array
            array contening the perdicted vector (1,N) size.

        Returns
        -------
        lower : array
            array contening the lowers boundry values (1,N) size.
        upper : array
            array contening the upper boundry values (1,N) size.

        '''
        mae=-1*np.mean(self.scaler2.inverse_transform(np.reshape(self.residual,(-1,1))))
        std_deviation=np.std(self.scaler2.inverse_transform(np.reshape(self.residual,(-1,1))))
        sc = 1.96       #1.96 for a 95% accuracy
        margin_error = mae + sc * std_deviation
        lower = prediction - margin_error
        upper = prediction + margin_error
        return lower,upper

    def fit_predict_without_dec(self,df,look_back,freq_period,len_prediction,verbose,nb_features=1,nb_epochs=50,nb_batch=100,nb_layers=50,attention=True,loss="mape",metric="mse",optimizer="Adamax"):
        df=df.reset_index()
        i=Input(shape=(nb_features,look_back))
        x=LSTM(nb_layers,return_sequences=True,activation='relu',input_shape=(nb_features,look_back))(i)
        x=Activation("relu")(x)
        x=Dropout(0.2)(x)
        x=LSTM(nb_layers,return_sequences=True)(x)
        x=Dropout(0.2)(x)
        #x=attention_block(x)
        x=Dense(int(nb_layers/2),activation='relu')(x)
        output=Dense(1)(x)
        model = Model(inputs=[i], outputs=[output])
        model.compile(loss=loss,optimizer=optimizer,metrics=[metric])
        x_train,y_train=decoupe_dataframe(df["y"], look_back)
        nb_epochs=nb_epochs*7*8
        try :
            x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])    
        except Exception as e :
            if e ==IndexError :
                print("Not enought data in the input compared to the look back size. Put more data as input")        	
        hist=model.fit(x_train,y_train,epochs=nb_epochs,batch_size=nb_batch,verbose=verbose)
        data_residual=np.array(df["y"][-1*look_back : ])
        prediction=np.zeros((1,len_prediction))
        for i in progressbar(range(len_prediction), "Computing: ", 40):
            dataset = np.reshape(data_residual,(1, 1,look_back))
            prediction_residual=(model.predict(dataset))
            data_residual=np.append(data_residual[1 :],[prediction_residual]).reshape(-1,1)                               
            prediction[0,i]=prediction_residual
        return prediction,prediction,prediction

    def retrain(self,df,look_back,freq_period,directory,nb_features=1,nb_epochs=10,nb_batch=35) :
        self.df=df
        self.look_back=look_back
        self.freq_period=freq_period
        self.nb_epochs=nb_epochs
        trend_x, trend_y,seasonal_x,seasonal_y,residual_x,residual_y=self.prepare_data(df,look_back,freq_period)
        model_trend,model_seasonal,model_residual=self.load_models(directory)
        model_trend=self.train_model(model_trend,trend_x,trend_y,"trend")
        model_seasonal=self.train_model(model_seasonal,seasonal_x,seasonal_y,"seasonal")
        model_residual=self.train_model(model_residual,residual_x,residual_y,"residual")
        self.model_trend=model_trend     
        self.model_seasonal=model_seasonal
        self.model_residual=model_residual

    def load_models(self,directory="."):
        print(r""+directory+"/"+"trend"+".h5")
        model_trend=load_model(r""+directory+"/"+"trend"+".h5")
        model_seasonal=load_model(r""+directory+"/"+"seasonal"+".h5")
        model_residual=load_model(r""+directory+"/"+"residual"+".h5")
        print("loaded")
        self.model_trend=model_trend
        self.model_seasonal=model_seasonal
        self.model_residual=model_residual
        return model_trend,model_seasonal,model_residual

