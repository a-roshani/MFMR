import numpy as np
import math
import statistics
from tensorflow import keras

def lstm(dataset,  TIME_STEPS, h , val_spl,unit ,batch_size,vb,Epoch,Lr):
  Epoch = int(Epoch)
  vb = int(vb)
  b_size = int(batch_size)
  unit = int(unit)
  prediction = int(h)
  TIME_STEPS = int(TIME_STEPS)
  def create_dataset(X,look_back=1):
    dataX, dataY = [], []
    for i in range(len(X)-look_back):
	     a = X[i:(i+look_back)]
	     dataX.append(a)
	     dataY.append(X[i + look_back])
    return np.array(dataX), np.array(dataY)
  # reshape to [samples, time_steps, n_features]
  X_train, y_train = create_dataset(dataset,TIME_STEPS)
  X_train = X_train.reshape((X_train.shape[0],TIME_STEPS,1))
  model = keras.Sequential()
  model.add(
    keras.layers.LSTM(
      units=unit,
      input_shape=(X_train.shape[1], X_train.shape[2]),
      activation="sigmoid",
      stateful=False ,
    )
  )
  #model.add(keras.layers.Dropout(rate=0.1))
  model.add(keras.layers.Dense(units=1))
  opt = keras.optimizers.Adam(learning_rate=Lr)
  model.compile(loss='mean_squared_error', optimizer=opt)

  history = model.fit(
    X_train, y_train,
    epochs=Epoch,
    batch_size=b_size,
    verbose=vb,
    validation_split=val_spl,
    shuffle=False
  )
  
  x_input = np.array(dataset[-TIME_STEPS:])
  temp_input = list(x_input)
  lst_output=[]
  i=0
  while(i<prediction):
    if(len(temp_input)>TIME_STEPS):
        x_input=np.array(temp_input[1:])
        x_input = x_input.reshape((1, TIME_STEPS, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        lst_output.append(yhat[0][0])
        i=i+1
    else:
        x_input = x_input.reshape((1, TIME_STEPS, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i=i+1
  dta = dataset+lst_output  
  return dta, history
