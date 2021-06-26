import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm

class ModelLSTM:
    # 이전 10초간 데이터를 바탕으로 10초 후 RCP 예측
    def __init__(self, X=None, y=None, time_shift=10):
        if X is None or y is None:
            pass 
        else:
            self.X = X
            self.y = y
            self.time_shift = time_shift
            self.prepare_data()

    def prepare_data(self):
        # data
        X = self.X 
        y = self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
        print('X_train shape : ', X_train.shape)
        print('y_train shape : ', y_train.shape)
        print('X_test shape : ', X_test.shape)
        print('y_test shape : ', y_test.shape)
        
        print('-------Reshape-----------')
        # LSTM은 3D input 필요: (sample size, timestep, feature size)
        def reshape_array(input_array):
            print(f'input shape: {input_array.shape}')
            output_array = input_array.reshape(input_array.shape[0], self.time_shift, int(input_array.shape[1]/self.time_shift))
            return output_array 
        
        X_train = reshape_array(X_train)
        X_test = reshape_array(X_test)
        print('X_train reshape : ', X_train.shape)
        print('X_test reshape : ', X_test.shape)
        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)
        print('y_train reshape : ', y_train.shape)
        print('y_test reshape : ', y_test.shape)

        self.X_train = X_train 
        self.X_test = X_test 
        self.y_train = y_train 
        self.y_test = y_test

    def create_model(self):
        # Model structure
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(10, activation = 'relu', input_shape=(self.X_train.shape[1], self.X_train.shape[2]))) # (timestep, feature)
        # DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(20))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.y_train.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model 

    def train_model(self, num_epoch=1000, batch_size=512, patience=100):
        # model
        model = self.create_model()
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        
        # Fitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience)
        history = model.fit(self.X_train, self.y_train, epochs=num_epoch, batch_size=batch_size, validation_data=(self.X_test, self.y_test), verbose=1, callbacks=[early_stopping])
        self.model = model
        # Loss plotting
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        y_hat = self.model.predict(X_test)
        plt.scatter(y_test, y_hat)
        plt.show()

    def predict_model(self, X_predict):
        predictions = self.model.predict(X_predict)
        return predictions

class ModelDNN:
    # Teamfight 시작 시 resources 기준으로 teamfight 승패예측
    def __init__(self, X=None, y=None):
        if X is None or y is None:
            pass 
        else:
            self.X = X
            self.y = y
            self.prepare_data()
    
    def prepare_data(self):
        # data
        X = self.X 
        y = self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

        # y_train = tf.keras.utils.to_categorical(y_train, len(np.unique(y)))
        # y_test = tf.keras.utils.to_categorical(y_test, len(np.unique(y)))
        
        print('X_train shape : ', X_train.shape)
        print('y_train shape : ', y_train.shape)
        print('X_test shape : ', X_test.shape)
        print('y_test shape : ', y_test.shape)

        self.X_train = X_train 
        self.X_test = X_test 
        self.y_train = y_train 
        self.y_test = y_test 

    def create_model(self):
        # Model structure
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, activation = 'relu', input_shape=(self.X_train.shape[1], ))) # (feature,)
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(64, activation = 'relu', input_shape=(self.X_train.shape[1], ))) # (feature,)
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # binary classification
        # model.add(tf.keras.layers.Dense(self.y_train.shape[1], activation='softmax')) # multi-label
        model.summary()
        return model 
    
    def train_model(self, num_epoch=1000, batch_size=512, patience=100, verbose=1):
        # model
        model = self.create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        
        # Fitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience)
        history = model.fit(self.X_train, self.y_train, epochs=num_epoch, batch_size=batch_size, validation_data=(self.X_test, self.y_test), verbose=verbose, callbacks=[early_stopping])        
        self.model = model

        # Loss plotting
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='validation acc')
        plt.legend()
        plt.show()