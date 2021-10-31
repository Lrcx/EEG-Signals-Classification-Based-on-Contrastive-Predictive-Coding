import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import regularizers
import torch
from lunwen import network_encoder


def cpc_model():
    data=torch.load("total_data.pt")["samples"].squeeze(1).numpy()
    labels=torch.load("total_data.pt")["labels"].numpy()
    return data,labels


def process_data():
    Data,Label=cpc_model()
    # np.random.seed(3666)
    random = np.random.permutation(Data.shape[0])
    Data = Data[random]
    Label = Label[random]

    sic1=int(Data.shape[0]*1)
    len=int(sic1*0.9)
    return Data[0:len], Label[0:len], Data[len:sic1], Label[len:sic1]
    #return Data[0:sic1], Label[0:sic1], Data[sic1:], Label[sic1:]

def process_data1():
    Data,Label=cpc_model()
    sic1=int(Data.shape[0]*0.1)
    len=int(sic1*0.9)
    return Data[0:len], Label[0:len], Data[len:sic1], Label[len:sic1]
    #return Data[0:sic1], Label[0:sic1], Data[sic1:], Label[sic1:]

class SortedNumberGenerator(object):
    ''' Data generator providing lists of sorted numbers '''
    def __init__(self, batch_size,isTrain):
        self.isTrain=isTrain
        self.batch_size = batch_size
        self.X_train,self.y_train,self.X_test,self.y_test=process_data()

    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    def __len__(self):
        if self.isTrain:
            return len(self.X_train)//self.batch_size
        else:
            return len(self.X_test)//self.batch_size
    def next(self):
        if self.isTrain:
            idx=np.random.choice(len(self.X_train),self.batch_size,replace=False)
            X_train=self.X_train[idx]
            return np.stack([X_train,X_train,X_train,X_train],axis=1),keras.utils.to_categorical(self.y_train[idx],num_classes=5)
        else:
            idx = np.random.choice(len(self.X_test), self.batch_size, replace=False)
            X_test=self.X_test[idx]
            return np.stack([X_test,X_test,X_test,X_test],axis=1),keras.utils.to_categorical(self.y_test[idx],num_classes=5)

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cpc=keras.models.load_model("eeg2.h5")
        # self.encoder_input = keras.layers.Input((4, 3000))
        # self.encoder_output = network_encoder(self.encoder_input, 128)
        # self.encoder_model = keras.models.Model(self.encoder_input, self.encoder_output, name='encoder')
        self.normal=keras.layers.BatchNormalization()
        self.encoder=keras.Sequential(
            [
             keras.layers.Conv1D(filters=8, kernel_size=4, kernel_initializer="he_uniform", strides=1, padding='same',use_bias=True, kernel_regularizer=regularizers.l2(0.001),activation=keras.activations.relu),
            keras.layers.MaxPool1D(pool_size=2),
            keras.layers.Conv1D(filters=16, kernel_size=6, kernel_initializer="he_uniform", strides=1, padding='same',use_bias=True, kernel_regularizer=regularizers.l2(0.001),activation=keras.activations.relu),
            keras.layers.MaxPool1D(pool_size=2),
             ]
        )
        self.flatten=keras.layers.Flatten()
        self.result=keras.layers.Dense(5,activation=keras.activations.softmax)


    def call(self,inputs):
        # inputs=self.encoder_model(inputs)
        inputs=self.cpc(inputs)
        inputs=self.normal(inputs)
        output= self.encoder(inputs)
        output=self.flatten(output)
        logits=self.result(output)
        return logits

def train_model():
    model = MyModel()
    #model.layers[0].trainable = False
    BATCH_SIZE=32
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
    train_data = SortedNumberGenerator(BATCH_SIZE, True)
    validation_data = SortedNumberGenerator(BATCH_SIZE, False)
    callbacks = [keras.callbacks.ReduceLROnPlateau( factor=1 / 3, patience=2, min_lr=1e-6)]
    h = model.fit_generator(generator=train_data, steps_per_epoch=len(train_data), epochs=50, verbose=1,validation_data=validation_data, validation_steps=len(validation_data),callbacks=callbacks)
    model.save_weights("model.h5")
    _, _,X_test, y_test = process_data()
    # X_test=np.stack([X_test,X_test,X_test,X_test],axis=1)
    # y_pred = model(X_test)
    # y_pred=tf.argmax(y_pred,axis=1)
    # result=classification_report(y_test.flatten(),y_pred.numpy().flatten())
    # print(result)
    data, labels = cpc_model()
    data = np.stack([data, data, data, data], axis=1)
    y_pred = model(data)
    y_pred = tf.argmax(y_pred, axis=1)
    result = classification_report(labels.flatten(), y_pred.numpy().flatten(),digits=4)
    print(result)
    print(confusion_matrix(labels,y_pred))
#print(model(np.arange(32*10*251).reshape(32,10,251)))






if __name__ == '__main__':
    train_model()
    # cpc_model()
