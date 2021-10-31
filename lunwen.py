import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import torch
from tensorflow.keras import regularizers
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ECGHandler(object):
    def __init__(self):
        self.X_train,self.y_train,self.X_test,self.y_test=self.load_dataset();


    def load_dataset(self):  # 得到训练集和验证集
        train_data = torch.load('total_data.pt')["samples"].squeeze(1).numpy()
        train_label = torch.load('total_data.pt')["labels"].numpy()
        random = np.random.choice(len(train_data), len(train_data), replace=False)
        train_data = train_data[random]
        train_label = train_label[random]
        sic=int(len(train_label)*0.9)
        return train_data[:sic],train_label[:sic],train_data[sic:], train_label[sic:]

    def get_batch_by_labels(self, subset, labels):
        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Find samples matching labels
        idxs = []
        for i, label in enumerate(labels):
            idx = np.where(y == label)[0]
            idx_sel = np.random.choice(idx, 1)[0]
            idxs.append(idx_sel)
        batch = X[np.array(idxs)]
        return batch.astype('float32'), labels.astype('int32')

    def get_n_samples(self, subset):
        if subset == 'train':
            y_len = self.y_train.shape[0]
        elif subset == 'test':
            y_len = self.y_test.shape[0]
        return y_len

class SortedNumberGenerator(object):
    ''' Data generator providing lists of sorted numbers '''
    def __init__(self, batch_size,subset, terms, positive_samples=16, predict_terms=4):
        # Set params
        self.positive_samples = positive_samples
        self.subset=subset
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.terms = terms
        self.mnist_handler = ECGHandler()
        self.n_samples = self.mnist_handler.get_n_samples(subset) // terms
        # self.n_samples=self.mnist_handler.get_n_samples(subset)//batch_size+1
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    def __len__(self):
        return self.n_samples
    def next(self):
        # Build sentences  建立句子
        image_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))   #(32,8)
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')               #(32,1)
        sentence = np.random.choice([0, 1, 2, 3, 4], self.positive_samples)
        for i in range(16):
            image_labels[i] = sentence[i]  # 设置其前16行是正样本对

        sentence = np.random.choice([0, 1, 2, 3, 4], 16)
        num = np.arange(5)
        for i in range(16, 32):
            predict = [sentence[i - 16] for _ in range(8)]
            temp = np.random.choice(num[num != predict[0]], 1)
            for j in range(4, 8):
                predict[j] = temp
            image_labels[i] = predict
            sentence_labels[i]=0
        # print(image_labels)
        # print(sentence_labels)
        images, _ = self.mnist_handler.get_batch_by_labels(self.subset, image_labels.flatten())
        images = images.reshape((self.batch_size, self.terms + self.predict_terms,images.shape[1]))
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)       #0-32之间选32个不同的数
        return [x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]

def network_encoder(x, code_size=128):
    x = keras.layers.Dense(units=64, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=64, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=64, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=64, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)   #encoder输出是128
    return x

def network_autoregressive(x):
    ''' Define the network that integrates information along the sequence '''
    x = keras.layers.GRU(256, return_sequences=False, name='ar_context')(x)
    return x

def network_prediction(context, code_size, predict_terms):
    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):
    ''' Computes dot product between true and predicted embedding vectors '''
    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):
        preds, y_encoded = inputs        # 32 4 128
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension
        dot_product_probs = K.sigmoid(dot_product)         #输出为（None,1）
        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def network_cpc(image_shape, terms, predict_terms, code_size, learning_rate):
    ''' Define the CPC network combining encoder and autoregressive model '''
    K.set_learning_phase(1)
    encoder_input = keras.layers.Input((terms,image_shape[0]))
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    x_input = keras.layers.Input((terms, image_shape[0]))
    x_encoded=encoder_model(x_input)
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)
    y_input = keras.layers.Input((predict_terms, image_shape[0]))
    y_encoded=encoder_model(y_input)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])                       #计算编码和预测的loss
    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)

    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return cpc_model


def train_model(epochs, batch_size, code_size, lr=1e-4, terms=4, predict_terms=4, image_size=3000):
    train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,positive_samples=batch_size // 2, predict_terms=predict_terms)
    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='test', terms=terms,positive_samples=batch_size // 2, predict_terms=predict_terms)
    # Prepares the model
    model = network_cpc(image_shape=(image_size,), terms=terms, predict_terms=predict_terms,code_size=code_size, learning_rate=lr)
    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]
    # Trains the model
    h=model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )
    print(h.history)
    plt.plot(h.history['binary_accuracy'],ls='--')
    plt.plot(h.history['val_binary_accuracy'])
    plt.title('cpc-based model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'],loc='upper left')
    plt.show()

    encoder=model.layers[1]
    encoder.save("eeg2.h5")

if __name__ == "__main__":
    train_model(
        epochs=20,
        batch_size=32,
        code_size=128,
        lr=0.001,
        terms=4,
        predict_terms=4,
    )