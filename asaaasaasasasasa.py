from keras import Sequential
from keras.src.regularizers import l2
from keras.layers import Dropout, Dense
from keras.src.layers import BatchNormalization


class Model:
    def __init__(self):
        self.pathModel = "model/face_ind_model.keras"
        self.model = None
        self.History = None

    def create_model(self, x_train, num_class):
        self.model = Sequential()
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, input_shape=x_train[0].shape, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(num_class, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
