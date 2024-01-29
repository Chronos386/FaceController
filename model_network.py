import os
import keras
import random
import numpy as np
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from keras.src.regularizers import l2
from keras.src.layers import BatchNormalization, Dropout

current_script_directory = os.path.dirname(os.path.abspath(__file__))


class User:
    def __init__(self, user_id, embeddings):
        self.user_id = user_id
        self.embeddings = embeddings


class Model:
    def __init__(self):
        self.pathModel = f"{current_script_directory}/model/face_ind_model.keras"
        self.model = None
        self.History = None

    def create_model(self, x_train, num_class):
        self.model = Sequential()
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, input_shape=x_train[0].shape, activation='relu', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation='relu', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation='relu', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(num_class, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def model_training(self, x_train, y_train, x_test, y_test):
        self.History = self.model.fit(x_train, y_train, batch_size=11, epochs=1000, validation_data=(x_test, y_test))
        self.model.save(self.pathModel)

    def history_output(self):
        acc = self.History.history["accuracy"]
        val_acc = self.History.history["val_accuracy"]
        loss = self.History.history["loss"]
        val_loss = self.History.history["val_loss"]

        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, "bo", label="Точность на этапе обучения")
        plt.plot(epochs, val_acc, "b", label="Точность на этапе проверки")
        plt.title("Точность на этапах обучения и проверки")
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, "bo", label="Потери на этапе обучения")
        plt.plot(epochs, val_loss, "b", label="Потери на этапе проверки")
        plt.title("Потери на этапах обучения и проверки")
        plt.legend()
        plt.show()

    def check_best_model(self, x_test):
        x_test = np.expand_dims(x_test, axis=0)
        best_model = keras.models.load_model(self.pathModel)
        score = best_model.predict(x_test)
        max_index = np.where(score[0] == max(score[0]))
        print(max(score[0]), " ", max_index)
        print(score[0])
        max_index = np.where(score[0] == max(score[0]))[0][0]
        if max(score[0]) > 0.9:
            return max_index
        else:
            return 0

    def del_usr(self, x_train, y_train, x_test, y_test, idx):
        best_model = keras.models.load_model(self.pathModel)
        weights_bak = best_model.layers[-1].get_weights()
        num_class = best_model.layers[-1].units - 1
        best_model.pop()  # Удаление последнего слоя
        best_model.add(Dense(num_class, activation='softmax', name=f"name_{random.randint(1000, 10000)}"))

        weights_new = best_model.layers[-1].get_weights()
        # скопируйте исходные веса обратно
        weights_new[0][0, 0:idx] = weights_bak[0][0, 0:idx]
        weights_new[1][0:idx] = weights_bak[1][0:idx]
        if idx != num_class:
            weights_new[0][0, idx:] = weights_bak[0][0, idx + 1:]
            weights_new[1][idx:] = weights_bak[1][idx + 1:]
        best_model.layers[-1].set_weights(weights_new)
        best_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        best_model.fit(x_train, y_train, batch_size=11, epochs=100, validation_data=(x_test, y_test))
        best_model.save(self.pathModel)

    def add_new_usr(self, x_train, y_train, x_test, y_test, idx):
        best_model = keras.models.load_model(self.pathModel)
        weights_bak = best_model.layers[-1].get_weights()
        num_class = best_model.layers[-1].units + 1
        best_model.pop()  # Удаление последнего слоя
        best_model.add(Dense(num_class, activation='softmax', name=f"name_{random.randint(1000, 10000)}"))

        weights_new = best_model.layers[-1].get_weights()
        # скопируйте исходные веса обратно
        weights_new[0][0, 0:idx] = weights_bak[0][0, 0:idx]
        weights_new[1][0:idx] = weights_bak[1][0:idx]
        if idx != num_class - 1:
            weights_new[0][0, idx + 1:] = weights_bak[0][0, idx:]
            weights_new[1][idx + 1:] = weights_bak[1][idx:]
        # используйте средний вес для инициализации нового класса.
        weights_new[0][:, idx] = np.mean(weights_bak[0], axis=1)
        weights_new[1][idx] = np.mean(weights_bak[1])
        best_model.layers[-1].set_weights(weights_new)
        best_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        best_model.fit(x_train, y_train, batch_size=1, epochs=10, validation_data=(x_test, y_test))
        best_model.save(self.pathModel)
