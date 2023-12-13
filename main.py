import os
import cv2
import keras
import random
import cv2 as cv
import numpy as np
import face_recognition
from statistics import mode
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from keras.src.regularizers import l2
from keras.src.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.src.layers import BatchNormalization, Dropout


def split_array(arr):
    n = len(arr)
    chunk_size = n // 3
    remaining = n % 3
    parts = [arr[i:i + chunk_size] for i in range(0, n, chunk_size)]
    if remaining == 1:
        parts[0].append(arr[len(arr) - 1])
    if remaining == 2:
        parts[0].append(arr[len(arr) - 2])
        parts[1].append(arr[len(arr) - 1])
    return parts


def split_data(list_files_paths):
    all_files_path = []
    for folder_path in list_files_paths:
        files = os.listdir(folder_path[0])
        all_files_path.append(files)

    all_files = []
    for i in range(0, len(all_files_path)):
        files = []
        folder = all_files_path[i]
        for file_name in folder:
            file_path = os.path.join(list_files_paths[i][0], file_name)
            files.append(np.load(file_path))
        all_files.append((files, list_files_paths[i][1]))

    train_emb_data = []
    verif_emb_users = []
    test_emb_users = []

    for person_embeddings in all_files:
        parts = split_array(person_embeddings[0])
        train_emb_data.append(User(user_id=person_embeddings[1], embeddings=parts[0]))
        verif_emb_users.append(User(user_id=person_embeddings[1], embeddings=parts[1]))
        test_emb_users.append(User(user_id=person_embeddings[1], embeddings=parts[2]))

    train_emb_data += verif_emb_users
    return train_emb_data, test_emb_users


def create_embedding(path):
    image = face_recognition.load_image_file(path)
    checker = False
    embedding = None
    if len(face_recognition.face_encodings(image)) != 0:
        embedding = face_recognition.face_encodings(image)[0]
        checker = True
    return embedding, checker


class User:
    def __init__(self, user_id, embeddings):
        self.user_id = user_id
        self.embeddings = embeddings


class Model:
    def __init__(self):
        self.pathModel = "model/face_ind_model.keras"
        self.model = None
        self.History = None

    def create_model(self, x_train, num_class):
        self.model = Sequential()
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, input_shape=x_train[0].shape, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(64, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation='tanh', kernel_regularizer=l2(1e-4)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation='tanh', kernel_regularizer=l2(1e-4)))
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


class FaceController:
    def __init__(self):
        self.list_path = []
        self.list_names = []
        self.model = Model()
        self.unknown_emb = "unknown/unknown_emb"
        self.unknown_photos = "unknown/unknown_photos"

    def use_camera(self):
        capture = cv.VideoCapture(0)
        faces = cv.CascadeClassifier('camera/faces.xml')
        list_ind = []
        while True:
            ret, img = capture.read()
            results = faces.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
            for (x, y, w, h) in results:
                cv.rectangle(img, (x, y), (x + w, y + h), (125, 125, 255), thickness=3)
                cv.imshow('camera', img[y:y + h, x:x + w])
                cv2.imwrite("camera/camera_photo.jpg", img[y:y + h, x:x + w])
                emb, checker = create_embedding("camera/camera_photo.jpg")
                if checker:
                    ind = self.model.check_best_model(emb)
                    list_ind.append(ind)

            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break
            if len(list_ind) == 10:
                break
        capture.release()
        cv.destroyAllWindows()
        print(f"Это: {self.list_names[mode(list_ind)]}")

    def use_photo(self, path):
        faces = cv.CascadeClassifier('photos/faces.xml')
        img = cv.imread(path)

        results = faces.detectMultiScale(img, scaleFactor=2, minNeighbors=5, minSize=(10, 10))
        for (x, y, w, h) in results:
            # cv.imshow('camera', img[y:y + h, x:x + w])
            cv2.imwrite("photos/photos_photo.jpg", img[y:y + h, x:x + w])
            emb, checker = create_embedding("photos/photos_photo.jpg")
            if checker:
                ind = self.model.check_best_model(emb)
                print(f"Это: {self.list_names[ind]}")
                cv2.waitKey(0)

    def add_new_unknown(self):
        files = os.listdir(self.unknown_photos)
        files_emb = os.listdir(self.unknown_emb)
        start_len = len(files_emb)
        i = 0
        for file in files:
            unknown_emb, checker = create_embedding(f"{self.unknown_photos}/{file}")
            if checker:
                np.save(f"{self.unknown_emb}/{i + start_len}.npy", unknown_emb)
                i += 1
        for file in files:
            try:
                os.remove(f"{self.unknown_photos}/{file}")
            except FileNotFoundError:
                print(f"Файл {file} не найден.")
            except Exception as e:
                print(f"Произошла ошибка при удалении файла {file}: {e}")

    def create_enb_new_person(self):
        files = os.listdir(self.unknown_photos)
        files_emb = os.listdir(self.unknown_emb)
        start_len = len(files_emb)
        i = 0
        for file in files:
            unknown_emb, checker = create_embedding(f"{self.unknown_photos}/{file}")
            if checker:
                np.save(f"{self.unknown_emb}/{i + start_len}.npy", unknown_emb)
                i += 1
            else:
                print(file)
        for file in files:
            try:
                os.remove(f"{self.unknown_photos}/{file}")
            except FileNotFoundError:
                print(f"Файл {file} не найден.")
            except Exception as e:
                print(f"Произошла ошибка при удалении файла {file}: {e}")

    def create_and_train_model(self):
        train_users, test_users = split_data(self.list_path)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for us in train_users:
            for emb in us.embeddings:
                x_train.append(emb)
                y_train.append(us.user_id)
        for us in test_users:
            for emb in us.embeddings:
                x_test.append(emb)
                y_test.append(us.user_id)

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_train = to_categorical(y_train, num_classes=len(list_path))
        y_test = encoder.fit_transform(y_test)
        y_test = to_categorical(y_test, num_classes=len(list_path))

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        num_class = len(y_train[0])
        self.model.create_model(x_train, num_class)
        self.model.model_training(x_train, y_train, x_test, y_test)
        self.model.history_output()

    def add_new_user(self, new_ind):
        train_users, test_users = split_data(self.list_path)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for us in train_users:
            for emb in us.embeddings:
                x_train.append(emb)
                y_train.append(us.user_id)
        for us in test_users:
            for emb in us.embeddings:
                x_test.append(emb)
                y_test.append(us.user_id)

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_train = to_categorical(y_train, num_classes=len(list_path))
        y_test = encoder.fit_transform(y_test)
        y_test = to_categorical(y_test, num_classes=len(list_path))

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self.model.add_new_usr(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, idx=new_ind)

    def del_user(self, del_ind):
        train_users, test_users = split_data(self.list_path)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for us in train_users:
            for emb in us.embeddings:
                x_train.append(emb)
                y_train.append(us.user_id)
        for us in test_users:
            for emb in us.embeddings:
                x_test.append(emb)
                y_test.append(us.user_id)

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_train = to_categorical(y_train, num_classes=len(list_path))
        y_test = encoder.fit_transform(y_test)
        y_test = to_categorical(y_test, num_classes=len(list_path))

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self.model.del_usr(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, idx=del_ind)

    def update_list_path(self, new_list):
        self.list_path.clear()
        self.list_path = new_list

    def update_list_names(self, new_list):
        self.list_names.clear()
        self.list_names = new_list


if __name__ == '__main__':
    list_path = [("unknown_persons", 0), ("D:\\metadata\\12002015\\1", 1), ("D:\\metadata\\12002015\\2", 2),
                 ("D:\\metadata\\12002015\\3", 3), ("D:\\metadata\\12002015\\4", 4), ("D:\\metadata\\12002015\\5", 5),
                 ("D:\\metadata\\12002015\\6", 6), ("D:\\metadata\\12002015\\7", 7), ("D:\\metadata\\tch\\1", 8),
                 ("D:\\metadata\\12002333\\1", 9)]
    list_names = ["Неизвестный", "М_1", "Свилогузов", "Шатохин", "М_4", "Савченко", "Слапыгина", "М_6",
                  "Александр Геннадиевич", "Никита Мартон"]
    face_controller = FaceController()
    face_controller.update_list_path(list_path)
    face_controller.update_list_names(list_names)
    # face_controller.create_and_train_model()
    print("Включаю камеру")
    face_controller.use_camera()
    print("Введи что-нибудь, а потом нажми enter")
    input()
    # face_controller.use_photo(path="test/2.jpg")
