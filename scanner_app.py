import os
import cv2
import queue
import cv2 as cv
import numpy as np
import face_recognition
import multiprocessing
from PyQt5.QtCore import Qt
from statistics import mode
from model_network import Model, User
from keras.src.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QLabel, QWidget, QMessageBox

current_script_directory = os.path.dirname(os.path.abspath(__file__))
faces = cv.CascadeClassifier(f"camera/faces.xml")


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


def analyze_face(emb_queue, img_queue, model_queue):
    img = img_queue.get_nowait()
    model = model_queue.get_nowait()
    results = faces.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
    if len(results) == 0:
        emb_queue.put(None)
    else:
        for (x, y, w, h) in results:
            cv.rectangle(img, (x, y), (x + w, y + h), (125, 125, 255), thickness=3)
            cv2.imwrite(f"camera/camera_photo.jpg", img[y:y + h, x:x + w])
            emb, checker = create_embedding(f"camera/camera_photo.jpg")
            if checker:
                ind = model.check_best_model(emb)
                while not emb_queue.empty():
                    _ = emb_queue.get()
                emb_queue.put(ind)
            else:
                emb_queue.put(None)


class ScannerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.notCanceled = True
        self.notification = None
        self.setWindowTitle('Сканер')
        self.setMinimumSize(480, 320)
        self.setMaximumSize(480, 320)

        self.list_path = []
        self.list_names = []
        self.model = Model()
        self.current_img = None
        self.unknown_emb = f"{current_script_directory}/unknown/unknown_emb"
        self.unknown_photos = f"{current_script_directory}/unknown/unknown_photos"

        font = QFont()
        font.setPointSize(16)
        self.start_text = QLabel(
            "<html><head/><body><p align=\"center\"><b>НАЖМИТЕ ЧТОБЫ НАЧАТЬ</b><br><b>СКАНИРОВАНИЕ</b></p></body></html>",
            self)
        self.start_text.setFont(font)

        self.main_pict = QLabel(self)
        self.arrow_pict = QLabel(self)

        self.start_scan_button = QPushButton("НАЧАТЬ СКАНИРОВАНИЕ", self)
        self.start_scan_button.setStyleSheet(f"font-weight: bold; font-size: {20}px;")
        self.start_scan_button.setFixedHeight(40)

        self.back_button = QPushButton("ОТМЕНИТЬ СКАНИРОВАНИЕ", self)
        self.back_button.setStyleSheet(f"font-weight: bold; font-size: {10}px;")
        self.back_button.setFixedHeight(20)

        self.arrow_pict.setPixmap(QPixmap(self.scaledArrow("photos/arrow.png")))

        self.back_button.hide()
        self.main_pict.hide()

        layout = QVBoxLayout(self)
        layout.addWidget(self.start_text, alignment=Qt.AlignHCenter)
        layout.addWidget(self.main_pict, alignment=Qt.AlignHCenter)
        layout.addWidget(self.arrow_pict, alignment=Qt.AlignHCenter)
        layout.addWidget(self.back_button)
        layout.addWidget(self.start_scan_button)

        self.setLayout(layout)
        self.back_button.clicked.connect(lambda: self.pressCancelButton())
        self.start_scan_button.clicked.connect(lambda: self.pressStartButton())

    def use_camera(self):
        list_ind = []
        img_queue = multiprocessing.Queue()
        emb_queue = multiprocessing.Queue()
        model_queue = multiprocessing.Queue()
        capture = cv2.VideoCapture(0)
        _, img = capture.read()
        img_queue.put(img)
        model_queue.put(self.model)
        analyze_process = multiprocessing.Process(target=analyze_face, args=(emb_queue, img_queue, model_queue))
        analyze_process.start()
        while True & self.notCanceled:
            _, img = capture.read()
            self.update_frame(img)
            cv2.waitKey(1)
            try:
                ind = emb_queue.get_nowait()
                if ind is not None:
                    list_ind.append(ind)
                if len(list_ind) != 3:
                    img_queue.put(img)
                    model_queue.put(self.model)
                    analyze_process = multiprocessing.Process(target=analyze_face, args=(emb_queue, img_queue, model_queue))
                    analyze_process.start()
            except queue.Empty:
                pass
            if len(list_ind) == 3:
                break
        analyze_process.kill()
        if self.notCanceled:
            self.current_img = None
            print(f"Это: {self.list_names[mode(list_ind)]}")
            self.showNotification(self.list_names[mode(list_ind)])
            self.pressCancelButton()
        self.notCanceled = True

    def use_photo(self, path):
        img = cv.imread(path)

        results = faces.detectMultiScale(img, scaleFactor=2, minNeighbors=5, minSize=(10, 10))
        for (x, y, w, h) in results:
            cv2.imwrite(f"{current_script_directory}/photos/photos_photo.jpg", img[y:y + h, x:x + w])
            emb, checker = create_embedding(f"{current_script_directory}/photos/photos_photo.jpg")
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
        y_train = to_categorical(y_train, num_classes=len(self.list_path))
        y_test = encoder.fit_transform(y_test)
        y_test = to_categorical(y_test, num_classes=len(self.list_path))

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
        y_train = to_categorical(y_train, num_classes=len(self.list_path))
        y_test = encoder.fit_transform(y_test)
        y_test = to_categorical(y_test, num_classes=len(self.list_path))

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
        y_train = to_categorical(y_train, num_classes=len(self.list_path))
        y_test = encoder.fit_transform(y_test)
        y_test = to_categorical(y_test, num_classes=len(self.list_path))

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self.model.del_usr(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, idx=del_ind)

    def update_list_path(self, new_list):
        self.list_path.clear()
        self.list_path = new_list

    def update_list_names(self, new_list):
        self.list_names.clear()
        self.list_names = new_list

    def scaledArrow(self, imagePath):
        pixmap = QPixmap(imagePath)
        return pixmap.scaled(self.main_pict.size(), 457, 600)

    def pressStartButton(self):
        self.back_button.show()
        self.main_pict.show()

        self.arrow_pict.hide()
        self.start_text.hide()
        self.start_scan_button.hide()
        try:
            self.use_camera()
        except Exception as e:
            print(f"POPA = {e}")

    def showNotification(self, person):
        message_box = QMessageBox()
        message_box.setMinimumSize(240, 160)
        message_box.setMaximumSize(240, 160)

        if person == "Неизвестный":
            message_box.setWindowTitle("Неудача")
            message_box.setText("К сожалению, Вас не удалось распознать.")
        else:
            message_box.setWindowTitle("Успех")
            message_box.setText(f"Добро пожаловать, {person}!")
        message_box.addButton(QMessageBox.Ok)
        self.notification = message_box
        self.notification.show()

    def pressCancelButton(self):
        self.notCanceled = False
        self.back_button.hide()
        self.main_pict.hide()

        self.arrow_pict.show()
        self.start_text.show()
        self.start_scan_button.show()

    def update_frame(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.main_pict.setPixmap(pixmap.scaledToWidth(480))
