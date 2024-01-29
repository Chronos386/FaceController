import multiprocessing
import sys
from PyQt5.QtWidgets import QApplication
from scanner_app import ScannerApp

if __name__ == '__main__':
    list_path = [("unknown_persons", 0), ("metadata/12002015/1", 1), ("metadata/12002015/2", 2),
                 ("metadata/12002015/3", 3), ("metadata/12002015/4", 4), ("metadata/12002015/5", 5),
                 ("metadata/12002015/6", 6), ("metadata/12002015/7", 7), ("metadata/tch/1", 8),
                 ("metadata/12002333/1", 9)]
    list_names = ["Неизвестный", "М_1", "Свилогузов", "Шатохин", "М_4", "Савченко", "Слапыгина", "М_6",
                  "Александр Геннадиевич", "Никита Мартон"]
    # face_controller = FaceController()
    # face_controller.update_list_path(list_path)
    # face_controller.update_list_names(list_names)
    # choose = input("Тренировать модель - 1; Не тренировать модель - 2: ")
    # if choose == "1":
    #     face_controller.create_and_train_model()
    # choose = input("Включить камеру - 1; Распознавать фото - 2: ")
    # if choose == "1":
    #     print("Включаю камеру")
    #     face_controller.use_camera()
    # else:
    #     print("Распознаю фото")
    #     face_controller.use_photo(path=f"{current_script_directory}/test/2.jpg")
    # print("Введите что-нибудь, а потом нажми enter")
    # input()
    multiprocessing.set_start_method('spawn')
    app = QApplication(sys.argv)
    scanner = ScannerApp()
    scanner.update_list_path(list_path)
    scanner.update_list_names(list_names)
    scanner.show()
    sys.exit(app.exec_())
