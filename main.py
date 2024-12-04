import sys
import cv2
import requests
from datetime import datetime
from constants import Constants
from translator import Translator
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QScrollArea,
    QFrame,
    QSizePolicy,
)


class BaseRequestThread(QThread):
    finished_signal = pyqtSignal(dict, name="finish")

    def __init__(self, url, data=None, files=None, method='post', expect_json=True, parent=None):
        super(BaseRequestThread, self).__init__(parent)
        self.url = url
        self.data = data or {}
        self.files = files
        self.method = method
        self.expect_json = expect_json

    def run(self):
        try:
            response = requests.request(self.method, self.url, data=self.data, files=self.files)
            if response.status_code == 200:
                if self.expect_json:
                    self.finished_signal.emit({"success": True, "data": response.json()})
                else:
                    self.finished_signal.emit({"success": True})
            else:
                self.finished_signal.emit({"success": False, "code": response.status_code, "message": response.text})
        except Exception as e:
            self.finished_signal.emit({"success": False, "code": 500, "message": str(e)})


class ScheduleRequestThread(BaseRequestThread):
    def __init__(self, frame_data, parent=None):
        url = f"{Constants.server_url}/api/device/schedule/"
        data = {"device_id": Constants.device_id}
        files = {"file": ("image.jpg", frame_data, "image/jpeg")}
        super(ScheduleRequestThread, self).__init__(url, data=data, files=files, method='post', parent=parent)


class MarkRequestThread(BaseRequestThread):
    def __init__(self, frame_data, parent=None):
        url = f"{Constants.server_url}/api/device/attendance/"
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%dT%H:%M:%S.000")
        data = {"device_id": Constants.device_id, "datetime": "2024-12-02T17:40:00.000"}  # formatted_time
        files = {"file": ("image.jpg", frame_data, "image/jpeg")}
        super(MarkRequestThread, self).__init__(url, data=data, files=files, method='post', parent=parent)


class CancelMarkRequestThread(BaseRequestThread):
    def __init__(self, student_id, schedule_id, parent=None):
        url = f"{Constants.server_url}/api/device/attendance/"
        data = {"device_id": Constants.device_id, "student_id": str(student_id), "schedule_id": str(schedule_id)}
        super(CancelMarkRequestThread, self).__init__(url, data=data, method='delete', expect_json=False, parent=parent)


class CameraThread(QThread):
    update_frame_signal = pyqtSignal(object, name="update_frame")

    def __init__(self, parent=None):
        super(CameraThread, self).__init__(parent)
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.update_frame_signal.emit(frame)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QWidget):
    def __init__(self, languages):
        super().__init__(flags=Qt.WindowFlags(Qt.Window))
        self.schedule_button = None
        self.attendance_button = None
        self.description_label = None
        self.main_footer = None
        self.floating_frame = None
        self.main_header_label = None
        self.schedule_screen = None
        self.camera_screen = None
        self.main_screen = None
        self.stack = QStackedLayout()
        self.setLayout(self.stack)
        self.languages = languages
        self.translator = Translator(self.languages)
        self.chosen_action = 0
        self.camera_thread = None
        self.timer = None
        self.count_errors = 0
        self.is_loaded = False
        self.current_frame = None
        self.server_thread = None
        self.header_label = None
        self.video_label = None
        self.cancel_button = None
        self.loading_label = None
        self.initUI()

    def initUI(self):
        self.initMainScreen()
        self.initCameraScreen()
        self.main_header_label.setText(
            self.translator.get("room_name", room=Constants.room, building=Constants.building)
        )

    def initMainScreen(self):
        if self.main_screen is not None:
            self.stack.removeWidget(self.main_screen)
            self.main_screen.deleteLater()

        self.main_screen = QWidget(flags=Qt.WindowFlags(Qt.Window))
        main_layout = QVBoxLayout(self.main_screen)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.main_header_label = self.createHeaderLabel("")
        main_layout.addWidget(self.main_header_label, alignment=Qt.AlignTop)

        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(20, 20, 20, 20)
        center_layout.setSpacing(20)

        self.floating_frame = self.createFloatingFrame()
        center_layout.addWidget(self.floating_frame, alignment=Qt.AlignBottom)
        main_layout.addLayout(center_layout)

        self.main_footer = self.createFooter()
        main_layout.addWidget(self.main_footer, alignment=Qt.AlignBottom)

        self.stack.insertWidget(0, self.main_screen)

    def initCameraScreen(self):
        # Инициализируем экран камеры только один раз
        if self.camera_screen is not None:
            return

        self.camera_screen = QWidget(flags=Qt.WindowFlags(Qt.Window))
        layout = QVBoxLayout(self.camera_screen)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.header_label = self.createHeaderLabel("")
        self.header_label.setFixedHeight(70)
        layout.addWidget(self.header_label, alignment=Qt.Alignment())

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label, alignment=Qt.Alignment())

        footer_widget = self.createCameraFooter()
        layout.addWidget(footer_widget, alignment=Qt.Alignment())

        self.loading_label = QLabel(self.translator.get("loading"))
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("""
            font-size: 20px;
            color: white;
            background-color: #2f7829;
            border-radius: 10px;
            padding: 10px;
        """)
        self.loading_label.setVisible(self.is_loaded)
        video_layout = QVBoxLayout(self.video_label)
        video_layout.addWidget(self.loading_label, alignment=Qt.AlignCenter)
        self.video_label.setLayout(video_layout)

        self.stack.addWidget(self.camera_screen)

    @staticmethod
    def createHeaderLabel(text):
        header_label = QLabel(text)
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("""
            background-color: #2f7829;
            color: white;
            font-size: 28px;
            font-weight: bold;
            padding: 20px;
        """)
        return header_label

    def createFloatingFrame(self):
        floating_frame = QFrame(flags=Qt.WindowFlags(Qt.Window))
        floating_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 20px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        """)
        floating_frame.setContentsMargins(20, 20, 20, 20)
        floating_frame_layout = QVBoxLayout(floating_frame)
        floating_frame_layout.setContentsMargins(20, 20, 20, 20)

        self.description_label = QLabel(self.translator.get("select_action"))
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setStyleSheet("font-size: 24px; color: #000000; font-weight: bold;")
        floating_frame_layout.addWidget(self.description_label, alignment=Qt.AlignTop)
        floating_frame_layout.addStretch()

        button_layout = QHBoxLayout()
        self.attendance_button = self.createActionButton(
            "icons/face.png", self.translator.get("mark_attendance"), action=lambda: self.switchToCameraScreen(1)
        )
        button_layout.addWidget(self.attendance_button, alignment=Qt.Alignment())
        self.schedule_button = self.createActionButton(
            "icons/calendar.png", self.translator.get("view_schedule"), action=lambda: self.switchToCameraScreen(2)
        )
        button_layout.addWidget(self.schedule_button, alignment=Qt.Alignment())

        button_container = QWidget(flags=Qt.WindowFlags(Qt.Window))
        button_container.setLayout(button_layout)
        button_container.setFixedHeight(200)
        floating_frame_layout.addWidget(button_container, alignment=Qt.AlignBottom)
        return floating_frame

    def createFooter(self):
        footer = QScrollArea()
        footer.setWidgetResizable(True)
        footer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        footer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        footer_content = QFrame(flags=Qt.WindowFlags(Qt.Window))
        footer_layout = QHBoxLayout(footer_content)
        footer_layout.setSpacing(10)
        footer_layout.setContentsMargins(10, 10, 10, 10)
        for lang in self.languages:
            lang_button = QPushButton()
            lang_button.setStyleSheet(f"""
                border-radius: 25px;
                border: 2px solid #FFFFFF;
                width: 50px;
                height: 50px;
                border-image: url({lang['icon']});
            """)
            lang_button.setFixedSize(50, 50)
            lang_button.clicked.connect(lambda checked, code=lang['code']: self.changeLanguage(code))
            footer_layout.addWidget(lang_button, alignment=Qt.Alignment())
        footer_content.setLayout(footer_layout)
        footer.setWidget(footer_content)
        footer.setFixedHeight(70)
        footer.setStyleSheet("background-color: #2f7829;")
        return footer

    def changeLanguage(self, lang_code):
        self.translator.set_language(lang_code)
        self.refreshUI()

    def refreshUI(self):
        # Обновляем тексты на главном экране
        self.main_header_label.setText(
            self.translator.get("room_name", room=Constants.room, building=Constants.building)
        )
        self.description_label.setText(self.translator.get("select_action"))
        self.attendance_button = self.updateActionButton(
            self.attendance_button, "icons/face.png", self.translator.get("mark_attendance")
        )
        self.schedule_button = self.updateActionButton(
            self.schedule_button, "icons/calendar.png", self.translator.get("view_schedule")
        )

        # Обновляем текст на экране камеры
        if self.chosen_action == 1:
            self.header_label.setText(self.translator.get("mark_attendance"))
        elif self.chosen_action == 2:
            self.header_label.setText(self.translator.get("view_schedule"))
        self.cancel_button.setText(self.translator.get("cancel"))
        self.loading_label.setText(self.translator.get("loading"))

    @staticmethod
    def updateActionButton(button, icon_path, text):
        # Обновляем текст на кнопке действия
        layout = button.layout()
        if layout:
            text_label = layout.itemAt(1).widget()
            text_label.setText(text)
        return button

    def createCameraFooter(self):
        footer_widget = QWidget(flags=Qt.WindowFlags(Qt.Window))
        footer_widget.setFixedHeight(70)
        footer_widget.setStyleSheet("background-color: #026d34;")
        footer_layout = QHBoxLayout(footer_widget)

        self.cancel_button = QPushButton(self.translator.get("cancel"))
        self.cancel_button.setStyleSheet("""
            background-color: #ff4d4d;
            color: white;
            font-size: 20px;
            font-weight: bold;
            padding: 10px;
            border-radius: 10px;
        """)
        self.cancel_button.clicked.connect(self.stopCamera)
        footer_layout.addWidget(self.cancel_button, alignment=Qt.Alignment())
        return footer_widget

    @staticmethod
    def createActionButton(icon_path, text, action=None):
        button = QPushButton()
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button.setStyleSheet("""
            background-color: #93ff73; 
            border-radius: 20px; 
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(5)

        icon_label = QLabel()
        icon_label.setPixmap(QIcon(icon_path).pixmap(80, 80))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label, alignment=Qt.Alignment())

        text_label = QLabel(text)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("font-size: 24px; color: #026d34; font-weight: bold;")
        layout.addWidget(text_label, alignment=Qt.Alignment())

        button.setLayout(layout)

        if action:
            button.clicked.connect(action)

        return button

    def switchToCameraScreen(self, action_type):
        self.chosen_action = action_type
        if self.chosen_action == 1:
            self.header_label.setText(self.translator.get("mark_attendance"))
        else:
            self.header_label.setText(self.translator.get("view_schedule"))
        self.stack.setCurrentWidget(self.camera_screen)
        self.startCamera()

        self.timer = QTimer(self)
        if self.chosen_action == 1:
            self.timer.timeout.connect(self.sendFrameForMark)
        else:
            self.timer.timeout.connect(self.sendFrameForSchedule)
        self.timer.setSingleShot(True)
        self.timer.start(3000)

    def startCamera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.update_frame_signal.connect(self.updateFrame)
        self.camera_thread.start()

    def stopCamera(self):
        self.chosen_action = 0
        if self.timer:
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None
        self.count_errors = 0
        self.is_loaded = False
        self.current_frame = None
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.stack.setCurrentWidget(self.main_screen)  # Возвращаемся на главный экран

    def updateFrame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = rgb_image
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def sendFrame(self, request_thread_class, response_handler):
        self.is_loaded = True
        self.loading_label.setText(self.translator.get("loading"))
        self.loading_label.setVisible(True)
        self.cancel_button.setEnabled(False)

        _, buffer = cv2.imencode('.jpg', self.current_frame)
        frame_data = buffer.tobytes()

        self.server_thread = request_thread_class(frame_data)
        self.server_thread.finished_signal.connect(response_handler)
        self.server_thread.start()

    def sendFrameForSchedule(self):
        self.sendFrame(ScheduleRequestThread, self.handleScheduleServerResponse)

    def sendFrameForMark(self):
        self.sendFrame(MarkRequestThread, self.handleMarkServerResponse)

    def handleServerResponse(self, result, success_callback, retry_callback):
        self.loading_label.setVisible(False)
        self.cancel_button.setEnabled(True)

        if result["success"]:
            success_callback(result.get("data"))
        else:
            self.count_errors += 1
            print(f"КОД {result['code']}: {result['message']}")
            if self.count_errors > 10:
                self.showErrorMessage(result["code"], result["message"])
            else:
                retry_callback()

    def handleScheduleServerResponse(self, result):
        self.handleServerResponse(
            result,
            success_callback=self.processScheduleData,
            retry_callback=self.sendFrameForSchedule
        )

    def handleMarkServerResponse(self, result):
        self.handleServerResponse(
            result,
            success_callback=self.showMarkMessage,
            retry_callback=self.sendFrameForMark
        )

    def handleCancelMarkServerResponse(self, result, message):
        self.loading_label.setVisible(False)
        self.cancel_button.setEnabled(True)

        if result["success"]:
            self.stopCamera()
        else:
            self.showCancelErrorMessage(result["code"], result["message"], message)

    def processScheduleData(self, data):
        self.switchToScheduleScreen(data)

    def createScheduleScreen(self, schedule_data):
        schedule_screen = QWidget(flags=Qt.WindowFlags(Qt.Window))
        layout = QVBoxLayout(schedule_screen)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        header_widget = self.createScheduleHeader(schedule_data)
        layout.addWidget(header_widget, alignment=Qt.Alignment())

        scroll_area = self.createScheduleContent(schedule_data)
        layout.addWidget(scroll_area, alignment=Qt.Alignment())

        return schedule_screen

    def createScheduleHeader(self, schedule_data):
        header_layout = QHBoxLayout()
        back_button = QPushButton("")
        back_button.setFixedSize(100, 40)
        back_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
            }
        """)
        back_button.setIcon(QIcon("icons/arrow.png"))
        back_button.setIconSize(QSize(34, 34))
        back_button.clicked.connect(self.backToMain)
        header_layout.addWidget(back_button, alignment=Qt.AlignLeft)

        if schedule_data:
            group = schedule_data[0]['group']
            group_label_text = self.translator.get("schedule_group", group=group)
        else:
            group_label_text = self.translator.get("schedule")
        group_label = QLabel(group_label_text)
        group_label.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: bold;
        """)
        header_layout.addWidget(group_label, alignment=Qt.AlignCenter)

        header_widget = QWidget(flags=Qt.WindowFlags(Qt.Window))
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background-color: #026d34; padding: 10px;")
        return header_widget

    def createScheduleContent(self, schedule_data):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("background-color: #f2f2f2; border: none;")

        scroll_content = QWidget(flags=Qt.WindowFlags(Qt.Window))
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)
        scroll_layout.setContentsMargins(15, 15, 15, 15)

        if not schedule_data:
            empty_label = QLabel(self.translator.get("no_classes"))
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("font-size: 22px; color: #666;")
            scroll_layout.addWidget(empty_label, alignment=Qt.Alignment())
        else:
            grouped_schedule = {}
            for lesson in schedule_data:
                grouped_schedule.setdefault(lesson["date"], []).append(lesson)

            sorted_schedule = sorted(grouped_schedule.items(), key=lambda x: x[0])
            for date, lessons in sorted_schedule:
                lessons.sort(key=lambda x: x["time_start"])
                day_block = self.createDayBlock(date, lessons)
                scroll_layout.addWidget(day_block, alignment=Qt.Alignment())

        scroll_area.setWidget(scroll_content)
        return scroll_area

    def createDayBlock(self, date, lessons):
        day_block = QFrame(flags=Qt.WindowFlags(Qt.Window))
        day_block.setStyleSheet("""
            background-color: #e7f2d0;
            border-radius: 20px;
            border: 2px solid #497037;
            padding: 15px;
        """)
        day_layout = QVBoxLayout(day_block)

        date_weekday = self.translator.get("date_weekday", date=date, weekday=lessons[0]['weekday'])
        day_header = QLabel(date_weekday)
        day_header.setStyleSheet("border: none; color: #497037; font-size: 22px; font-weight: bold; margin-bottom: 10px;")
        day_layout.addWidget(day_header, alignment=Qt.Alignment())

        for lesson in lessons:
            lesson_label = QLabel(
                f"{lesson['time_start']} - {lesson['time_end']} | {lesson['subject']}\n"
                f"{lesson['teacher']}\n"
                f"{lesson['room']} ({lesson['building']})"
            )
            lesson_label.setStyleSheet("border: none; color: #497037; font-weight: bold; font-size: 18px; margin-bottom: 10px;")
            lesson_label.setWordWrap(True)
            day_layout.addWidget(lesson_label, alignment=Qt.Alignment())

            if lesson != lessons[-1]:
                divider = QFrame(flags=Qt.WindowFlags(Qt.Window))
                divider.setFrameShape(QFrame.HLine)
                divider.setFrameShadow(QFrame.Sunken)
                divider.setStyleSheet("background-color: #497037;")
                day_layout.addWidget(divider, alignment=Qt.Alignment())

        return day_block

    def switchToScheduleScreen(self, schedule_data=None):
        self.timer = None
        self.count_errors = 0
        self.is_loaded = False
        self.current_frame = None
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        if not schedule_data:
            schedule_data = []
        self.schedule_screen = self.createScheduleScreen(schedule_data)
        self.stack.addWidget(self.schedule_screen)
        self.stack.setCurrentWidget(self.schedule_screen)

    def backToMain(self):
        self.chosen_action = 0
        self.stack.setCurrentWidget(self.main_screen)

    def showOverlayDialog(self, title, message, buttons):
        overlay = QWidget(self, flags=Qt.WindowFlags(Qt.Window))
        overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0.5);")
        overlay.setGeometry(self.rect())
        overlay.setWindowFlags(Qt.SubWindow | Qt.FramelessWindowHint)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        overlay.setAttribute(Qt.WA_DeleteOnClose)
        overlay.show()

        dialog = QFrame(overlay, flags=Qt.WindowFlags(Qt.FramelessWindowHint))
        dialog.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        """)
        dialog.setFixedSize(400, 200)
        dialog.move((self.width() - dialog.width()) // 2, (self.height() - dialog.height()) // 2)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF0000;")
        layout.addWidget(title_label, alignment=Qt.Alignment())

        text_label = QLabel(message)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("font-size: 14px; color: #333333;")
        layout.addWidget(text_label, alignment=Qt.Alignment())

        button_layout = QHBoxLayout()
        for button_text, button_style, button_action in buttons:
            button = QPushButton(button_text)
            button.setStyleSheet(button_style)
            button.clicked.connect(lambda _, b=button_action: b(overlay, dialog))
            button_layout.addWidget(button, alignment=Qt.Alignment())
        layout.addLayout(button_layout)

        dialog.show()
        overlay.show()

    def showMarkMessage(self, message):
        title = self.translator.get("success")
        msg_text = (
            f"{self.translator.get('group')}: {message['group_name']}\n"
            f"{self.translator.get('student')}: {message['name']}\n"
            f"{self.translator.get('subject')}: {message['subject_name']}\n"
            f"{self.translator.get('is_correct')}"
        )
        buttons = [
            (self.translator.get("confirm"), """
                background-color: #93ff73;
                color: #026d34;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
            """, self.cancelAction),
            (self.translator.get("cancel"), """
                background-color: #ff4d4d;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
            """, lambda overlay, dialog: self.cancelMark(overlay, dialog, message)),
        ]
        self.showOverlayDialog(title, msg_text, buttons)

    def showErrorMessage(self, code, text):
        title = self.translator.get("error")
        msg_text = f"{self.translator.get('code')}: {code}\n{self.translator.get('message')}: {text}"
        buttons = [
            (self.translator.get("retry"), """
                background-color: #93ff73;
                color: #026d34;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
            """, self.retryAction),
            (self.translator.get("cancel"), """
                background-color: #ff4d4d;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
            """, self.cancelAction),
        ]
        self.showOverlayDialog(title, msg_text, buttons)

    def showCancelErrorMessage(self, code, text, message):
        title = self.translator.get("error")
        msg_text = f"{self.translator.get('code')}: {code}\n{self.translator.get('message')}: {text}"
        buttons = [
            (self.translator.get("retry"), """
                background-color: #93ff73;
                color: #026d34;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
            """, lambda overlay, dialog: self.cancelMark(overlay, dialog, message)),
            (self.translator.get("cancel"), """
                background-color: #ff4d4d;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
            """, self.cancelAction),
        ]
        self.showOverlayDialog(title, msg_text, buttons)

    def cancelMark(self, overlay, dialog, message):
        overlay.deleteLater()
        dialog.deleteLater()

        self.server_thread = CancelMarkRequestThread(message["student_id"], message["schedule_id"])
        self.server_thread.finished_signal.connect(
            lambda result: self.handleCancelMarkServerResponse(result, message)
        )
        self.server_thread.start()

    def retryAction(self, overlay, dialog):
        self.count_errors = 0
        overlay.deleteLater()
        dialog.deleteLater()
        if self.chosen_action == 1:
            self.sendFrameForMark()
        else:
            self.sendFrameForSchedule()

    def cancelAction(self, overlay, dialog):
        overlay.deleteLater()
        dialog.deleteLater()
        self.stopCamera()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_languages = [
        {"code": "ru", "icon": "icons/ru.png"},
        {"code": "zh", "icon": "icons/zh.png"},
        {"code": "en", "icon": "icons/en.png"},
        {"code": "ja", "icon": "icons/ja.png"},
        {"code": "de", "icon": "icons/de.png"},
        {"code": "fr", "icon": "icons/fr.png"},
    ]
    window = MainWindow(app_languages)
    window.setWindowTitle("Интерфейс устройства")
    window.resize(1024, 600)
    window.show()

    sys.exit(app.exec_())
