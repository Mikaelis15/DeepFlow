from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QWidget
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt
import config

class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.USERNAME = config.USERNAME
        self.PASSWORD = config.PASSWORD
        self.init_ui()
        self.load_stylesheet()

    def init_ui(self):
        self.setWindowTitle("DeepFlow - Login")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.setFixedSize(400, 420)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(15)
        
        logo_label = QLabel()
        pixmap = QPixmap("assets/icon.png") 
        logo_label.setPixmap(pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel("DeepFlow AI")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setObjectName("HeaderLabel")
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.username_input.setFixedHeight(45)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setFixedHeight(45)

        self.login_button = QPushButton("Log In")
        self.login_button.clicked.connect(self.handle_login)
        self.login_button.setFixedHeight(50)

        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setObjectName("ErrorLabel")
        self.error_label.setFixedHeight(20)

        main_layout.addWidget(logo_label)
        main_layout.addWidget(title_label)
        main_layout.addSpacing(20)
        main_layout.addWidget(self.username_input)
        main_layout.addWidget(self.password_input)
        main_layout.addWidget(self.error_label)
        main_layout.addStretch()
        main_layout.addWidget(self.login_button)
        
    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if username == self.USERNAME and password == self.PASSWORD:
            self.accept()
        else:
            self.error_label.setText("Invalid credentials. Please try again.")
            self.password_input.clear()

    def load_stylesheet(self):
        try:
            with open("assets/style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Stylesheet 'gui/style.qss' not found.")
