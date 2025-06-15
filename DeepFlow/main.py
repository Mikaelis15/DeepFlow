import sys
from PyQt6.QtWidgets import QApplication
from gui.login_window import LoginWindow
from gui.main_window import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    login_window = LoginWindow()
    
    if login_window.exec():
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())

