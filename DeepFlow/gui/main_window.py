import sys
import os
import webbrowser
import tempfile
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QFileDialog, QStackedWidget)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from gui.widgets import LoadingSpinner
from backend.processing import AnalysisWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.csv_path = None
        self.temp_csv_path = None 
        self.init_ui()
        self.load_stylesheet()

    def init_ui(self):
        self.setWindowTitle("Bloodcount AI - Analysis Dashboard")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.setMinimumSize(900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(20, 20, 20, 20)

        left_panel = QFrame()
        left_panel.setObjectName("SidePanel")
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(20, 20, 20, 20)
        left_panel.setFixedWidth(320)

        title_label = QLabel("Analysis Control")
        title_label.setObjectName("HeaderLabel")
        
        self.drag_drop_label = QLabel("\n\nDrag & Drop CSV File\n\nor\n\nClick to Select\n\n")
        self.drag_drop_label.setObjectName("DragDropLabel")
        self.drag_drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drag_drop_label.setMinimumHeight(200)
        self.drag_drop_label.mousePressEvent = self.open_file_dialog
        self.setAcceptDrops(True)

        self.status_label = QLabel("Please load a file to begin.")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)

        self.start_button = QPushButton("Start Bloodcount")
        self.start_button.clicked.connect(self.start_analysis)
        self.start_button.setEnabled(False)
        self.start_button.setFixedHeight(50)
        
        self.open_csv_button = QPushButton("Open Classification Results")
        self.open_csv_button.clicked.connect(self.open_results_csv)
        self.open_csv_button.setFixedHeight(40)
        self.open_csv_button.setStyleSheet("background-color: #457b9d;")
        self.open_csv_button.hide()

        left_panel_layout.addWidget(title_label)
        left_panel_layout.addSpacing(20)
        left_panel_layout.addWidget(self.drag_drop_label)
        left_panel_layout.addSpacing(10)
        left_panel_layout.addWidget(self.status_label)
        left_panel_layout.addStretch()
        left_panel_layout.addWidget(self.start_button)
        left_panel_layout.addSpacing(10)
        left_panel_layout.addWidget(self.open_csv_button) 

        right_panel = QFrame(); right_panel.setObjectName("ResultsPanel")
        right_panel_layout = QVBoxLayout(right_panel); right_panel_layout.setContentsMargins(30, 0, 0, 0)
        self.results_title = QLabel("Analysis Results"); self.results_title.setObjectName("TitleLabel"); self.results_title.hide()
        results_hbox = QHBoxLayout(); results_hbox.setSpacing(20)
        self.rbc_count_widget = self.create_result_display("RBC Count", "0")
        self.plt_count_widget = self.create_result_display("PLT Count", "0")
        results_hbox.addWidget(self.rbc_count_widget); results_hbox.addWidget(self.plt_count_widget)
        self.plot_area_stack = QStackedWidget()
        self.plot_placeholder = QLabel("Cluster plot will appear here after analysis."); self.plot_placeholder.setObjectName("PlotLabel"); self.plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spinner_container = QWidget(); spinner_layout = QVBoxLayout(spinner_container); spinner_layout.setAlignment(Qt.AlignmentFlag.AlignCenter); self.loading_spinner = LoadingSpinner(); spinner_layout.addWidget(self.loading_spinner)
        self.plot_label = QLabel(); self.plot_label.setObjectName("PlotLabel"); self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_area_stack.addWidget(self.plot_placeholder); self.plot_area_stack.addWidget(spinner_container); self.plot_area_stack.addWidget(self.plot_label)
        right_panel_layout.addWidget(self.results_title); right_panel_layout.addSpacing(15); right_panel_layout.addLayout(results_hbox); right_panel_layout.addSpacing(15); right_panel_layout.addWidget(self.plot_area_stack, 1)
        main_layout.addWidget(left_panel); main_layout.addWidget(right_panel)

    def create_result_display(self, title, initial_value):
        widget = QFrame(); widget.setObjectName("ResultCard")
        layout = QVBoxLayout(widget); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title); title_label.setObjectName("ResultTitleLabel")
        value_label = QLabel(initial_value); value_label.setObjectName("ResultValueLabel")
        layout.addWidget(value_label, alignment=Qt.AlignmentFlag.AlignCenter); layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        widget.hide()
        return widget

    def start_analysis(self):
        if not self.csv_path: return
        self.start_button.setEnabled(False)
        self.open_csv_button.hide()
        self.plot_area_stack.setCurrentIndex(1)
        self.loading_spinner.start_animation()
        self.worker_thread = QThread()
        self.analysis_worker = AnalysisWorker(self.csv_path)
        self.analysis_worker.moveToThread(self.worker_thread)
        self.analysis_worker.status_update.connect(self.update_status_message)
        self.worker_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.worker_thread.start()

    def update_status_message(self, message):
        self.status_label.setText(message)

    def on_analysis_finished(self, results):
        self.loading_spinner.stop_animation()
        self.update_status_message("Analysis complete. Ready for next file.")
        
        rbc_count, plt_count, plot_path, temp_csv_path = results
        self.temp_csv_path = temp_csv_path
        
        self.results_title.show()
        self.rbc_count_widget.show(); self.plt_count_widget.show()
        self.rbc_count_widget.findChild(QLabel, "ResultValueLabel").setText(f"{rbc_count:,}")
        self.plt_count_widget.findChild(QLabel, "ResultValueLabel").setText(f"{plt_count:,}")
        
        pixmap = QPixmap(plot_path)
        self.plot_label.setPixmap(pixmap.scaled(self.plot_area_stack.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.plot_area_stack.setCurrentIndex(2)
        
        self.open_csv_button.show()
        self.start_button.setEnabled(True)
        self.worker_thread.quit()
        self.worker_thread.wait()

    def on_analysis_error(self, error_message):
        self.loading_spinner.stop_animation()
        self.update_status_message(f"Error: {error_message}")
        self.plot_placeholder.setText(f"Analysis Failed:\n{error_message}")
        self.plot_area_stack.setCurrentIndex(0)
        self.open_csv_button.hide()
        self.start_button.setEnabled(True)
        self.worker_thread.quit()
        self.worker_thread.wait()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event):
        url = event.mimeData().urls()[0]
        if url.isLocalFile() and url.toLocalFile().endswith('.csv'):
            self.handle_file_path(url.toLocalFile())

    def open_file_dialog(self, event):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if path: self.handle_file_path(path)

    def handle_file_path(self, path):
        self.cleanup_temp_file()
        self.csv_path = path
        filename = os.path.basename(path)
        self.drag_drop_label.setText(f"Loaded:\n{filename}")
        self.start_button.setEnabled(True)
        self.update_status_message("File loaded. Ready to start analysis.")
        self.open_csv_button.hide()
        self.plot_placeholder.setText("Cluster plot will appear here after analysis.")
        self.plot_area_stack.setCurrentIndex(0)
        self.results_title.hide()
        self.rbc_count_widget.hide()
        self.plt_count_widget.hide()

    def open_results_csv(self):
        if self.temp_csv_path and os.path.exists(self.temp_csv_path):
            try:
                webbrowser.open(self.temp_csv_path)
            except Exception as e:
                self.update_status_message(f"Error opening file: {e}")
        else:
            self.update_status_message("Results file not found.")
            
    def cleanup_temp_file(self):
        if self.temp_csv_path and os.path.exists(self.temp_csv_path):
            try:
                os.remove(self.temp_csv_path)
                self.temp_csv_path = None
            except OSError as e:
                print(f"Error removing temp file: {e}")

    def closeEvent(self, event):
        self.cleanup_temp_file()
        event.accept()

    def load_stylesheet(self):
        try:
            with open("gui/style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Stylesheet 'gui/style.qss' not found.")

