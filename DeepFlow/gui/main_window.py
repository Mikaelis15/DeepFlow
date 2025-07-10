import os
import sys
import webbrowser
import tempfile

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFrame, QFileDialog, QStackedWidget, QMessageBox
)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt, QThread

from gui.widgets import LoadingSpinner
from backend.processing import IDECWorker, VaDEWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.csv_path = None
        self.temp_csv_path = None
        self.idec_thread = None
        self.vade_thread = None

        self.init_ui()
        self.load_stylesheet()
        self.showFullScreen()

    def init_ui(self):
        self.setWindowTitle("Bloodcount AI - Analysis Dashboard")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.setMinimumSize(900, 700)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

        self.setup_control_panel(main_layout)
        self.setup_results_panel(main_layout)

    def setup_control_panel(self, layout):
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setFixedWidth(320)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Analysis Control")
        title.setObjectName("HeaderLabel")

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
        self.start_button.clicked.connect(self.run_idec_pipeline)
        self.start_button.setEnabled(False)
        self.start_button.setFixedHeight(50)

        self.vade_button = QPushButton("Start VaDE Clustering")
        self.vade_button.clicked.connect(self.run_vade_pipeline)
        self.vade_button.setFixedHeight(50)
        self.vade_button.hide()

        self.open_csv_button = QPushButton("Open IDEC Results")
        self.open_csv_button.clicked.connect(self.open_results_csv)
        self.open_csv_button.setFixedHeight(50)
        self.open_csv_button.hide()

        self.close_button = QPushButton("Close Application")
        self.close_button.clicked.connect(self.close_application)
        self.close_button.setFixedHeight(50)

        vbox.addWidget(title)
        vbox.addSpacing(20)
        vbox.addWidget(self.drag_drop_label)
        vbox.addSpacing(10)
        vbox.addWidget(self.status_label)
        vbox.addStretch()
        vbox.addWidget(self.start_button)
        vbox.addSpacing(10)
        vbox.addWidget(self.vade_button)
        vbox.addSpacing(10)
        vbox.addWidget(self.open_csv_button)
        vbox.addSpacing(20)
        vbox.addWidget(self.close_button)

        layout.addWidget(panel)

    def setup_results_panel(self, layout):
        panel = QFrame()
        panel.setObjectName("ResultsPanel")

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(30, 0, 0, 0)

        self.results_title = QLabel("Analysis Results")
        self.results_title.setObjectName("TitleLabel")
        self.results_title.hide()

        vbox.addWidget(self.results_title)
        vbox.addSpacing(15)

        result_box = QHBoxLayout()
        result_box.setSpacing(20)

        self.rbc_count_widget = self.create_result_display("RBC Count", "0")
        self.plt_count_widget = self.create_result_display("PLT Count", "0")
        self.rbc_subtype1_widget = self.create_result_display("Pure RBCs", "0")
        self.rbc_subtype2_widget = self.create_result_display("Reticulocytes", "0")
        self.rbc_subtype3_widget = self.create_result_display("RBC Clumps", "0")

        self.rbc_subtype1_widget.hide()
        self.rbc_subtype2_widget.hide()
        self.rbc_subtype3_widget.hide()

        result_box.addWidget(self.rbc_count_widget)
        result_box.addWidget(self.plt_count_widget)
        result_box.addWidget(self.rbc_subtype1_widget)
        result_box.addWidget(self.rbc_subtype2_widget)
        result_box.addWidget(self.rbc_subtype3_widget)

        self.plot_area_stack = QStackedWidget()
        self.plot_placeholder = QLabel("Cluster plot will appear here after analysis.")
        self.plot_placeholder.setObjectName("PlotLabel")
        self.plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.loading_spinner = LoadingSpinner()
        spinner_container = QWidget()
        spinner_layout = QVBoxLayout(spinner_container)
        spinner_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spinner_layout.addWidget(self.loading_spinner)

        self.plot_label = QLabel()
        self.plot_label.setObjectName("PlotLabel")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.plot_area_stack.addWidget(self.plot_placeholder)
        self.plot_area_stack.addWidget(spinner_container)
        self.plot_area_stack.addWidget(self.plot_label)

        vbox.addLayout(result_box)
        vbox.addSpacing(15)
        vbox.addWidget(self.plot_area_stack, 1)

        layout.addWidget(panel)

    def create_result_display(self, title, value):
        frame = QFrame()
        frame.setObjectName("ResultCard")
        layout = QVBoxLayout(frame)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        value_label = QLabel(value)
        value_label.setObjectName("ResultValueLabel")

        title_label = QLabel(title)
        title_label.setObjectName("ResultTitleLabel")

        layout.addWidget(value_label)
        layout.addWidget(title_label)

        frame.hide()
        return frame

    def run_idec_pipeline(self):
        if not self.csv_path:
            return

        self.start_button.setEnabled(False)
        self.vade_button.hide()
        self.open_csv_button.hide()
        self.plot_area_stack.setCurrentIndex(1)
        self.loading_spinner.start_animation()
        self.update_status_message("Running IDEC analysis...")

        self.idec_thread = QThread()
        self.idec_worker = IDECWorker(self.csv_path)
        self.idec_worker.moveToThread(self.idec_thread)

        self.idec_worker.status_update.connect(self.update_status_message)
        self.idec_worker.finished.connect(self.on_idec_finished)
        self.idec_worker.error.connect(self.on_analysis_error)

        self.idec_thread.started.connect(self.idec_worker.run)
        self.idec_thread.start()

    def run_vade_pipeline(self):
        if not self.temp_csv_path:
            self.update_status_message("No intermediate data available for VaDE.")
            return

        self.start_button.setEnabled(False)
        self.vade_button.setEnabled(False)
        self.open_csv_button.hide()
        self.plot_area_stack.setCurrentIndex(1)
        self.loading_spinner.start_animation()
        self.update_status_message("Running VaDE clustering...")

        self.vade_thread = QThread()
        self.vade_worker = VaDEWorker(self.temp_csv_path)
        self.vade_worker.moveToThread(self.vade_thread)

        self.vade_worker.status_update.connect(self.update_status_message)
        self.vade_worker.finished.connect(self.on_vade_finished)
        self.vade_worker.error.connect(self.on_analysis_error)
        self.vade_worker.plot_generated.connect(self.on_vade_plot_generated) # Connect new signal

        self.vade_thread.started.connect(self.vade_worker.run)
        self.vade_thread.start()

    def on_idec_finished(self, results):
        self.loading_spinner.stop_animation()
        rbc_count, plt_count, plot_path, self.temp_csv_path = results

        self.results_title.show()
        self.rbc_count_widget.show()
        self.plt_count_widget.show()

        self.rbc_count_widget.findChild(QLabel, "ResultValueLabel").setText(f"{rbc_count:,}")
        self.plt_count_widget.findChild(QLabel, "ResultValueLabel").setText(f"{plt_count:,}")

        self.rbc_subtype1_widget.hide()
        self.rbc_subtype2_widget.hide()
        self.rbc_subtype3_widget.hide()

        pixmap = QPixmap(plot_path)
        if not pixmap.isNull():
            self.plot_label.setPixmap(
                pixmap.scaled(
                    self.plot_area_stack.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.plot_label.setText("Plot image not found or invalid.")

        self.plot_area_stack.setCurrentIndex(2)

        self.update_status_message("IDEC analysis complete. You can now start VaDE clustering.")
        self.open_csv_button.show()
        self.vade_button.show()
        self.vade_button.setEnabled(True)
        self.start_button.setEnabled(True)

        self.idec_thread.quit()
        self.idec_thread.wait()

    def on_vade_finished(self, results):
        self.loading_spinner.stop_animation()
        self.update_status_message("VaDE clustering of RBC subtypes complete.")

        subtype1_count, subtype2_count, subtype3_count, plot_path = results

        self.rbc_subtype1_widget.findChild(QLabel, "ResultValueLabel").setText(f"{subtype1_count:,}")
        self.rbc_subtype2_widget.findChild(QLabel, "ResultValueLabel").setText(f"{subtype2_count:,}")
        self.rbc_subtype3_widget.findChild(QLabel, "ResultValueLabel").setText(f"{subtype3_count:,}")

        self.rbc_subtype1_widget.show()
        self.rbc_subtype2_widget.show()
        self.rbc_subtype3_widget.show()

        # Plot display logic is now handled by on_vade_plot_generated
        # pixmap = QPixmap(plot_path)
        # if not pixmap.isNull():
        #     self.plot_label.setPixmap(
        #         pixmap.scaled(
        #             self.plot_area_stack.size(),
        #             Qt.AspectRatioMode.KeepAspectRatio,
        #             Qt.TransformationMode.SmoothTransformation,
        #         )
        #     )
        # else:
        #     self.plot_label.setText("VaDE plot image not found or invalid.")
        # self.plot_area_stack.setCurrentIndex(2)

        self.open_csv_button.show()
        self.vade_button.setEnabled(True)
        self.start_button.setEnabled(True)

        self.vade_thread.quit()
        self.vade_thread.wait()

    def on_vade_plot_generated(self, plot_path, error_message):
        if error_message:
            self.plot_label.setText(f"VaDE Plot Error: {error_message}")
            self.plot_area_stack.setCurrentIndex(0) # Show placeholder or error message
        else:
            pixmap = QPixmap(plot_path)
            if not pixmap.isNull():
                self.plot_label.setPixmap(
                    pixmap.scaled(
                        self.plot_area_stack.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self.plot_area_stack.setCurrentIndex(2) # Show plot
            else:
                self.plot_label.setText("VaDE plot image not found or invalid.")
                self.plot_area_stack.setCurrentIndex(0) # Show placeholder

    def on_analysis_error(self, error_msg):
        self.loading_spinner.stop_animation()
        self.update_status_message(f"Error during analysis: {error_msg}")
        self.start_button.setEnabled(True)
        self.vade_button.setEnabled(True)
        self.open_csv_button.hide()
        self.plot_area_stack.setCurrentIndex(0)

        if self.idec_thread:
            self.idec_thread.quit()
            self.idec_thread.wait()
        if self.vade_thread:
            self.vade_thread.quit()
            self.vade_thread.wait()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(".csv"):
                self.handle_file_path(file_path)

    def open_file_dialog(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.handle_file_path(file_path)

    def handle_file_path(self, file_path):
        self.csv_path = file_path
        file_name = os.path.basename(file_path)
        self.drag_drop_label.setText(f"Loaded:\n{file_name}")
        self.start_button.setEnabled(True)
        self.status_label.setText("File loaded successfully. Ready to start analysis.")
        self.clear_results()

    def clear_results(self):
        self.results_title.hide()
        self.rbc_count_widget.hide()
        self.plt_count_widget.hide()
        self.rbc_subtype1_widget.hide()
        self.rbc_subtype2_widget.hide()
        self.rbc_subtype3_widget.hide()
        self.plot_area_stack.setCurrentIndex(0)
        self.plot_placeholder.setText("Cluster plot will appear here after analysis.")
        self.open_csv_button.hide()
        self.vade_button.hide()
        self.vade_button.setEnabled(False)

    def open_results_csv(self):
        if self.temp_csv_path and os.path.exists(self.temp_csv_path):
            webbrowser.open(self.temp_csv_path)

    def update_status_message(self, message):
        self.status_label.setText(message)

    def load_stylesheet(self):
        try:
            with open("assets/style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Stylesheet 'assets/style.qss' not found. Please ensure it's in the correct directory.")

    def close_application(self):
        reply = QMessageBox.question(self, 'Close Application',
                                     "Are you sure you want to close the application? Any temporary analysis files will be deleted.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.clean_up_temp_files()

            if self.idec_thread and self.idec_thread.isRunning():
                self.idec_worker.stop()
                self.idec_thread.quit()
                self.idec_thread.wait()
            if self.vade_thread and self.vade_thread.isRunning():
                self.vade_worker.stop()
                self.vade_thread.quit()
                self.vade_thread.wait()

            self.close()

    def clean_up_temp_files(self):
        if self.temp_csv_path and os.path.exists(self.temp_csv_path):
            try:
                os.remove(self.temp_csv_path)
                print(f"Deleted temporary file: {self.temp_csv_path}")
                self.temp_csv_path = None
            except OSError as e:
                print(f"Error deleting temp file {self.temp_csv_path}: {e}")
