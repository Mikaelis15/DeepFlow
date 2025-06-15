from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import QTimer, Qt

class LoadingSpinner(QWidget):
    """An improved, two-part loading spinner."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_angle)
        self.setFixedSize(60, 60)
        self.color_primary = QColor("#e63946")    # Primary animated color
        self.color_secondary = QColor("#444444")  # Static background track color

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect().adjusted(5, 5, -5, -5)
        
        # Draw the static background track
        pen_bg = QPen(self.color_secondary)
        pen_bg.setWidth(6)
        painter.setPen(pen_bg)
        painter.drawEllipse(rect)
        
        # Draw the animated spinning arc
        pen_fg = QPen(self.color_primary)
        pen_fg.setWidth(6)
        pen_fg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_fg)
        
        arc_length = 270  # The length of the spinning arc in degrees
        painter.drawArc(rect, self.angle * 16, arc_length * 16)

    def update_angle(self):
        """Update the rotation angle and repaint."""
        self.angle = (self.angle + 10) % 360
        self.update()

    def start_animation(self):
        """Start the spinner animation."""
        self.timer.start(20)

    def stop_animation(self):
        """Stop the spinner animation."""
        self.timer.stop()
