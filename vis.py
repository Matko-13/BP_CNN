from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QToolButton, QLabel, QLineEdit,
    QGraphicsScene, QMenu, QGraphicsView, QSlider
)
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence

import src.tools 
import cv2
import numpy as np
import sys
import math

class ZoomEventFilter(QObject):

    def __init__(self, view: QGraphicsView, on_mouse_move=None):
        super().__init__(view)
        self.view = view
        self.on_mouse_move = on_mouse_move

    def eventFilter(self, obj, event):
        if obj in (self.view, self.view.viewport()):
            if event.type() == QEvent.Type.Wheel:
                scale_factor = 1.15
                if event.angleDelta().y() > 0:
                    if self.view.transform().m11() < 10:
                        self.view.scale(scale_factor, scale_factor)
                else:
                    if self.view.transform().m11() > 0.2:
                        self.view.scale(1.0 / scale_factor, 1.0 / scale_factor)
                return True
            if event.type() == QEvent.Type.MouseMove and self.on_mouse_move:
                self.on_mouse_move(event.position().toPoint())
        return super().eventFilter(obj, event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()


        #premenne
        self.overlay = False
        self.mode = 0
        self.base_mode = 0
        self.alpha = 0.5
        self.max_index = src.tools.dataset_len() - 1
        self.pm_stack = [None] * 9
        self.acc_stack = [None] * 3
        self.time_stack = [None] * 3
        self.cv_stack = [None] * 5
        self.mask_stack = {} 
        self.scene = QGraphicsScene()
        self.pixmap_item = self.scene.addPixmap(QPixmap())

        #window
        self.setWindowTitle("Vizualizácia")
        self.resize(1600, 1000)

        #cely layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        #sidebar settings
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setStyleSheet("background-color: #2c3e50;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)

        #labele
        self.status_label = QLabel("Vstup")
        self.time_label = QLabel("Čas: ---")
        self.acc_label = QLabel("Presnosť: ---")
        self.pixel_label = QLabel("Trieda: ---")

        #id number input
        self.number_input = QLineEdit()
        self.number_input.setPlaceholderText("ID")
        self.number_input.returnPressed.connect(lambda: self.update_pixmaps(int(self.number_input.text())) if self.number_input.text() else None)

        #button na menu
        self.btn_menu = QToolButton()
        self.btn_menu.setText("Select model   ")
        self.btn_menu.setPopupMode(QToolButton.InstantPopup)

        #menu
        sidebar_menu = QMenu(self)
        menu_input = sidebar_menu.addAction("Vstup")
        menu_annot = sidebar_menu.addAction("Anotácia")
        menu_model1 = sidebar_menu.addAction("Model v1.0")
        menu_model2 = sidebar_menu.addAction("Model v1.2")
        menu_model3 = sidebar_menu.addAction("Model v1.3")

        menu_input.triggered.connect(lambda: self.update_mode(0, "Vstup"))
        menu_annot.triggered.connect(lambda: self.update_mode(2, "Anotácia"))
        menu_model1.triggered.connect(lambda: self.update_mode(4, "Model v1.0"))
        menu_model2.triggered.connect(lambda: self.update_mode(6, "Model v1.2"))
        menu_model3.triggered.connect(lambda: self.update_mode(8, "Model v1.3"))

        self.btn_menu.setMenu(sidebar_menu)

        #toggle overlay
        self.latch_btn = QPushButton("Overlay OFF")
        self.latch_btn.setCheckable(True)
        self.latch_btn.toggled.connect(self.on_latch_toggled)

        #alpha slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)      
        self.slider.setValue(70)          
        self.slider.valueChanged.connect(self.update_overlay)

        #sidebar layout
        sidebar_layout.addWidget(self.status_label)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.acc_label)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.time_label)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.number_input)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.btn_menu)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.latch_btn)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.slider)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.pixel_label)
        sidebar_layout.addStretch()

        #scene 
        self.view = QGraphicsView(self.scene)

        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)

        #init mouse_move a scroll callback
        self.zoom_filter = ZoomEventFilter(self.view, self.on_mouse_move)
        self.view.viewport().installEventFilter(self.zoom_filter)

        #hlavny layout
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.view, 1)  

        #keybindy
        QShortcut(QKeySequence("O"), self, activated=self.latch_btn.toggle)
        QShortcut(QKeySequence("Space"), self, activated=self.reset_zoom)
        QShortcut(QKeySequence("1"), self, activated=lambda: self.update_mode(0, "Vstup"))
        QShortcut(QKeySequence("2"), self, activated=lambda: self.update_mode(2, "Annotácia"))
        QShortcut(QKeySequence("3"), self, activated=lambda: self.update_mode(4, "Model v1.0"))
        QShortcut(QKeySequence("4"), self, activated=lambda: self.update_mode(6, "Model v1.2"))
        QShortcut(QKeySequence("5"), self, activated=lambda: self.update_mode(8, "Model v1.3"))

    def on_latch_toggled(self, checked):
        self.overlay = checked
        self.latch_btn.setText("Overlay ON" if checked else "Overlay OFF")
        self.update_mode(self.base_mode, self.status_label.text())

    def update_mode(self, mode, title_text):
        self.base_mode = mode
        self.mode = mode + 1 if self.overlay else mode
        self.status_label.setText(title_text)
        self.update_scene()
        self.update_labels()

    def bgr_to_pixmap(self, img: np.ndarray) -> QPixmap:
        img = np.ascontiguousarray(img)  
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    
    def update_pixmaps(self, index):
        if 0 <= index <= self.max_index:
            model1_output, image, annotation, self.time_stack[0] =  src.tools.make_set(index, 1, True)
            model2_output, self.time_stack[1] =  src.tools.make_set(index, 2, False)
            model3_output, self.time_stack[2] =  src.tools.make_set(index, 3, False)

            self.acc_stack[0] = src.tools.evaluate_images(annotation, model1_output)
            self.acc_stack[1] = src.tools.evaluate_images(annotation, model2_output)
            self.acc_stack[2] = src.tools.evaluate_images(annotation, model3_output)

            self.mask_stack = {
                2: annotation,
                4: model1_output,
                6: model2_output,
                8: model3_output,
            }

            self.cv_stack[0] = image_rgb = np.ascontiguousarray(image[:, :, ::-1])
            self.cv_stack[1] = annotation
            self.cv_stack[2] = model1_output
            self.cv_stack[3] = model2_output
            self.cv_stack[4] = model3_output

            annot_overlay = cv2.addWeighted(image_rgb, self.alpha, annotation, 1 - self.alpha, 0)
            overlay1 = cv2.addWeighted(image_rgb, self.alpha, model1_output, 1 - self.alpha, 0)
            overlay2 = cv2.addWeighted(image_rgb, self.alpha, model2_output, 1 - self.alpha, 0)
            overlay3 = cv2.addWeighted(image_rgb, self.alpha, model3_output, 1 - self.alpha, 0)

            self.pm_stack[0] = self.bgr_to_pixmap(image_rgb)
            self.pm_stack[1] = self.bgr_to_pixmap(annotation)
            self.pm_stack[2] = self.bgr_to_pixmap(annot_overlay)
            self.pm_stack[3] = self.bgr_to_pixmap(model1_output)
            self.pm_stack[4] = self.bgr_to_pixmap(overlay1)
            self.pm_stack[5] = self.bgr_to_pixmap(model2_output)
            self.pm_stack[6] = self.bgr_to_pixmap(overlay2)
            self.pm_stack[7] = self.bgr_to_pixmap(model3_output)
            self.pm_stack[8] = self.bgr_to_pixmap(overlay3)

            self.update_scene()
            self.update_labels()
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        else:
            self.number_input.clear()
            self.number_input.setPlaceholderText(f"MAX {self.max_index}")


    def update_scene(self):
        if self.mode == 0 or self.mode == 1:
            if self.pm_stack[0] is not None:
                self.pixmap_item.setPixmap(self.pm_stack[0])
                self.scene.setSceneRect(self.pm_stack[0].rect())  
        else:
            if self.pm_stack[self.mode - 1] is not None:
                self.pixmap_item.setPixmap(self.pm_stack[self.mode -1 ])
                self.scene.setSceneRect(self.pm_stack[self.mode -1].rect())

    def update_labels(self):
        if self.mode > 3 and self.acc_stack[0] is not None:
            index = int((self.mode - 4) / 2)
            self.acc_label.setText(f"Presnosť: {self.acc_stack[index]:.2f}%")
            self.time_label.setText(f"Čas: {self.time_stack[index] * 1000:.2f} ms")
        else:
            self.acc_label.setText("Presnosť: ---")
            self.time_label.setText("Čas: ---")

    def on_mouse_move(self, view_pos):
        mask = self.mask_stack.get(self.base_mode)
        if mask is None:
            self.pixel_label.setText("Trieda: ---")
            return

        scene_pos = self.view.mapToScene(view_pos)
        item_pos = self.pixmap_item.mapFromScene(scene_pos)
        x = math.floor(item_pos.x())
        y = math.floor(item_pos.y())

        h, w, _ = mask.shape
        if 0 <= x < w and 0 <= y < h:
            r, g, b = mask[y, x]
            name = src.tools.rgb_to_class(r, g, b)
            self.pixel_label.setText(f"Trieda: {name or '---'}")
        else:
            self.pixel_label.setText("Trieda: ---")

    def update_overlay(self, value):

        if self.cv_stack[0] is not None:
            self.alpha = value / 100
            annot_overlay = cv2.addWeighted(self.cv_stack[0], self.alpha, self.cv_stack[1], 1 - self.alpha, 0)
            overlay1 = cv2.addWeighted(self.cv_stack[0], self.alpha, self.cv_stack[2], 1 - self.alpha, 0)
            overlay2 = cv2.addWeighted(self.cv_stack[0], self.alpha, self.cv_stack[3], 1 - self.alpha, 0)
            overlay3 = cv2.addWeighted(self.cv_stack[0], self.alpha, self.cv_stack[4], 1 - self.alpha, 0)

            self.pm_stack[2] = self.bgr_to_pixmap(annot_overlay)
            self.pm_stack[4] = self.bgr_to_pixmap(overlay1)
            self.pm_stack[6] = self.bgr_to_pixmap(overlay2)
            self.pm_stack[8] = self.bgr_to_pixmap(overlay3)

            self.update_scene()

    def reset_zoom(self):
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())