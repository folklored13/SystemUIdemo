import sys
import os
import cv2
import random
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL import Image
import torch
from torchvision import transforms


class FlowerClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('icon.ico'))
        self.initUI()
        self.camera = None
        self.timer = QTimer()
        self.current_pixmap=None
        # 类别标签（示例，需替换实际类别）
        self.classes = ['玫瑰', '郁金香', '向日葵', '雏菊', '兰花']

    # def load_model(self):
    #     # 加载训练好的PyTorch模型
    #     model = torch.load('flower_model.pth')
    #     model.eval()
    #     return model

    def initUI(self):
        self.setWindowTitle('花卉图像分类系统')
        self.setGeometry(300, 300, 1200, 800)
        self.setup_main_window()
        self.setup_styles()

    def setup_main_window(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局
        layout = QHBoxLayout()
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()

        layout.addWidget(left_panel, 35)
        layout.addWidget(right_panel, 65)
        main_widget.setLayout(layout)

    def create_left_panel(self):
        panel = QFrame()
        panel.setMinimumWidth(400)
        vbox = QVBoxLayout()

        # 图像显示
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #2E2E2E; border-radius: 10px;")
        self.image_label.setMinimumSize(380, 380)

        # 控制按钮
        btn_upload = QPushButton("上传图片")
        btn_upload.clicked.connect(self.open_image)
        btn_upload.setCursor(Qt.PointingHandCursor)

        btn_camera = QPushButton("启动摄像头")
        btn_camera.clicked.connect(self.toggle_camera)
        btn_camera.setCursor(Qt.PointingHandCursor)

        # 按钮容器
        btn_container = QHBoxLayout()
        btn_container.addWidget(btn_upload)
        btn_container.addWidget(btn_camera)

        vbox.addWidget(self.image_label)
        vbox.addLayout(btn_container)
        panel.setLayout(vbox)
        return panel

    def create_right_panel(self):
        panel = QFrame()
        vbox = QVBoxLayout()

        # 结果标题
        title = QLabel("分类结果")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; color: #4CAF50; margin: 20px 0;")

        # 结果展示表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(['花卉种类', '置信度'])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)

        vbox.addWidget(title)
        vbox.addWidget(self.result_table)
        panel.setLayout(vbox)
        return panel

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #F5F5F5;
            }
            QFrame {
                background: white;
                border-radius: 15px;
                padding: 20px;
            }
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #45a049;
            }
            QTableWidget {
                border: 2px solid #E0E0E0;
                border-radius: 10px;
                alternate-background-color: #F8F8F8;
            }
            QHeaderView::section {
                background: #4CAF50;
                color: white;
                padding: 12px;
            }
        """)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg)")

        if file_path:
            try:
                # 增强文件验证
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件路径不存在: {file_path}")
                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"文件不可读: {file_path}")

                # 使用QImage进行格式验证
                image = QImage(file_path)
                if image.isNull():
                    raise ValueError("不支持的图片格式或文件已损坏")

                # 显式释放旧资源
                current_pixmap = self.image_label.pixmap()
                if current_pixmap:
                    current_pixmap.detach()

                # 创建新pixmap
                self.current_pixmap = QPixmap.fromImage(image)
                self.image_label.setPixmap(self.current_pixmap.scaled(
                    380, 380,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))

                self.generate_demo_results()

            except Exception as e:
                error_msg = f"""
                图片加载失败详情：
                路径: {file_path}
                错误类型: {type(e).__name__}
                详细信息: {str(e)}
                """
                QMessageBox.critical(self, "加载错误", error_msg)
                # 安全清空显示
                self.image_label.clear()
                self.result_table.clearContents()
                self.result_table.setRowCount(0)


    def toggle_camera(self):
        if not self.camera:
            self.camera = cv2.VideoCapture(0)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.sender().setText("关闭摄像头")
        else:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.sender().setText("启动摄像头")
            self.image_label.clear()

    def update_frame(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                try:
                    # 转换颜色空间
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 创建独立拷贝
                    temp_image = QImage(frame.data, frame.shape[1], frame.shape[0],
                                        frame.strides[0], QImage.Format_RGB888).copy()

                    # 显示图像
                    self.image_label.setPixmap(
                        QPixmap.fromImage(temp_image).scaled(
                            380, 380,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                    )
                except Exception as e:
                    print(f"摄像头帧处理错误: {str(e)}")

    def generate_demo_results(self):
        """生成模拟分类结果"""
        # 创建随机概率数据
        fake_probs = [random.random() for _ in range(5)]
        total = sum(fake_probs)
        probabilities = [p / total for p in fake_probs]
        sorted_probs = sorted(probabilities, reverse=True)

        # 显示结果
        self.show_results(sorted_probs)

    def show_results(self, probabilities):
        """更新结果表格（演示版）"""
        self.result_table.setRowCount(5)

        for row in range(5):
            # 随机选择类别
            class_name = QTableWidgetItem(random.choice(self.classes))
            # 生成格式化百分比
            confidence = QTableWidgetItem(f"{probabilities[row] * 100:.2f}%")

            # 设置颜色渐变效果
            color_value = int(255 * (1 - probabilities[row]))
            confidence.setForeground(QColor(color_value, 200, color_value))

            self.result_table.setItem(row, 0, class_name)
            self.result_table.setItem(row, 1, confidence)



if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 检查支持的图片格式
    supported_formats = QImageReader.supportedImageFormats()
    print("支持的图像格式:", [str(fmt, 'utf-8') for fmt in supported_formats])

    window = FlowerClassificationApp()
    window.show()
    sys.exit(app.exec_())