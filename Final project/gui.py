import sys
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QHBoxLayout, QComboBox, QFrame, QLineEdit, QTextEdit
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import cv2
from filter import Filter
from transformation import Transformation
from resize import Resize
from info import Info
from analysis import Analysis
from compress import Compress

class ImageDropWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Image GUI")
        self.resize(1000, 600)
        self.setAcceptDrops(True)

        self.current_image_path = None
        self.filtered_img = None
        self.transformed_img = None
        self.resized_img = None
        self.compressed_data = None
        self.compression_method = None
        
        self.last_filter_choice = "None"
        self.last_filter_params = {}
        self.last_resize_choice = "None"
        self.last_resize_params = {}
        self.last_trans_choice = "None"
        self.last_trans_params = {}

        self.menu_layout = QVBoxLayout()

        self.filter_label = QLabel("Filters")
        self.menu_layout.addWidget(self.filter_label)

        self.combo1 = QComboBox()
        self.combo1.addItems([
            "None", "Gray", "Blur", "Median", "Laplacian", "Sobel",
            "Gradient", "Sharpen", "Binary", "Adjust"
        ])
        self.combo1.currentTextChanged.connect(self.show_filter_inputs)
        self.menu_layout.addWidget(self.combo1)

        self.filter_input1 = QLineEdit()
        self.filter_input1.setPlaceholderText("Param 1")
        self.filter_input2 = QLineEdit()
        self.filter_input2.setPlaceholderText("Param 2")
        self.filter_input3 = QLineEdit()
        self.filter_input3.setPlaceholderText("Param 3")

        for widget in (self.filter_input1, self.filter_input2, self.filter_input3):
            widget.hide()
            self.menu_layout.addWidget(widget)

        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        self.menu_layout.addWidget(self.apply_filter_btn)

        self.resize_label = QLabel("Resize")
        self.menu_layout.addWidget(self.resize_label)

        self.combo_resize = QComboBox()
        self.combo_resize.addItems([
            "None", "Default", "Nearest Neighbor", "Bilinear", "Bicubic"
        ])
        self.combo_resize.currentTextChanged.connect(self.show_resize_inputs)
        self.menu_layout.addWidget(self.combo_resize)

        self.resize_input1 = QLineEdit()
        self.resize_input1.setPlaceholderText("Width")
        self.resize_input2 = QLineEdit()
        self.resize_input2.setPlaceholderText("Height")

        for widget in (self.resize_input1, self.resize_input2):
            widget.hide()
            self.menu_layout.addWidget(widget)

        self.apply_resize_btn = QPushButton("Apply Resize")
        self.apply_resize_btn.clicked.connect(self.apply_resize)
        self.menu_layout.addWidget(self.apply_resize_btn)

        self.trans_label = QLabel("Transformations")
        self.menu_layout.addWidget(self.trans_label)

        self.combo_trans = QComboBox()
        self.combo_trans.addItems([
            "None", "Rotate", "Crop", "Shear X", "Shear Y", "Translate"
        ])
        self.combo_trans.currentTextChanged.connect(self.show_trans_inputs)
        self.menu_layout.addWidget(self.combo_trans)

        self.trans_input1 = QLineEdit()
        self.trans_input1.setPlaceholderText("Param 1")
        self.trans_input2 = QLineEdit()
        self.trans_input2.setPlaceholderText("Param 2")
        self.trans_input3 = QLineEdit()
        self.trans_input3.setPlaceholderText("Param 3")
        self.trans_input4 = QLineEdit()
        self.trans_input4.setPlaceholderText("Param 4")

        for widget in (self.trans_input1, self.trans_input2, self.trans_input3, self.trans_input4):
            widget.hide()
            self.menu_layout.addWidget(widget)

        self.apply_transform_btn = QPushButton("Apply Transformation")
        self.apply_transform_btn.clicked.connect(self.apply_transformation)
        self.menu_layout.addWidget(self.apply_transform_btn)

        self.analysis_label = QLabel("Analysis")
        self.menu_layout.addWidget(self.analysis_label)

        self.combo_analysis = QComboBox()
        self.combo_analysis.addItems([
            "None", "Threshold Analysis", "Histogram"
        ])
        self.combo_analysis.currentTextChanged.connect(self.perform_analysis)
        self.menu_layout.addWidget(self.combo_analysis)

        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(200)
        self.info_display.setPlaceholderText("Image info will appear here")
        self.menu_layout.addWidget(self.info_display)

        self.compress_label = QLabel("Compression")
        self.menu_layout.addWidget(self.compress_label)

        self.combo_compress = QComboBox()
        self.combo_compress.addItems([
            "None", "Huffman", "Golomb-Rice", "Arithmetic", "LZW", 
            "RLE", "Symbol-Based", "Bit-Plane", "DCT", "Predictive", "Wavelet"
        ])
        self.combo_compress.currentTextChanged.connect(self.show_compress_inputs)
        self.menu_layout.addWidget(self.combo_compress)

        self.compress_input1 = QLineEdit()
        self.compress_input1.setPlaceholderText("Param 1")
        self.compress_input1.hide()
        self.menu_layout.addWidget(self.compress_input1)

        self.apply_compress_btn = QPushButton("Apply Compression")
        self.apply_compress_btn.clicked.connect(self.apply_compression)
        self.menu_layout.addWidget(self.apply_compress_btn)

        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        self.menu_layout.addWidget(self.save_btn)
        self.menu_layout.addStretch()

        self.drop_label = QLabel("Drag an image or Browse")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setFrameShape(QFrame.Shape.Box)

        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFrameShape(QFrame.Shape.Box)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.open_file)

        right_layout = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(self.drop_label)
        top.addWidget(self.preview_label)
        right_layout.addLayout(top)
        right_layout.addWidget(self.browse_btn)

        main = QHBoxLayout()
        main.addLayout(self.menu_layout, 1)
        main.addLayout(right_layout, 3)
        self.setLayout(main)

    def cv_to_pixmap(self, img):
        if img is None:
            return QPixmap()
        if len(img.shape) == 2:
            qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            self.load_image(url.toLocalFile())
            break

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        self.current_image_path = path
        pix = QPixmap(path)
        if not pix.isNull():
            scaled1 = pix.scaled(self.drop_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            scaled2 = pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio)

            self.drop_label.setPixmap(scaled1)
            self.preview_label.setPixmap(scaled2)

            self.update_info()
            self.show_filter_inputs()

    def show_filter_inputs(self):
        for w in (self.filter_input1, self.filter_input2, self.filter_input3):
            w.hide()

        choice = self.combo1.currentText()

        if choice == "Blur":
            self.filter_input1.setPlaceholderText("Kernel size (odd, e.g., 19)")
            self.filter_input2.setPlaceholderText("Sigma (e.g., 3)")
            self.filter_input1.show()
            self.filter_input2.show()
        elif choice == "Median":
            self.filter_input1.setPlaceholderText("Kernel size (odd, e.g., 7)")
            self.filter_input1.show()
        elif choice == "Laplacian":
            self.filter_input1.setPlaceholderText("Kernel size (odd, e.g., 3)")
            self.filter_input1.show()
        elif choice == "Sobel":
            self.filter_input1.setPlaceholderText("Kernel size (odd, e.g., 3)")
            self.filter_input1.show()
        elif choice == "Adjust":
            self.filter_input1.setPlaceholderText("Brightness (e.g., 0)")
            self.filter_input2.setPlaceholderText("Contrast (e.g., 0)")
            self.filter_input1.show()
            self.filter_input2.show()

        if choice in ["None", "Gray", "Gradient", "Sharpen", "Binary"]:
            self.apply_filter()

    def apply_filter(self):
        if not self.current_image_path:
            return

        f = Filter(self.current_image_path)
        choice = self.combo1.currentText()

        try:
            if choice == "Gray":
                img = f.to_gray()
            elif choice == "Blur":
                k = int(self.filter_input1.text()) if self.filter_input1.text() else 19
                sigma = int(self.filter_input2.text()) if self.filter_input2.text() else 3
                img = f.to_blur(k, sigma)
                self.last_filter_params = {'k': k, 'sigma': sigma}
            elif choice == "Median":
                k = int(self.filter_input1.text()) if self.filter_input1.text() else 7
                img = f.median(k)
                self.last_filter_params = {'k': k}
            elif choice == "Laplacian":
                k = int(self.filter_input1.text()) if self.filter_input1.text() else 3
                img = f.laplacian(k)
                self.last_filter_params = {'k': k}
            elif choice == "Sobel":
                k = int(self.filter_input1.text()) if self.filter_input1.text() else 3
                img = f.sobel(k)
                self.last_filter_params = {'k': k}
            elif choice == "Gradient":
                img = f.gradient()
            elif choice == "Sharpen":
                img = f.sharpen()
            elif choice == "Binary":
                img, _ = f.to_binary()
            elif choice == "Adjust":
                brightness = int(self.filter_input1.text()) if self.filter_input1.text() else 0
                contrast = int(self.filter_input2.text()) if self.filter_input2.text() else 0
                img = f.adjust(brightness, contrast)
                self.last_filter_params = {'brightness': brightness, 'contrast': contrast}
            else:
                img = f.img
        except:
            img = f.img

        self.filtered_img = img
        self.last_filter_choice = choice
        
        self.reapply_pipeline_from_filter()

    def reapply_pipeline_from_filter(self):
        if self.last_resize_choice != "None":
            self.reapply_resize()
        else:
            self.resized_img = None
            
        if self.last_trans_choice != "None":
            self.reapply_transformation()
        else:
            self.transformed_img = None
            
        self.update_preview()
        self.update_info()

    def reapply_resize(self):
        if self.filtered_img is None or self.last_resize_choice == "None":
            return
            
        temp_path = "/tmp/temp_for_resize.png"
        cv2.imwrite(temp_path, self.filtered_img)

        r = Resize(temp_path)
        
        try:
            width = self.last_resize_params.get('width')
            height = self.last_resize_params.get('height')
            
            if width and height:
                if self.last_resize_choice == "Default":
                    self.resized_img = r.resize(width, height)
                elif self.last_resize_choice == "Nearest Neighbor":
                    self.resized_img = r.resize_nn(width, height)
                elif self.last_resize_choice == "Bilinear":
                    self.resized_img = r.resize_bilinear(width, height)
                elif self.last_resize_choice == "Bicubic":
                    self.resized_img = r.resize_bicubic(width, height)
        except:
            pass

    def reapply_transformation(self):
        base_img = self.resized_img if self.resized_img is not None else self.filtered_img
        if base_img is None or self.last_trans_choice == "None":
            return

        temp_path = "/tmp/temp_filtered.png"
        cv2.imwrite(temp_path, base_img)

        t = Transformation(temp_path)
        
        try:
            if self.last_trans_choice == "Rotate":
                angle = self.last_trans_params.get('angle')
                if angle is not None:
                    self.transformed_img = t.rotate(angle)
            elif self.last_trans_choice == "Crop":
                x = self.last_trans_params.get('x')
                y = self.last_trans_params.get('y')
                w = self.last_trans_params.get('w')
                h = self.last_trans_params.get('h')
                if all(v is not None for v in [x, y, w, h]):
                    self.transformed_img = t.crop(x, y, w, h)
            elif self.last_trans_choice == "Shear X":
                f = self.last_trans_params.get('factor')
                if f is not None:
                    self.transformed_img = t.shear_x(f)
            elif self.last_trans_choice == "Shear Y":
                f = self.last_trans_params.get('factor')
                if f is not None:
                    self.transformed_img = t.shear_y(f)
            elif self.last_trans_choice == "Translate":
                tx = self.last_trans_params.get('tx')
                ty = self.last_trans_params.get('ty')
                if tx is not None and ty is not None:
                    self.transformed_img = t.translate(tx, ty)
        except:
            pass

    def update_preview(self):
        img = self.transformed_img if self.transformed_img is not None else (
            self.resized_img if self.resized_img is not None else self.filtered_img
        )
        if img is not None:
            pix = self.cv_to_pixmap(img)
            self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def show_resize_inputs(self):
        for w in (self.resize_input1, self.resize_input2):
            w.hide()

        choice = self.combo_resize.currentText()

        if choice in ["Default", "Nearest Neighbor", "Bilinear", "Bicubic"]:
            self.resize_input1.setPlaceholderText("Width")
            self.resize_input2.setPlaceholderText("Height")
            self.resize_input1.show()
            self.resize_input2.show()

    def apply_resize(self):
        base_img = self.filtered_img if self.filtered_img is not None else None
        if not self.current_image_path or base_img is None:
            return

        temp_path = "/tmp/temp_for_resize.png"
        cv2.imwrite(temp_path, base_img)

        r = Resize(temp_path)
        choice = self.combo_resize.currentText()

        try:
            if choice in ["Default", "Nearest Neighbor", "Bilinear", "Bicubic"]:
                width = int(self.resize_input1.text())
                height = int(self.resize_input2.text())
                
                self.last_resize_choice = choice
                self.last_resize_params = {'width': width, 'height': height}

                if choice == "Default":
                    img = r.resize(width, height)
                elif choice == "Nearest Neighbor":
                    img = r.resize_nn(width, height)
                elif choice == "Bilinear":
                    img = r.resize_bilinear(width, height)
                elif choice == "Bicubic":
                    img = r.resize_bicubic(width, height)
            else:
                return
        except:
            return

        self.resized_img = img
        
        if self.last_trans_choice != "None":
            self.reapply_transformation()
        else:
            self.transformed_img = None
            
        self.update_preview()
        self.update_info()

    def show_trans_inputs(self):
        for w in (self.trans_input1, self.trans_input2, self.trans_input3, self.trans_input4):
            w.hide()

        choice = self.combo_trans.currentText()

        if choice == "Rotate":
            self.trans_input1.setPlaceholderText("Angle")
            self.trans_input1.show()
        elif choice == "Crop":
            self.trans_input1.setPlaceholderText("x")
            self.trans_input2.setPlaceholderText("y")
            self.trans_input3.setPlaceholderText("w")
            self.trans_input4.setPlaceholderText("h")
            for w in (self.trans_input1, self.trans_input2, self.trans_input3, self.trans_input4):
                w.show()
        elif choice in ("Shear X", "Shear Y"):
            self.trans_input1.setPlaceholderText("Factor")
            self.trans_input1.show()
        elif choice == "Translate":
            self.trans_input1.setPlaceholderText("tx")
            self.trans_input2.setPlaceholderText("ty")
            self.trans_input1.show()
            self.trans_input2.show()

    def apply_transformation(self):
        base_img = self.resized_img if self.resized_img is not None else self.filtered_img
        if base_img is None:
            return

        temp_path = "/tmp/temp_filtered.png"
        cv2.imwrite(temp_path, base_img)

        t = Transformation(temp_path)
        c = self.combo_trans.currentText()

        try:
            if c == "Rotate":
                angle = float(self.trans_input1.text())
                img = t.rotate(angle)
                self.last_trans_choice = c
                self.last_trans_params = {'angle': angle}
            elif c == "Crop":
                x = int(self.trans_input1.text())
                y = int(self.trans_input2.text())
                w = int(self.trans_input3.text())
                h = int(self.trans_input4.text())
                img = t.crop(x, y, w, h)
                self.last_trans_choice = c
                self.last_trans_params = {'x': x, 'y': y, 'w': w, 'h': h}
            elif c == "Shear X":
                f = float(self.trans_input1.text())
                img = t.shear_x(f)
                self.last_trans_choice = c
                self.last_trans_params = {'factor': f}
            elif c == "Shear Y":
                f = float(self.trans_input1.text())
                img = t.shear_y(f)
                self.last_trans_choice = c
                self.last_trans_params = {'factor': f}
            elif c == "Translate":
                tx = int(self.trans_input1.text())
                ty = int(self.trans_input2.text())
                img = t.translate(tx, ty)
                self.last_trans_choice = c
                self.last_trans_params = {'tx': tx, 'ty': ty}
            else:
                return
        except:
            return

        self.transformed_img = img
        self.update_preview()
        self.update_info()

    def perform_analysis(self):
        if not self.current_image_path:
            return

        choice = self.combo_analysis.currentText()
        
        if choice == "None":
            self.update_info()
            return

        current_img = self.transformed_img if self.transformed_img is not None else (
            self.resized_img if self.resized_img is not None else (
                self.filtered_img if self.filtered_img is not None else None
            )
        )

        if current_img is None:
            temp_path = self.current_image_path
        else:
            temp_path = "/tmp/temp_analysis.png"
            cv2.imwrite(temp_path, current_img)

        try:
            a = Analysis(temp_path)
            info = Info(temp_path)
            width, height = info.get_resolution()
            size = info.get_size()
            file_type = info.get_type()
            channels = info.get_channel()

            info_text = f"""IMAGE INFORMATION
━━━━━━━━━━━━━━━━━━━━
Resolution: {width} x {height}
File Size: {size} MB
Type: {file_type}
Channels: {channels}

"""

            if choice == "Threshold Analysis":
                result = a.compute_threshold()
                info_text += f"""THRESHOLD ANALYSIS
━━━━━━━━━━━━━━━━━━━━
Average Threshold: {result['average_threshold']:.2f}
Otsu Threshold: {result['otsu_threshold']:.2f}
Difference: {result['difference']:.2f}
Optimal: {"Yes" if result['is_optimal'] else "No"}
"""
            elif choice == "Histogram":
                hist = a.compute_histogram()
                mean_val = hist.mean()
                max_val = hist.max()
                min_val = hist.min()
                info_text += f"""HISTOGRAM ANALYSIS
━━━━━━━━━━━━━━━━━━━━
Mean: {mean_val:.2f}
Max: {max_val:.2f}
Min: {min_val:.2f}
Total Bins: 256
"""

            self.info_display.setText(info_text)
        except Exception as e:
            self.info_display.setText(f"Analysis error: {str(e)}")

    def update_info(self):
        if not self.current_image_path:
            return

        current_img = self.transformed_img if self.transformed_img is not None else (
            self.resized_img if self.resized_img is not None else (
                self.filtered_img if self.filtered_img is not None else None
            )
        )

        if current_img is None:
            temp_path = self.current_image_path
        else:
            temp_path = "/tmp/temp_info.png"
            cv2.imwrite(temp_path, current_img)

        try:
            info = Info(temp_path)
            width, height = info.get_resolution()
            size = info.get_size()
            file_type = info.get_type()
            channels = info.get_channel()

            info_text = f"""IMAGE INFORMATION
━━━━━━━━━━━━━━━━━━━━
Resolution: {width} x {height}
File Size: {size} MB
Type: {file_type}
Channels: {channels}
"""
            self.info_display.setText(info_text)
        except:
            pass

    def show_compress_inputs(self):
        self.compress_input1.hide()
        
        choice = self.combo_compress.currentText()
        
        if choice == "Golomb-Rice":
            self.compress_input1.setPlaceholderText("M (power of 2, e.g., 4)")
            self.compress_input1.show()
        elif choice == "DCT":
            self.compress_input1.setPlaceholderText("Block size (e.g., 8)")
            self.compress_input1.show()
        elif choice == "Predictive":
            self.compress_input1.setPlaceholderText("Mode: left/top/avg")
            self.compress_input1.show()
        elif choice == "Wavelet":
            self.compress_input1.setPlaceholderText("Level (e.g., 1)")
            self.compress_input1.show()

    def apply_compression(self):
        current_img = self.transformed_img if self.transformed_img is not None else (
            self.resized_img if self.resized_img is not None else (
                self.filtered_img if self.filtered_img is not None else None
            )
        )
        
        if current_img is None and not self.current_image_path:
            return
            
        if current_img is None:
            temp_path = self.current_image_path
        else:
            temp_path = "/tmp/temp_compress.png"
            cv2.imwrite(temp_path, current_img)
        
        try:
            comp = Compress(temp_path)
            choice = self.combo_compress.currentText()
            
            if choice == "Huffman":
                encoded, huff_map = comp.huffman()
                stats = comp.get_compression_stats(encoded)
                self.compressed_data = (encoded, huff_map)
                self.compression_method = "huffman"
                self.show_compression_stats(stats, choice)
                
            elif choice == "Golomb-Rice":
                M = int(self.compress_input1.text()) if self.compress_input1.text() else 4
                encoded = comp.golomb_rice(M)
                encoded_str = "".join(encoded)
                stats = comp.get_compression_stats(encoded_str)
                self.compressed_data = encoded
                self.compression_method = "golomb"
                self.show_compression_stats(stats, choice)
                
            elif choice == "Arithmetic":
                encoded, cum = comp.arithmetic()
                self.compressed_data = (encoded, cum)
                self.compression_method = "arithmetic"
                info_text = f"""COMPRESSION: {choice}
━━━━━━━━━━━━━━━━━━━━
Encoded Value: {encoded:.10f}
Method: Arithmetic Coding
"""
                self.info_display.setText(info_text)
                
            elif choice == "LZW":
                result, dict_size = comp.lzw()
                stats = comp.get_compression_stats(result)
                self.compressed_data = (result, dict_size)
                self.compression_method = "lzw"
                stats['dictionary_size'] = dict_size
                self.show_compression_stats(stats, choice)
                
            elif choice == "RLE":
                encoded = comp.rle()
                stats = comp.get_compression_stats(encoded)
                self.compressed_data = encoded
                self.compression_method = "rle"
                self.show_compression_stats(stats, choice)
                
            elif choice == "Symbol-Based":
                encoded, symbol_map = comp.symbol_based()
                stats = comp.get_compression_stats(encoded)
                self.compressed_data = (encoded, symbol_map)
                self.compression_method = "symbol"
                self.show_compression_stats(stats, choice)
                
            elif choice == "Bit-Plane":
                bit_planes = comp.bit_plane()
                self.compressed_data = bit_planes
                self.compression_method = "bitplane"
                info_text = f"""COMPRESSION: {choice}
━━━━━━━━━━━━━━━━━━━━
Method: Bit-Plane Slicing
Planes Generated: {len(bit_planes)}
"""
                self.info_display.setText(info_text)
                
            elif choice == "DCT":
                block_size = int(self.compress_input1.text()) if self.compress_input1.text() else 8
                dct_result = comp.dct_blocks(block_size)
                self.compressed_data = dct_result
                self.compression_method = "dct"
                info_text = f"""COMPRESSION: {choice}
━━━━━━━━━━━━━━━━━━━━
Method: DCT Transform
Block Size: {block_size}x{block_size}
Output Shape: {dct_result.shape}
"""
                self.info_display.setText(info_text)
                
            elif choice == "Predictive":
                mode = self.compress_input1.text() if self.compress_input1.text() else 'left'
                residual, predicted = comp.predictive(mode)
                self.compressed_data = (residual, predicted)
                self.compression_method = "predictive"
                info_text = f"""COMPRESSION: {choice}
━━━━━━━━━━━━━━━━━━━━
Method: Predictive Coding
Mode: {mode}
Residual Range: [{residual.min()}, {residual.max()}]
"""
                self.info_display.setText(info_text)
                
            elif choice == "Wavelet":
                level = int(self.compress_input1.text()) if self.compress_input1.text() else 1
                wavelet_result = comp.wavelet(level)
                self.compressed_data = wavelet_result
                self.compression_method = "wavelet"
                info_text = f"""COMPRESSION: {choice}
━━━━━━━━━━━━━━━━━━━━
Method: Wavelet Transform
Level: {level}
Output Shape: {wavelet_result.shape}
"""
                self.info_display.setText(info_text)
                
        except Exception as e:
            self.info_display.setText(f"Compression error: {str(e)}")

    def show_compression_stats(self, stats, method):
        info_text = f"""COMPRESSION: {method}
━━━━━━━━━━━━━━━━━━━━
Original Bits: {stats['original_bits']}
Compressed Bits: {stats['compressed_bits']}
Compression Ratio: {stats['compression_ratio']}:1
Space Saving: {stats['space_saving']}%
"""
        if 'dictionary_size' in stats:
            info_text += f"Dictionary Size: {stats['dictionary_size']}\n"
        
        self.info_display.setText(info_text)

    def save_image(self):
        if self.compression_method:
            path, selected_filter = QFileDialog.getSaveFileName(
                self, 
                "Save Compressed/Decompressed Image", 
                "", 
                "Image Files (*.png *.jpg *.bmp);;NumPy Array (*.npy);;Text File (*.txt)"
            )
            if not path:
                return
                
            try:
                if path.endswith('.npy') or path.endswith('.txt'):
                    if path.endswith('.npy'):
                        if isinstance(self.compressed_data, np.ndarray):
                            np.save(path, self.compressed_data)
                        elif isinstance(self.compressed_data, tuple):
                            np.save(path, self.compressed_data[0])
                        else:
                            np.save(path, np.array(self.compressed_data))
                    else:
                        with open(path, 'w') as f:
                            f.write(str(self.compressed_data))
                else:
                    if not any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
                        if 'Image Files' in selected_filter or 'PNG' in selected_filter:
                            path += '.png'
                        elif 'JPEG' in selected_filter:
                            path += '.jpg'
                        elif 'BMP' in selected_filter:
                            path += '.bmp'
                        else:
                            path += '.png'
                    
                    img_to_save = None
                    
                    if self.compression_method in ["dct", "bitplane", "predictive", "wavelet"]:
                        if self.compression_method == "dct":
                            img_to_save = np.uint8(np.clip(self.compressed_data, 0, 255))
                        elif self.compression_method == "bitplane":
                            reconstructed = np.zeros_like(self.compressed_data[0], dtype=np.uint8)
                            for bit_position, plane in enumerate(self.compressed_data):
                                reconstructed += (plane << bit_position)
                            img_to_save = reconstructed
                        elif self.compression_method == "predictive":
                            residual, predicted = self.compressed_data
                            reconstructed = np.clip(predicted + residual, 0, 255).astype(np.uint8)
                            img_to_save = reconstructed
                        elif self.compression_method == "wavelet":
                            img_to_save = np.uint8(np.clip(self.compressed_data, 0, 255))
                    else:
                        img_to_save = self.transformed_img if self.transformed_img is not None else (
                            self.resized_img if self.resized_img is not None else self.filtered_img
                        )
                    
                    if img_to_save is not None:
                        cv2.imwrite(path, img_to_save)
                    else:
                        print("No image data available to save")
                        
            except Exception as e:
                print(f"Save error: {e}")
        else:
            img = self.transformed_img if self.transformed_img is not None else (
                self.resized_img if self.resized_img is not None else self.filtered_img
            )
            if img is None:
                return

            path, selected_filter = QFileDialog.getSaveFileName(self, "Save Image As", "", "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
            if path:
                if not any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
                    if 'PNG' in selected_filter:
                        path += '.png'
                    elif 'JPEG' in selected_filter:
                        path += '.jpg'
                    elif 'BMP' in selected_filter:
                        path += '.bmp'
                    else:
                        path += '.png'
                
                try:
                    cv2.imwrite(path, img)
                except Exception as e:
                    print(f"Save error: {e}")

app = QApplication(sys.argv)
window = ImageDropWindow()
window.show()
sys.exit(app.exec())