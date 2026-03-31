import sys
import os
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QGroupBox, QPushButton,
                             QDoubleSpinBox, QSpinBox, QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt

# 导入 Matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入你的引擎
try:
    from Virtual_Engine import VirtualSoilEngine
except ImportError:
    print("[报错] 找不到 Virtual_Engine.py！")


# ==========================================
# 1. 绘图画布类
# ==========================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor='#f0f0f0')
        self.axes1 = self.fig.add_subplot(121)
        self.axes2 = self.fig.add_subplot(122)
        self.fig.tight_layout(pad=4.0)
        super(MplCanvas, self).__init__(self.fig)


# ==========================================
# 2. 主窗口类
# ==========================================
class DDCM_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DDCM 数字孪生本构分析系统 v1.2")
        self.setMinimumSize(1300, 700)

        self.engine = None
        self.latest_df = None

        self.initUI()
        self.load_model()

    def load_model(self):
        # 🌟 默认路径对齐你最新的模型
        path = './GCNN_ann_baseline_outputasinput_C_all_train_6_4_3_11_1'
        if os.path.exists(path):
            try:
                self.engine = VirtualSoilEngine(model_dir=path)
                self.statusBar().showMessage(f"✅ 模型加载成功: {path}")
            except Exception as e:
                self.statusBar().showMessage(f"❌ 模型加载失败: {str(e)}")
        else:
            self.statusBar().showMessage("⚠️ 未找到模型，请检查路径。")

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # --- 左侧控制台 ---
        panel = QVBoxLayout()

        # DNA 模块
        box_dna = QGroupBox("🧪 土体初始 DNA")
        f_dna = QFormLayout()
        self.sp_p0 = QDoubleSpinBox();
        self.sp_p0.setRange(0, 1000);
        self.sp_p0.setValue(200.0);
        self.sp_p0.setSuffix(" kPa")
        self.sp_e0 = QDoubleSpinBox();
        self.sp_e0.setRange(0.1, 2.5);
        self.sp_e0.setValue(1.224);
        self.sp_e0.setSingleStep(0.01)
        self.sp_ocr = QDoubleSpinBox();
        self.sp_ocr.setRange(1, 10);
        self.sp_ocr.setValue(1.0)
        f_dna.addRow("初始围压 P0:", self.sp_p0)
        f_dna.addRow("初始孔隙比 e0:", self.sp_e0)
        f_dna.addRow("超固结比 OCR:", self.sp_ocr)
        box_dna.setLayout(f_dna)

        # 加载协议模块
        box_load = QGroupBox("⚙️ 加载控制协议")
        f_load = QFormLayout()
        self.sp_amp = QDoubleSpinBox();
        self.sp_amp.setRange(0.01, 5.0);
        self.sp_amp.setValue(0.19);
        self.sp_amp.setSuffix(" %")
        self.sp_cyc = QSpinBox();
        self.sp_cyc.setRange(1, 1000);
        self.sp_cyc.setValue(50)
        # 🌟 新增：推演精度控制 (解决 P 值骤降)
        self.sp_pts = QSpinBox();
        self.sp_pts.setRange(50, 5000);
        self.sp_pts.setValue(200)
        # 🌟 新增：应变速率控制 (dot_eps)
        self.sp_rate = QDoubleSpinBox()
        self.sp_rate.setRange(0.001, 10.0)
        self.sp_rate.setValue(0.1)  # 默认设置为 0.1，对齐 C03 数据
        self.sp_rate.setSingleStep(0.01)
        self.sp_rate.setToolTip("对应训练集中的 dot_eps 参数")
        f_load.addRow("轴向应变振幅:", self.sp_amp)
        f_load.addRow("循环圈数:", self.sp_cyc)
        f_load.addRow("单圈推演点数:", self.sp_pts)
        f_load.addRow("应变速率 (dot_eps):", self.sp_rate)  # 🌟 把速率加到界面上
        box_load.setLayout(f_load)

        # 运行按钮
        self.btn_run = QPushButton("🚀 开始数字孪生推演")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 8px;")
        self.btn_run.clicked.connect(self.run)

        self.btn_csv = QPushButton("💾 导出实验数据 (CSV)")
        self.btn_csv.setFixedHeight(40)
        self.btn_csv.setStyleSheet("background-color: #2980b9; color: white; border-radius: 8px;")
        self.btn_csv.setEnabled(False)
        self.btn_csv.clicked.connect(self.export)

        panel.addWidget(box_dna)
        panel.addWidget(box_load)
        panel.addStretch()
        panel.addWidget(self.btn_run)
        panel.addWidget(self.btn_csv)

        # --- 右侧绘图区 ---
        self.canvas = MplCanvas()

        layout.addLayout(panel, 1)
        layout.addWidget(self.canvas, 4)

    def run(self):
        if not self.engine: return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("⌛ 推演中...")
        QApplication.processEvents()

        try:
            # 1. 转换参数 (GUI 的 0.19% -> 引擎的 0.0019)
            amp = self.sp_amp.value() / 100.0
            p0, e0, ocr = self.sp_p0.value(), self.sp_e0.value(), self.sp_ocr.value()
            pts = self.sp_pts.value()
            # 🌟 读取界面上的速率值
            rate_val = self.sp_rate.value()
            # 2. 生成协议
            eps, deps = self.engine.generate_strain_protocol(amp, self.sp_cyc.value(), points_per_cycle=pts)

            # 3. 运行推演
            e_out, p_out, q_out = self.engine.run_virtual_experiment(p0, 0.0, e0, ocr, eps, deps, rate=rate_val)

            # 4. 暂存数据用于导出
            self.latest_df = pd.DataFrame({
                'Strain_eps1': e_out,
                'Stress_P_kPa': p_out,
                'Stress_Q_kPa': q_out
            })
            self.btn_csv.setEnabled(True)

            # 5. 更新图表
            self.plot(e_out, p_out, q_out, p0)
            self.statusBar().showMessage("✅ 虚拟实验推演完成！")

        except Exception as e:
            QMessageBox.critical(self, "推演失败", str(e))
        finally:
            self.btn_run.setEnabled(True)
            self.btn_run.setText("🚀 开始数字孪生推演")

    def plot(self, e, p, q, p0):
        # 滞回环
        self.canvas.axes1.cla()
        self.canvas.axes1.plot(e * 100, q, '#e74c3c', linewidth=1.2)
        self.canvas.axes1.set_xlabel("Axial Strain (%)");
        self.canvas.axes1.set_ylabel("Stress Q (kPa)")
        self.canvas.axes1.set_title("Cyclic Hysteresis")
        self.canvas.axes1.grid(True, alpha=0.3)

        # 路径图
        self.canvas.axes2.cla()
        self.canvas.axes2.plot(p, q, '#3498db', linewidth=1.2)
        self.canvas.axes2.scatter([p0], [0], color='black', zorder=5)
        self.canvas.axes2.set_xlabel("Mean Stress P (kPa)");
        self.canvas.axes2.set_ylabel("Stress Q (kPa)")
        self.canvas.axes2.set_title("Effective Stress Path")
        self.canvas.axes2.grid(True, alpha=0.3)

        self.canvas.draw()

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存数据", "DDCM_Result.csv", "CSV Files (*.csv)")
        if path:
            self.latest_df.to_csv(path, index=False, encoding='utf-8-sig')
            QMessageBox.information(self, "成功", "实验数据已导出！")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DDCM_MainWindow()
    win.show()
    sys.exit(app.exec())