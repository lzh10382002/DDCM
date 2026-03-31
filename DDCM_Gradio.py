import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 导入你的引擎 (Gradio 用原生 python 运行，基本不会迷路)
from Virtual_Engine import VirtualSoilEngine

# 全局加载模型 (只需加载一次，后续推演飞快)
MODEL_DIR = './GCNN_ann_baseline_outputasinput_C_all_train_6_4_3_11_1'
print("正在初始化引擎...")
try:
    engine = VirtualSoilEngine(model_dir=MODEL_DIR, device='cpu')
except Exception as e:
    engine = None
    print(f"引擎加载失败: {e}")


# ================= 核心推演函数 =================
def run_simulation(p0, e0, ocr, amp_pct, cycles, pts, rate):
    if engine is None:
        raise gr.Error("模型未成功加载，请检查终端报错！")

    # 1. 参数转换
    amp = amp_pct / 100.0

    # 2. 调用引擎
    eps, deps = engine.generate_strain_protocol(amp, int(cycles), points_per_cycle=int(pts))
    e_out, p_out, q_out = engine.run_virtual_experiment(p0, 0.0, e0, ocr, eps, deps, rate=rate)

    # 3. 绘制 Matplotlib 图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(e_out * 100, q_out, color='#e74c3c', linewidth=1.5)
    ax1.set_xlabel('Axial Strain (%)');
    ax1.set_ylabel('Stress Q (kPa)')
    ax1.set_title('Cyclic Hysteresis');
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(p_out, q_out, color='#3498db', linewidth=1.5)
    ax2.scatter([p0], [0.0], color='black', zorder=5)
    ax2.set_xlabel('Mean Stress P (kPa)');
    ax2.set_ylabel('Stress Q (kPa)')
    ax2.set_title('Effective Stress Path (P-Q)');
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 4. 生成 CSV 文件
    df_result = pd.DataFrame({
        'Step': np.arange(len(e_out)),
        'Axial_Strain_eps1': e_out,
        'Mean_Stress_P_kPa': p_out,
        'Deviatoric_Stress_Q_kPa': q_out
    })
    csv_path = "DDCM_Result.csv"
    df_result.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return fig, csv_path


# ================= 构建网页 UI =================
with gr.Blocks(theme=gr.themes.Soft(), title="DDCM 数字孪生系统") as demo:
    gr.Markdown("# 🌍 数据驱动智能土体本构系统 (DDCM) - Gradio版")
    gr.Markdown("基于多任务 Transformer 与物理能量约束的 AI 虚拟试验机")

    with gr.Row():
        # 左侧面板：参数输入
        with gr.Column(scale=1):
            gr.Markdown("### 🧪 1. 土体初始状态 (DNA)")
            p0 = gr.Number(value=200.0, label="初始围压 P0 (kPa)")
            e0 = gr.Number(value=1.224, label="初始孔隙比 e0", step=0.01)
            ocr = gr.Number(value=1.0, label="超固结比 OCR")

            gr.Markdown("### ⚙️ 2. 加载控制协议")
            amp = gr.Number(value=0.19, label="轴向应变振幅 (%)")
            cyc = gr.Number(value=50, label="循环圈数")
            pts = gr.Number(value=200, label="单圈推演点数 (精度)")
            rate = gr.Number(value=0.1, label="应变速率 dot_eps")

            run_btn = gr.Button("🚀 开始数字孪生推演", variant="primary")

        # 右侧面板：结果输出
        with gr.Column(scale=2):
            output_plot = gr.Plot(label="推演结果曲线")
            output_file = gr.File(label="📥 下载实验数据 (CSV)")

    # 绑定按钮与函数
    run_btn.click(
        fn=run_simulation,
        inputs=[p0, e0, ocr, amp, cyc, pts, rate],
        outputs=[output_plot, output_file]
    )

# 启动网页 (share=True 会生成一个公网临时链接！)
if __name__ == "__main__":
    demo.launch(share=False)
    # 如果你想外网访问，把上面那行改成 demo.launch(share=True) 即可！