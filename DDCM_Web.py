import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 导入你的底层物理引擎
try:
    from Virtual_Engine import VirtualSoilEngine
except ImportError:
    st.error("[报错] 找不到 Virtual_Engine.py，请确保它在同一文件夹下！")

# ================= 1. 网页全局配置 =================
st.set_page_config(page_title="DDCM 数字孪生本构", page_icon="🌍", layout="wide")
st.title("🌍 数据驱动智能土体本构系统 (DDCM Web版)")
st.markdown("基于多任务 Transformer 与物理能量约束的 AI 虚拟试验机")
st.divider()

# ================= 2. 左侧侧边栏：参数控制台 =================
st.sidebar.header("🧪 1. 土体初始状态 (DNA)")
p0 = st.sidebar.number_input("初始围压 P0 (kPa)", min_value=10.0, max_value=1000.0, value=200.0, step=10.0)
e0 = st.sidebar.number_input("初始孔隙比 e0", min_value=0.1, max_value=2.5, value=1.224, step=0.01)
ocr = st.sidebar.number_input("超固结比 OCR", min_value=1.0, max_value=10.0, value=1.0, step=0.1)

st.sidebar.header("⚙️ 2. 加载控制协议")
amp_percent = st.sidebar.number_input("轴向应变振幅 (%)", min_value=0.01, max_value=5.0, value=0.19, step=0.01)
cycles = st.sidebar.number_input("循环圈数", min_value=1, max_value=1000, value=50, step=10)
pts = st.sidebar.number_input("单圈推演点数 (精度)", min_value=50, max_value=5000, value=200, step=50)

# 🌟 新增的应变速率控制
rate = st.sidebar.number_input("应变速率 dot_eps", min_value=0.001, max_value=10.0, value=0.1, step=0.01,
                               help="对应训练集中的加载速率，影响 P 值的耗散快慢")

st.sidebar.info("💡 提示：在左侧调整参数后，点击右侧的主按钮即可在云端运行数字孪生推演。")

# ================= 3. 主界面：运行与结果展示 =================
MODEL_DIR = './GCNN_ann_baseline_outputasinput_C_all_train_6_4_3_11_1'

# 检查模型路径
if not os.path.exists(MODEL_DIR):
    st.error(f"❌ 找不到模型文件夹：{MODEL_DIR}。请检查路径！")
else:
    # 放置一个极其醒目的主运行按钮
    if st.button("🚀 开始数字孪生推演", type="primary", use_container_width=True):

        # 友好的加载动画
        with st.spinner('🧠 神经网络正在进行高频自回归推演，请稍候...'):
            try:
                # 1. 唤醒引擎
                engine = VirtualSoilEngine(model_dir=MODEL_DIR)

                # 2. 生成协议 (百分比转换为绝对小数)
                amp_absolute = amp_percent / 100.0
                eps_seq, deps_seq = engine.generate_strain_protocol(amp_absolute, cycles, points_per_cycle=pts)

                # 3. 开始预测 (🌟 这里传入了 rate 参数！)
                eps_out, p_out, q_out = engine.run_virtual_experiment(p0, 0.0, e0, ocr, eps_seq, deps_seq, rate=rate)

                # 4. 绘图展示
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#ffffff')

                # 图 1：滞回环
                ax1.plot(eps_out * 100, q_out, color='#e74c3c', linewidth=1.5)
                ax1.set_xlabel('Axial Strain (%)', fontsize=11)
                ax1.set_ylabel('Deviatoric Stress Q (kPa)', fontsize=11)
                ax1.set_title('Cyclic Hysteresis', fontsize=13)
                ax1.grid(True, linestyle='--', alpha=0.5)

                # 图 2：应力路径
                ax2.plot(p_out, q_out, color='#3498db', linewidth=1.5)
                ax2.scatter([p0], [0.0], color='black', zorder=5, label='Initial State')
                ax2.set_xlabel('Mean Effective Stress P (kPa)', fontsize=11)
                ax2.set_ylabel('Deviatoric Stress Q (kPa)', fontsize=11)
                ax2.set_title('Effective Stress Path (P-Q)', fontsize=13)
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.5)

                # 将图表渲染到网页上
                st.pyplot(fig)
                st.success("🎉 推演完成！神经网络已成功预测该参数下的本构响应。")

                # ================= 4. 网页版一键导出 CSV =================
                # 将数据打包成 DataFrame
                df_result = pd.DataFrame({
                    'Step': np.arange(len(eps_out)),
                    'Axial_Strain_eps1': eps_out,
                    'Mean_Stress_P_kPa': p_out,
                    'Deviatoric_Stress_Q_kPa': q_out
                })
                # 转换为 CSV 格式 (utf-8-sig 保证 Excel 打开不乱码)
                csv_data = df_result.to_csv(index=False).encode('utf-8-sig')

                # Streamlit 原生的下载按钮，极其优雅
                st.download_button(
                    label="💾 点击下载完整实验数据 (CSV)",
                    data=csv_data,
                    file_name="DDCM_Web_Result.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"推演过程中发生错误: {e}")