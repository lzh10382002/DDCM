import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
# ======================================================================
# 🌟 1. 接入 11_1_1.py 最新版大脑
# ======================================================================
from GCNN_ANN_Torch_outputasinput_smalltest_Cdata9_4_3_11_1 import ANNTorch, HYPERPARAMS


class VirtualSoilEngine:
    def __init__(self, model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        print(f"[INFO] 正在唤醒数字孪生本构引擎... 读取模型: {model_dir}")
        print(f"[INFO] 运算设备: {self.device}")

        # 实例化你的真实模型
        self.model_wrapper = ANNTorch(model_path=model_dir)
        self.model_wrapper.model.eval()
        self.model_wrapper.model.to(self.device)

        self.seq_len = HYPERPARAMS['seq_len']

        # ==========================================================
        # 🌟 2. 完美的 12 维特征索引对齐！
        # 原数组 input_idx = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
        # ==========================================================
        self.idx_P = 0  # 原 0: P_old
        self.idx_Q = 1  # 原 1: Q_old
        self.idx_eps = 2  # 原 2: 轴向应变 eps1
        self.idx_deps = 3  # 原 3: 应变增量 deps1
        self.idx_P0 = 4  # 原 6: 初始 P0
        self.idx_Q0 = 5  # 原 7: 初始 Q0
        self.idx_e0 = 6  # 原 8: 初始孔隙比 e0
        self.idx_OCR = 7  # 原 9: OCR
        self.idx_col10 = 8  # 原 10: 占位/时间
        self.idx_col11 = 9  # 原 11: 占位/时间
        self.idx_energy = 10  # 原 12: 疲劳时钟 (log_acc_energy)
        self.idx_phase = 11  # 原 13: 加卸载红绿灯 (np.sign)

    def generate_strain_protocol(self, amplitude, cycles, points_per_cycle=1000):
        print(f"[INFO] 虚拟试验机就绪: {cycles} 圈循环加载, 应变振幅 ±{amplitude * 100}%")
        eps_seq = []
        for _ in range(cycles):
            eps_seq.extend(np.linspace(0, amplitude, points_per_cycle // 4, endpoint=False))
            eps_seq.extend(np.linspace(amplitude, -amplitude, points_per_cycle // 2, endpoint=False))
            eps_seq.extend(np.linspace(-amplitude, 0, points_per_cycle // 4, endpoint=False))

        eps_seq = np.array(eps_seq)
        deps_seq = np.insert(np.diff(eps_seq), 0, 0.0)
        return eps_seq, deps_seq

    def run_virtual_experiment(self, P0, Q0, e0, OCR, eps_seq, deps_seq, rate=0.1):
        total_steps = len(eps_seq)
        print(f"[INFO] 引擎点火！开始 {total_steps} 步高频自回归推演...")

        hist_P, hist_Q, hist_energy = [P0], [Q0], [0.0]
        acc_energy = 0.0

        # 🌟 3. 初始化滑动窗口为 12 维！
        current_window = np.zeros((self.seq_len, 12))
        for i in range(self.seq_len):
            current_window[i, self.idx_P] = P0
            current_window[i, self.idx_Q] = Q0
            current_window[i, self.idx_P0] = P0
            current_window[i, self.idx_Q0] = Q0
            current_window[i, self.idx_e0] = e0
            current_window[i, self.idx_OCR] = OCR

        with torch.no_grad():
            for step in range(total_steps):
                curr_eps = eps_seq[step]
                curr_deps = deps_seq[step]

                batch_X = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(self.device)

                # 现在这里过归一化标尺，绝对不会再报错了！
                batch_X_norm = self.model_wrapper.normalize_input(batch_X)

                out_4dim = self.model_wrapper.model(batch_X_norm)
                delta_pred = out_4dim[:, 0:2]
                abs_pred = out_4dim[:, 2:4]

                P_old = current_window[-1, self.idx_P]
                Q_old = current_window[-1, self.idx_Q]

                # ==========================================================
                # 🌟 异构物理纠偏 (P弱拉，Q强拉)
                # ==========================================================
                T_pred_P = P_old + delta_pred[0, 0].item()
                T_pred_Q = Q_old + delta_pred[0, 1].item()

                P_new = (1.0 - 0.02) * T_pred_P + 0.02 * abs_pred[0, 0].item()
                Q_new = (1.0 - 0.15) * T_pred_Q + 0.15 * abs_pred[0, 1].item()

                # 物理护栏
                P_new = max(1.0, P_new)

                # ==========================================================
                # 🌟 4. 同步 11_1_1.py 的核心：疲劳时钟 & 红绿灯
                # ==========================================================
                # 完全还原你的计算公式: incremental_work = p_old * np.abs(deps1)
                step_energy = P_old * abs(curr_deps)
                acc_energy += step_energy
                current_log_energy = np.log1p(acc_energy)

                # 加卸载红绿灯
                current_phase = np.sign(curr_deps)

                hist_P.append(P_new)
                hist_Q.append(Q_new)
                hist_energy.append(acc_energy)

                # ==========================================================
                # 🌟 5. 构建下一帧，填入 12 个新数据！
                # ==========================================================
                next_row = np.zeros(12)
                next_row[self.idx_P] = P_new
                next_row[self.idx_Q] = Q_new
                next_row[self.idx_eps] = eps_seq[step + 1] if step + 1 < total_steps else curr_eps
                next_row[self.idx_deps] = deps_seq[step + 1] if step + 1 < total_steps else 0.0
                next_row[self.idx_P0] = P0
                next_row[self.idx_Q0] = Q0
                next_row[self.idx_e0] = e0
                next_row[self.idx_OCR] = OCR
                next_row[self.idx_col10] = rate
                next_row[self.idx_col11] = 0.0
                next_row[self.idx_energy] = current_log_energy  # 实时疲劳时钟
                next_row[self.idx_phase] = current_phase  # 实时红绿灯

                current_window = np.vstack((current_window[1:], next_row))

                if step % 5000 == 0 and step > 0:
                    print(
                        f"  -> 已推演 {step}/{total_steps} 步 (疲劳时钟: {current_log_energy:.4f}, 红绿灯: {current_phase})")

        return eps_seq, np.array(hist_P[:-1]), np.array(hist_Q[:-1])


# ================= 🚀 测试入口：体验一键生成 =================
if __name__ == "__main__":
    # ⚠️ 请确保这里填的是你最新 11_1_1.py 训练出来的模型保存目录！
    model_directory = './GCNN_ann_baseline_outputasinput_C_all_train_6_4_3_11_1'

    if not os.path.exists(model_directory):
        print(f"[报错] 找不到模型文件夹 {model_directory}！请检查路径。")
    else:
        engine = VirtualSoilEngine(model_dir=model_directory)

        # 生成循环加载方案 (模拟论文 C 系列：20 圈，应变振幅 1%)
        amplitude = 0.0019
        cycles = 50
        eps_seq, deps_seq = engine.generate_strain_protocol(amplitude, cycles, points_per_cycle=1000)

        # 对标论文数据库的 Kaolin 初始状态
        P0 = 200.0
        Q0 = 0.0
        e0 = 1.224
        OCR = 1.0

        eps_out, p_out, q_out = engine.run_virtual_experiment(P0, Q0, e0, OCR, eps_seq, deps_seq)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(eps_out * 100, q_out, 'r-', linewidth=1.2, alpha=0.8)
        plt.xlabel('Axial Strain (%)', fontsize=12)
        plt.ylabel('Deviatoric Stress Q (kPa)', fontsize=12)
        plt.title(f'Virtual Cyclic Test (e0={e0}, P0={P0}kPa)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(p_out, q_out, 'b-', linewidth=1.2, alpha=0.8)
        plt.scatter([P0], [Q0], color='black', zorder=5, label='Initial State')
        plt.xlabel('Mean Effective Stress P (kPa)', fontsize=12)
        plt.ylabel('Deviatoric Stress Q (kPa)', fontsize=12)
        plt.title('Effective Stress Path (P-Q)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('Real_Model_Cyclic_Test.png', dpi=300)
        print("[INFO] 测试完毕，真正的神经网络本构曲线已生成至 Real_Model_Cyclic_Test.png！")
        plt.tight_layout()
        plt.savefig('Real_Model_Cyclic_Test.png', dpi=300)
        print("[INFO] 测试完毕，真正的神经网络本构曲线已生成至 Real_Model_Cyclic_Test.png！")

        # ==========================================================
        # 🌟 新增：自动导出数据到 Excel / CSV 表格
        # ==========================================================
        print("[INFO] 正在生成数据表格...")

        # 1. 把你的三个一维数组组合成一个字典
        data_dict = {
            'Step': np.arange(len(eps_out)),  # 步数记录
            'Axial_Strain_eps1': eps_out,  # 轴向应变
            'Mean_Stress_P_kPa': p_out,  # 平均有效应力 P
            'Deviatoric_Stress_Q_kPa': q_out  # 偏应力 Q
        }

        # 2. 转换成 Pandas 的数据框 (DataFrame)
        df_result = pd.DataFrame(data_dict)

        # 3. 保存为 CSV 文件 (Excel 可以直接双击完美打开)
        output_filename = 'Real_Model_Cyclic_Test_Data.csv'
        df_result.to_csv(output_filename, index=False, encoding='utf-8-sig')

        print(f"[INFO] 恭喜！实验数据已成功导出至当前文件夹下的: {output_filename}！")
