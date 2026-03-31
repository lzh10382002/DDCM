# ================= 超参数配置区 =================
HYPERPARAMS = {
    # 架构超参数
    'seq_len': 5,  # 时序滑动窗口长度（Transformer 往前看几步历史）
    'd_model': 64,  # Transformer 的隐藏层维度
    'nhead': 4,  # 多头注意力的头数
    'num_encoder_layers': 2,  # Transformer Encoder 的层数
    'mlp_layers': [64, 128, 128, 4],  # ResNet-MLP 结构 (纯ANN：直接输出最终的 T1, T2, T3)

    # 训练超参数
    'learning_rate': 0.0005,
    'N_iter': 150000,  # 训练轮数
    'batch_size': 256,  # Mini-batch 大小
    'N_interv': 100,  # 打印间隔
    'patience': 5000,  # 学习率衰减耐心
    'min_lr': 1e-6,

    # 优化技术与路径
    'model_dir': './GCNN_ann_baseline_outputasinput_C_all_train_6_4_3_11_1',
    'train_model': True,
    'resume_train': False,

    'use_batch_norm': False,
    'use_dropout': False,
    'use_l2': False,

    'dropout_rate': 0.2,
    'l2_lambda': 1e-5,
}
# ==============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import math
from sklearn.metrics import r2_score
import matplotlib

matplotlib.use('Agg')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")


###############################################################################
################### 1. Transformer & ResNet-MLP Modules #######################
###############################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ResNetMLP(nn.Module):
    def __init__(self, in_dim, layers, dropout_rate, use_bn, use_dropout):
        super(ResNetMLP, self).__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        curr_dim = in_dim
        for out_dim in layers[:-1]:
            self.linears.append(nn.Linear(curr_dim, out_dim))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(out_dim))
            if use_dropout:
                self.dropouts.append(nn.Dropout(dropout_rate))
            curr_dim = out_dim

        self.out_layer = nn.Linear(curr_dim, layers[-1])

    def forward(self, x):
        H = x
        for i, linear in enumerate(self.linears):
            H_prev = H
            H = linear(H)
            if self.use_bn:
                H = self.bns[i](H)
            H = torch.nn.functional.silu(H)  # PyTorch 的 swish
            if self.use_dropout:
                H = self.dropouts[i](H)

            # Skip Connection (残差连接)
            if H_prev.shape[-1] == H.shape[-1]:
                H = H + H_prev

        # 纯 ANN：这里输出的直接就是最终的预测值 T_pred
        T_pred = self.out_layer(H)
        return T_pred


class PureANNTransformer(nn.Module):
    def __init__(self, input_dim=12, d_model=64, nhead=4, num_encoder_layers=2, mlp_layers=[64, 128, 2],
                 dropout_rate=0.2, use_bn=False, use_dropout=False):
        super(PureANNTransformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model * 2, dropout=dropout_rate,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.mlp = ResNetMLP(d_model, mlp_layers, dropout_rate, use_bn, use_dropout)

        # 依然保留多任务的不确定性 Loss 权重，保证对比公平
        self.log_var_p = nn.Parameter(torch.tensor(0.0))
        self.log_var_q = nn.Parameter(torch.tensor(0.0))

    def forward(self, X_seq_norm):
        """
        纯数据驱动模式：只需要归一化后的序列特征
        """
        x = self.input_projection(X_seq_norm)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)
        H_last = memory[:, -1, :]

        # ================= 终极物理约束 =================
        # 让 MLP 老老实实只负责输出微小的“物理变化量 (Delta p, Delta q)”
        delta_pred = self.mlp(H_last)
        return delta_pred


###############################################################################
######################## 2. Pure ANN Wrapper Class ############################
###############################################################################
class ANNTorch:
    def __init__(self, data=None, lb=None, ub=None, model_path=None):
        self.seq_len = HYPERPARAMS['seq_len']

        if model_path is not None:
            self.load_model(model_path)
        elif data is not None and lb is not None and ub is not None:
            input_idx = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11,12,13]
            self.lb_input = torch.tensor(lb[input_idx], dtype=torch.float32).to(device)
            self.ub_input = torch.tensor(ub[input_idx], dtype=torch.float32).to(device)

            # ================= 还原：直接使用绝对值边界 =================
            self.lb_output = torch.tensor(lb[4:6], dtype=torch.float32).to(device)
            self.ub_output = torch.tensor(ub[4:6], dtype=torch.float32).to(device)
            # ==========================================================

            self.model = PureANNTransformer(
                input_dim=12,
                d_model=HYPERPARAMS['d_model'],
                nhead=HYPERPARAMS['nhead'],
                num_encoder_layers=HYPERPARAMS['num_encoder_layers'],
                mlp_layers=HYPERPARAMS['mlp_layers'],
                dropout_rate=HYPERPARAMS['dropout_rate'],
                use_bn=HYPERPARAMS['use_batch_norm'],
                use_dropout=HYPERPARAMS['use_dropout']
            ).to(device)
        else:
            raise ValueError("Provide either model_path or data initialization parameters.")

        weight_decay = HYPERPARAMS['l2_lambda'] if HYPERPARAMS['use_l2'] else 0.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=HYPERPARAMS['learning_rate'], weight_decay=weight_decay)

    def normalize_input(self, X):
        delta = self.ub_input - self.lb_input
        delta = torch.where(delta == 0, torch.ones_like(delta) * 1e-8, delta)
        return 2.0 * (X - self.lb_input) / delta - 1.0

    def normalize_output(self, Y):
        delta = self.ub_output - self.lb_output
        delta = torch.where(delta == 0, torch.ones_like(delta) * 1e-8, delta)
        return 2.0 * (Y - self.lb_output) / delta - 1.0

    def compute_loss(self, T_pred, targets):
        # 1. 统一映射到 [-1, 1] 区间计算 MSE
        Y_pred_norm = self.normalize_output(T_pred)
        Y_true_norm = self.normalize_output(targets)

        loss_p_base = torch.mean(torch.abs(Y_pred_norm[:, 0] - Y_true_norm[:, 0]))
        loss_q_base = torch.mean(torch.abs(Y_pred_norm[:, 1] - Y_true_norm[:, 1]))

        # 2. 自适应不确定性 Loss 加权
        l1 = loss_p_base * torch.exp(torch.clamp(-self.model.log_var_p, -20.0, 20.0)) + self.model.log_var_p
        l2 = loss_q_base * torch.exp(torch.clamp(-self.model.log_var_q, -20.0, 20.0)) + self.model.log_var_q

        return l1 + l2

    def auto_detect_threshold(self, array_1d, is_drop=False, min_floor=1e-4):
        """
        自动计算物理序列的最佳跳变阈值 (最小真实跳变法)
        """
        if is_drop:
            # 1. 专门针对应变 eps1：计算后一步减前一步
            diffs = np.diff(array_1d)
            # 只保留下跌的值（负数），并取绝对值变成正数方便计算
            target_diffs = np.abs(diffs[diffs < 0])
        else:
            # 2. 针对静态 DNA (p0, e0等)：直接看绝对差值
            target_diffs = np.abs(np.diff(array_1d))

        # 过滤掉极其微小的底层浮点底噪
        valid_diffs = target_diffs[target_diffs > 1e-6]

        if len(valid_diffs) == 0:
            return min_floor

        # ================= 🌟 核心修复逻辑 =================
        # 任何大于安全底线 (min_floor) 的波动，我们都认定为是真实的物理跳变
        real_jumps = valid_diffs[valid_diffs > min_floor]

        if len(real_jumps) == 0:
            # 如果所有的波动都小于底线，说明这就是一个纯净的单一试验，直接返回底线兜底
            return min_floor

        # 最佳阈值：找到所有真实跳变里【最小的那个】，取它的一半！
        # 这样既越过了噪音墙，又绝对不会漏掉任何一次温柔的换土。
        best_threshold = np.min(real_jumps) * 0.5

        return max(min_floor, best_threshold)

    def create_sequences(self, data_array):
        """按滑动窗口将 2D 表格打包为 3D 时序序列，并自动剔除跨越不同试验的脏窗口"""

        # 🌟 1. 利用多维物理特征，在原始大表格上寻找试验交界点
        # 注意：这里直接操作 data_array，此时还没有提取 input_idx
        # 提取动态变量
        eps1_array = data_array[:, 2]
        # 提取 DNA 静态变量
        p0_array = data_array[:, 6]
        q0_array = data_array[:, 7]
        e0_array = data_array[:, 8]
        ocr_array = data_array[:, 9]

        # 1. 动态防线：应变发生时光倒流 (落差 > 0.04)
        thresh_eps = self.auto_detect_threshold(eps1_array, is_drop=True, min_floor=0.01)
        thresh_p0 = self.auto_detect_threshold(p0_array, min_floor=1e-3)
        thresh_q0 = self.auto_detect_threshold(q0_array, min_floor=1e-3)
        thresh_e0 = self.auto_detect_threshold(e0_array, min_floor=1e-3)
        thresh_ocr = self.auto_detect_threshold(ocr_array, min_floor=1e-3)

        # 将动态计算出的阈值应用到切分逻辑中 (彻底消灭硬编码！)
        cond_eps_drop = np.diff(eps1_array) < -thresh_eps
        cond_p0_change = np.abs(np.diff(p0_array)) > thresh_p0
        cond_q0_change = np.abs(np.diff(q0_array)) > thresh_q0
        cond_e0_change = np.abs(np.diff(e0_array)) > thresh_e0
        cond_ocr_change = np.abs(np.diff(ocr_array)) > thresh_ocr

        # 只要触发任何一条，一刀切断！
        jump_mask = cond_eps_drop | cond_p0_change | cond_q0_change | cond_e0_change | cond_ocr_change
        jump_indices = np.where(jump_mask)[0] + 1

        # 🌟 2. 一刀切断，生成多个纯净的独立试验段
        segments = np.split(data_array, jump_indices)

        X_seq_list, X_prev_list, Y_list = [], [], []
        input_idx = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11,12,13]  # 输入特征所在的列

        # 🌟 3. 在每个独立的纯净试验段内，安全地滑动窗口
        for seg in segments:
            num_samples = len(seg)
            if num_samples < self.seq_len + 1:
                continue  # 扔掉太短的垃圾碎片

            # ⚠️ 从索引 1 开始，保证我们总能拿到 i-1 (上一个窗口)
            for i in range(1, num_samples - self.seq_len + 1):
                window = seg[i: i + self.seq_len]
                window_prev = seg[i - 1: i - 1 + self.seq_len]

                X_seq_list.append(window[:, input_idx])
                X_prev_list.append(window_prev[:, input_idx])

                # 真实目标：最后一步的 p_new 和 q_new
                Y_list.append(window[-1, 4:6])
        print(
            f"[INFO] 自动切分完毕！动态阈值: eps1(跌落 > {thresh_eps:.4f}), p0(> {thresh_p0:.4f}), q0(> {thresh_q0:.4f}), e0(> {thresh_e0:.5f}), OCR(> {thresh_ocr:.4f})")
        # print(f"[INFO] 滑动窗口工厂：避开了 {len(jump_indices)} 处交界点，生成了纯净窗口。")
        return np.array(X_seq_list), np.array(X_prev_list), np.array(Y_list)

    def nn_train(self, Data, N_iter, N_interv, batch_size):
        self.model.train()
        X_seq, X_prev, Y = self.create_sequences(Data)

        X_seq_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
        X_prev_t = torch.tensor(X_prev, dtype=torch.float32).to(device)
        Y_t = torch.tensor(Y, dtype=torch.float32).to(device)

        dataset = TensorDataset(X_seq_t, X_prev_t, Y_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        history, accuracy_history, lr_history = [], [], []
        r2_h, mae_h, mse_h, rmse_h, mape_h = [], [], [], [], []

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                         patience=HYPERPARAMS['patience'], min_lr=HYPERPARAMS['min_lr'])

        start_time = time.time()
        print("[INFO] 开始训练 (双头架构：方向盘 + GPS 导航模式)...")

        epochs = max(1, N_iter // len(dataloader))
        global_step = 0

        for epoch in range(epochs):
            for batch_X_raw, batch_X_prev_raw, batch_Y in dataloader:
                self.optimizer.zero_grad()

                # 🌟 1. 计划采样：决定今天是“温室”还是“断奶”
                if global_step < N_iter / 3.0:
                    P = 1.0
                elif global_step < N_iter * (2.0 / 3.0):
                    P = 1.0 - (global_step - N_iter / 3.0) / (N_iter / 3.0)
                else:
                    P = 0.0

                import random
                is_route_A = random.random() < P
                true_p_old_base = batch_X_raw[:, -1, 0:2]

                # 声明变量防止报错
                delta_pred = None
                abs_pred = None
                drift_error = None

                if is_route_A:
                    # ================= 路线 A：温室阶段 (带噪音) =================
                    noisy_X_raw = batch_X_raw.clone()
                    noise_scale = min(1.0, global_step / (HYPERPARAMS['N_iter'] * 0.2))
                    current_noise_ratio = 0.015 * noise_scale
                    base_stress = torch.abs(noisy_X_raw[:, :, 0:2])
                    tiny_noise = torch.randn_like(noisy_X_raw[:, :, 0:2]) * (base_stress * current_noise_ratio + 1e-3)
                    noisy_X_raw[:, :, 0:2] += tiny_noise

                    batch_X_norm = self.normalize_input(noisy_X_raw)
                    out_4dim = self.model(batch_X_norm)

                    delta_pred = out_4dim[:, 0:2]
                    abs_pred = out_4dim[:, 2:4]
                    drift_error = noisy_X_raw[:, -1, 0:2] - true_p_old_base



                else:
                    # ================= 路线 B：实战断奶 (平滑的物理漂移拉练) =================
                    with torch.no_grad():
                        batch_X_prev_norm = self.normalize_input(batch_X_prev_raw)
                        out_prev = self.model(batch_X_prev_norm)
                        prev_delta = out_prev[:, 0:2]
                        prev_abs = out_prev[:, 2:4]
                        prev_p_old = batch_X_prev_raw[:, -1, 0:2]
                        prev_pred_final = 0.95 * (prev_p_old + prev_delta) + 0.05 * prev_abs
                    X_curr_fake = batch_X_raw.clone()
                    true_p_old_base = batch_X_raw[:, -1, 0:2]
                    # 🌟 终极优化：废弃暴力的 multiplier 放大法，改用物理级平滑漂移注入！
                    # 1. 计算真实的闭环单步误差
                    model_error = prev_pred_final - true_p_old_base
                    # 2. 生成一个随训练进度增加的、非常柔和的宏观漂移上限 (比如最大 8%)
                    drift_limit = torch.abs(true_p_old_base) * 0.08 * (global_step / HYPERPARAMS['N_iter'])
                    # 3. 把单步误差和我们的物理漂移上限结合起来，生成一个圆润的漂移量
                    # torch.sign 保证漂移方向和模型真实犯错的方向一致，但幅度是由我们控制的宏观平滑值
                    smooth_drift = torch.sign(model_error) * drift_limit
                    # 加上一丁点基础白噪音防止模型死记硬背
                    white_noise = torch.randn_like(true_p_old_base) * 0.5
                    # 最终的注入误差 = 平滑漂移 + 极弱白噪音 (不再有神经质的放大震荡！)
                    final_injected_error = smooth_drift + white_noise
                    # 5% 安全锁依然保留
                    max_allowed = torch.abs(true_p_old_base) * 0.05
                    final_injected_error = torch.clamp(final_injected_error, -max_allowed, max_allowed)
                    # 🌟 核心改进：把平滑的漂移量广播到整个窗口，绝对不破坏物理刚度斜率！
                    X_curr_fake[:, :, 0:2] += final_injected_error.unsqueeze(1)
                    batch_X_norm = self.normalize_input(X_curr_fake)
                    out_4dim = self.model(batch_X_norm)
                    delta_pred = out_4dim[:, 0:2]
                    abs_pred = out_4dim[:, 2:4]
                    drift_error = X_curr_fake[:, -1, 0:2] - true_p_old_base

                # ================= 🌟 终极双头 Loss 融合 🌟 =================
                # 1. 物理增量头目标：真实增量 - 纠偏拉力
                delta_true = batch_Y - true_p_old_base
                target_delta = delta_true - 0.05 * drift_error  # 每步抹掉 5% 漂移

                T_pred_from_delta = true_p_old_base + delta_pred
                T_target_from_delta = true_p_old_base + target_delta

                loss_delta = self.compute_loss(T_pred_from_delta, T_target_from_delta)

                # 2. 绝对坐标头目标：直接盯着最终真实值 (GPS)
                loss_abs = self.compute_loss(abs_pred, batch_Y)

                # 总 Loss：增量保形状，绝对保基线
                loss = loss_delta + 1.0 * loss_abs

                T_pred_abs_log = T_pred_from_delta

                # =============== 反向传播 ===============
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                if global_step % N_interv == 0:
                    elapsed = time.time() - start_time
                    loss_val = loss.item()
                    history.append(loss_val)
                    lr_history.append(self.optimizer.param_groups[0]['lr'])
                    with torch.no_grad():
                        pred_np = T_pred_abs_log.detach().cpu().numpy()
                        actual_np = batch_Y.cpu().numpy()
                        eps = 1e-10
                        rel_err = np.abs((actual_np - pred_np) / (np.abs(actual_np) + eps))
                        accuracy = np.mean(rel_err < 0.05) * 100
                        accuracy_history.append(accuracy)
                        r2 = r2_score(actual_np, pred_np)
                        mae = np.mean(np.abs(actual_np - pred_np))
                        mse = np.mean(np.square(actual_np - pred_np))
                        rmse = np.sqrt(mse)
                        mask = np.abs(actual_np) > 1e-9
                        mape = np.mean(
                            np.abs((actual_np[mask] - pred_np[mask]) / np.abs(actual_np[mask]))) * 100 if np.any(
                            mask) else 0.0
                        r2_h.append(r2);
                        mae_h.append(mae);
                        mse_h.append(mse);
                        rmse_h.append(rmse);
                        mape_h.append(mape)
                    print(
                        f'Step: {global_step}, Loss: {loss_val:.3e}, Acc: {accuracy:.2f}%, LR: {self.optimizer.param_groups[0]["lr"]:.6f}, R2: {r2:.4f}, Time: {elapsed:.2f}s')
                    start_time = time.time()

                global_step += 1
                if global_step >= N_iter: break
            scheduler.step(loss)
            if global_step >= N_iter: break
        return history, accuracy_history, lr_history, r2_h, mae_h, mse_h, rmse_h, mape_h

    def nn_predict(self, Te_data, batch_size=256, closed_loop=True):
        self.model.eval()
        X_seq_true, _, Y_true_abs = self.create_sequences(Te_data)

        if not closed_loop:
            print("[INFO] 正在进行单步预测 (Teacher Forcing)...")
            X_seq_t = torch.tensor(X_seq_true, dtype=torch.float32).to(device)
            dataset = TensorDataset(X_seq_t)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for batch_X_raw in dataloader:
                    batch_X_norm = self.normalize_input(batch_X_raw[0])

                    # ======================================================
                    # 🌟 修复点：接住 4 维输出，并提取前 2 维作为增量
                    out_4dim = self.model(batch_X_norm)
                    delta_pred = out_4dim[:, 0:2]  # 只取增量头参与单步预测计算
                    # ======================================================

                    p_old_physical = batch_X_raw[0][:, -1, 0:2].to(device)
                    T_pred_abs = p_old_physical + delta_pred
                    preds.append(T_pred_abs.cpu().numpy())
            return np.vstack(preds), Y_true_abs

        num_samples = len(X_seq_true)
        preds = []
        current_window = X_seq_true[0].copy()

        print("[INFO] 正在进行自回归(闭环)步进预测...")
        with torch.no_grad():
            for i in range(num_samples):
                batch_X_raw = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)
                batch_X_norm = self.normalize_input(batch_X_raw)
                out_4dim = self.model(batch_X_norm)

                delta_pred = out_4dim[:, 0:2]
                abs_pred = out_4dim[:, 2:4]

                p_old_physical = batch_X_raw[:, -1, 0:2]

                # 🌟 终极融合：网络自己给自己做残差修正！
                # P_增量推演 = P_old + Delta (极其敏锐，形状完美，但有累积误差)
                # P_GPS定位 = abs_pred (形状可能圆滑，但绝对位置永远不漂)
                T_pred_delta_path = p_old_physical + delta_pred

                gamma = 0.15  # 自我修正力度 (5% 听 GPS 的)
                T_pred_final = (1.0 - gamma) * T_pred_delta_path + gamma * abs_pred

                p_new_pred = T_pred_final[0, 0].item()
                q_new_pred = T_pred_final[0, 1].item()
                # =================================================================
                # 🌟 你的神级构想：闭环预测时的“无监督物理修正” 🌟
                # 我们没有真实的 P_old，但我们有大自然的物理法则！
                # =================================================================

                # 1. P 的护栏：土体不能受拉，绝对不允许出现负的 p。
                # 只要发现它要跌向深渊，强行把它托底在极其微小的正数 (比如 1.0 kPa)
                if p_new_pred < 1.0:
                    p_new_pred = 1.0

                # 2. Q 的护栏 (可选)：不能超过临界状态比 M
                # 假设你的土的 M 值大约是 1.2 (你可以根据你的实际土样调整这个 M 值)
                M_limit = 1.5
                max_q = M_limit * p_new_pred
                if q_new_pred > max_q:
                    q_new_pred = max_q
                elif q_new_pred < -max_q:
                    q_new_pred = -max_q

                preds.append([p_new_pred, q_new_pred])

                if i < num_samples - 1:
                    next_row = X_seq_true[i + 1, -1, :].copy()

                    # ================= 🌟 终极多维物理判断法 (满血 5 维版) =================
                    # X_seq_true 提取后的索引映射：
                    # 0: p_old, 1: q_old, 2: eps1, 3: deps1, 4: p0, 5: q0, 6: e0, 7: OCR

                    true_eps_curr = X_seq_true[i, -1, 2]
                    true_eps_next = X_seq_true[i + 1, -1, 2]

                    true_p0_curr = X_seq_true[i, -1, 4]
                    true_p0_next = X_seq_true[i + 1, -1, 4]

                    true_q0_curr = X_seq_true[i, -1, 5]
                    true_q0_next = X_seq_true[i + 1, -1, 5]

                    true_e0_curr = X_seq_true[i, -1, 6]
                    true_e0_next = X_seq_true[i + 1, -1, 6]

                    true_ocr_curr = X_seq_true[i, -1, 7]
                    true_ocr_next = X_seq_true[i + 1, -1, 7]

                    # 这里直接用 1e-3 兜底判断即可（因为前面 create_sequences 已经把非真实的断层全删了）
                    cond_eps_drop = (true_eps_curr - true_eps_next) > 0.04
                    cond_p0_change = abs(true_p0_next - true_p0_curr) > 1e-3
                    cond_q0_change = abs(true_q0_next - true_q0_curr) > 1e-3
                    cond_e0_change = abs(true_e0_next - true_e0_curr) > 1e-3
                    cond_ocr_change = abs(true_ocr_next - true_ocr_curr) > 1e-3

                    # 只要 5 个 DNA 指纹里有任何一个变了，立刻重置窗口！
                    if cond_eps_drop or cond_p0_change or cond_q0_change or cond_e0_change or cond_ocr_change:
                        print(f"[INFO] 抓到新试验起点！在第 {i} 步重置滑动窗口...")
                        current_window = X_seq_true[i + 1].copy()
                    else:
                        next_row[0] = p_new_pred
                        next_row[1] = q_new_pred
                        current_window = np.vstack((current_window[1:], next_row))

        return np.array(preds), Y_true_abs

    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir, 'model.pth'))

        metadata = {
            'lb_output': self.lb_output.cpu().numpy().tolist(),
            'ub_output': self.ub_output.cpu().numpy().tolist(),
            'lb_input': self.lb_input.cpu().numpy().tolist(),
            'ub_input': self.ub_input.cpu().numpy().tolist()
        }
        np.save(os.path.join(model_dir, "metadata.npy"), metadata)

    def load_model(self, model_dir):
        metadata = np.load(os.path.join(model_dir, "metadata.npy"), allow_pickle=True).item()
        self.lb_output = torch.tensor(metadata['lb_output']).to(device)
        self.ub_output = torch.tensor(metadata['ub_output']).to(device)
        self.lb_input = torch.tensor(metadata['lb_input']).to(device)
        self.ub_input = torch.tensor(metadata['ub_input']).to(device)

        self.model = PureANNTransformer(
            input_dim=12, d_model=HYPERPARAMS['d_model'], nhead=HYPERPARAMS['nhead'],
            num_encoder_layers=HYPERPARAMS['num_encoder_layers'], mlp_layers=HYPERPARAMS['mlp_layers'],
            dropout_rate=HYPERPARAMS['dropout_rate'], use_bn=HYPERPARAMS['use_batch_norm'],
            use_dropout=HYPERPARAMS['use_dropout']
        ).to(device)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=device))

    # ================= 错误评估与绘图辅助函数 =================
    def error_indicator(self, actu, pred, N_out):
        names = locals()
        model_order = 1
        Indicator = np.zeros((N_out + 1, model_order * 5))
        for mi in range(1, model_order + 1):
            names['R' + str(mi)] = 0;
            names['MAE' + str(mi)] = 0;
            names['MSE' + str(mi)] = 0
            names['RMSE' + str(mi)] = 0;
            names['MAPE' + str(mi)] = 0

            for oi in range(N_out):
                y_true = actu[:, oi];
                y_pred = pred[:, oi]
                r2 = r2_score(y_true, y_pred)
                mae = np.mean(np.abs(y_true - y_pred))
                mse = np.mean(np.square(y_true - y_pred))
                rmse = np.sqrt(mse)
                mask = np.abs(y_true) > 1e-9
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100 if np.any(
                    mask) else 0.0

                names['R' + str(mi)] += r2;
                names['MAE' + str(mi)] += mae;
                names['MSE' + str(mi)] += mse
                names['RMSE' + str(mi)] += rmse;
                names['MAPE' + str(mi)] += mape

                Indicator[oi, mi - 1] = r2;
                Indicator[oi, mi + model_order - 1] = mae;
                Indicator[oi, mi + 2 * model_order - 1] = mse
                Indicator[oi, mi + 3 * model_order - 1] = rmse;
                Indicator[oi, mi + 4 * model_order - 1] = mape

            Indicator[N_out, mi - 1] = names['R' + str(mi)] / N_out
            Indicator[N_out, mi + model_order - 1] = names['MAE' + str(mi)] / N_out
            Indicator[N_out, mi + 2 * model_order - 1] = names['MSE' + str(mi)] / N_out
            Indicator[N_out, mi + 3 * model_order - 1] = names['RMSE' + str(mi)] / N_out
            Indicator[N_out, mi + 4 * model_order - 1] = names['MAPE' + str(mi)] / N_out
        return Indicator

    def AP_scatter(self, actu, pred, N_out, save_path=None):
        plt.rcParams["figure.figsize"] = (15, 5)
        fig, ax = plt.subplots(1, N_out)
        if N_out == 1: ax = [ax]
        titles = ['p (kPa)', 'q (kPa)']
        for i in range(N_out):
            y_true = actu[:, i];
            y_pred = pred[:, i]
            r2 = r2_score(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean(np.square(y_true - y_pred))
            rmse = np.sqrt(mse)
            mask = np.abs(y_true) > 1e-9
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100 if np.any(mask) else 0.0

            ax[i].scatter(y_true, y_pred, marker='o', alpha=0.7)
            min_val, max_val = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
            margin = (max_val - min_val) * 0.1
            ax[i].set_xlim(min_val - margin, max_val + margin)
            ax[i].set_ylim(min_val - margin, max_val + margin)
            ax[i].plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'k--', alpha=0.75)

            metrics_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}\nMAPE = {mape:.2f}%'
            ax[i].text(0.05, 0.95, metrics_text, transform=ax[i].transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            ax[i].set_xlabel(f'Actual {titles[i]}')
            ax[i].set_ylabel(f'Predicted {titles[i]}')
            ax[i].set_title(f'Output: {titles[i]}')
            ax[i].grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.close()
        return fig

    def plot_comparison(self, actu, pred, N_out, title="Training Data", save_path=None):
        plt.rcParams["figure.figsize"] = (15, 6 * N_out)
        fig, axes = plt.subplots(N_out, 1)
        if N_out == 1: axes = [axes]

        for i in range(N_out):
            ax = axes[i]
            x = np.arange(len(actu[:, i]))
            ax.plot(x, actu[:, i], 'b-', label='Actual', linewidth=2)
            ax.plot(x, pred[:, i], 'r--', label='Predicted', linewidth=2)

            y_true = actu[:, i];
            y_pred = pred[:, i]
            r2 = r2_score(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean(np.square(y_true - y_pred))
            rmse = np.sqrt(mse)
            mask = np.abs(y_true) > 1e-9
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100 if np.any(mask) else 0.0

            metrics_text = f'R² = {r2:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}%'
            ax.text(0.5, 0.02, metrics_text, transform=ax.transAxes, fontsize=10, horizontalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

            ax.set_title(f'Comparison of Actual and Predicted Values for Output T{i + 1}')
            ax.set_xlabel('Sample Index');
            ax.set_ylabel('Value');
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.fill_between(x, actu[:, i], pred[:, i], color='gray', alpha=0.3, label='Error')

        plt.suptitle(f'{title} Prediction Comparison', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path: plt.savefig(save_path)
        plt.close()
        return fig

    def Loss_curve(self, history, accuracy=None, lr_history=None, save_path=None):
        plt.rcParams["figure.figsize"] = (10, 6)
        fig, ax1 = plt.subplots(1, 1)

        ax1.set_xlabel('Iterations');
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.plot(history, color='tab:blue', label='Loss');
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_yscale('log');
        ax1.grid(True, linestyle='--', alpha=0.3)

        if accuracy is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Accuracy (%)', color='tab:red');
            ax2.plot(accuracy, color='tab:red', label='Accuracy')
            ax2.tick_params(axis='y', labelcolor='tab:red');
            ax2.set_ylim(0, 100)

            if lr_history is not None:
                ax3 = ax1.twinx()
                ax3.spines["right"].set_position(("axes", 1.1))
                ax3.set_ylabel('Learning Rate', color='tab:green');
                ax3.plot(lr_history, color='tab:green', label='LR', linestyle='-.')
                ax3.tick_params(axis='y', labelcolor='tab:green');
                ax3.set_yscale('log')

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines3, labels3 = ax3.get_legend_handles_labels()
                ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')

        plt.title('Training Progress');
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.close()
        return fig


###############################################################################
################################ Main Entry ###################################
###############################################################################
if __name__ == "__main__":
    opt_name = "Pure_ANN_Transformer"
    model_dir = HYPERPARAMS.get('model_dir', f'./saved_model_{opt_name}')
    HYPERPARAMS['model_dir'] = model_dir
    print(f"[INFO] 模型保存目录: {model_dir}")


    def load_and_add_fatigue_clock(file_path, has_header=False):
        """
        has_header=True: 文件有表头（测试集），Pandas会自动跳过第一行的英文字母。
        has_header=False: 文件全是纯数字（训练集），Pandas会把第一行也当成数据读取。
        """
        if has_header:
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, header=None)

        data_matrix = df.values

        # 1. 提取 DNA 识别试验边界 (防止拼凑文件导致的 cumsum 污染)
        p0_array = data_matrix[:, 6]
        q0_array = data_matrix[:, 7]
        e0_array = data_matrix[:, 8]
        ocr_array = data_matrix[:, 9]

        # 只要 DNA 突变，说明换了新土样，生成切断点
        jump_mask = (np.abs(np.diff(p0_array)) > 1e-3) | \
                    (np.abs(np.diff(q0_array)) > 1e-3) | \
                    (np.abs(np.diff(e0_array)) > 1e-3) | \
                    (np.abs(np.diff(ocr_array)) > 1e-3)
        jump_indices = np.where(jump_mask)[0] + 1

        # 把大文件切成一个个纯净的独立试验
        segments = np.split(data_matrix, jump_indices)

        acc_energy_list = []
        phase_flag_list = []

        for seg in segments:
            p_old = seg[:, 0]
            deps1 = seg[:, 3]

            # =======================================================
            # A. 修复 Bug：独立的疲劳时钟
            # 保证每个试验的损伤都是从 0 开始算，绝不跨试验污染！
            incremental_work = p_old * np.abs(deps1)
            acc_energy = np.cumsum(incremental_work)
            acc_energy_list.append(acc_energy)

            # =======================================================
            # B. 落实你的神级思路：加卸载红绿灯 (Phase Flag)
            # np.sign 会把正数变成 1(加载)，负数变成 -1(卸载)，0 就是 0
            # 这相当于明确告诉 AI：“现在刚度突变了！给我狠狠抓峰值！”
            phase_flag = np.sign(deps1)
            phase_flag_list.append(phase_flag)

        # 重新把切开的试验拼回去
        total_acc_energy = np.concatenate(acc_energy_list)
        log_acc_energy = np.log1p(total_acc_energy)
        total_phase_flag = np.concatenate(phase_flag_list)

        # 组装特征：原数据 + 疲劳时钟(变为了第12列) + 加卸载标志(变为了第13列)
        fatigue_col = log_acc_energy.reshape(-1, 1)
        phase_col = total_phase_flag.reshape(-1, 1)

        new_data_matrix = np.hstack((data_matrix, fatigue_col, phase_col))

        return new_data_matrix
    # Data Loading (前6列为特征，7-9列为输出目标)
    Tr = load_and_add_fatigue_clock('Train/C_Test_1_8_16_22_39.csv', has_header=False)
    Tv = Tr.copy()

    # 测试集 (有抬头，所以 has_header=True)
    Te = load_and_add_fatigue_clock('test/C03_ANN.csv', has_header=True)
    Data = np.vstack((Tr, Tv))
    lb = np.min(Data, axis=0)
    ub = np.max(Data, axis=0)

    train_model = HYPERPARAMS['train_model']
    N_out = 2  # 预测输出维度 (T1_new, T2_new, T3_new)

    if train_model:
        model = ANNTorch(Data, lb=lb, ub=ub)
        history, acc_h, lr_h, r2_h, mae_h, mse_h, rmse_h, mape_h = model.nn_train(
            Data, HYPERPARAMS['N_iter'], HYPERPARAMS['N_interv'], HYPERPARAMS['batch_size'])

        model.save_model(model_dir)

        N = min(len(history), len(acc_h), len(lr_h), len(r2_h), len(mae_h), len(mse_h), len(rmse_h), len(mape_h))
        training_metrics = np.column_stack((
            history[:N], acc_h[:N], lr_h[:N], r2_h[:N], mae_h[:N], mse_h[:N], rmse_h[:N], mape_h[:N]
        ))
        np.savetxt(model_dir + '/training_metrics.csv', training_metrics,
                   fmt='%0.10f', delimiter=',', header='Loss,Accuracy,Learning_Rate,R2,MAE,MSE,RMSE,MAPE')

        model.Loss_curve(history, acc_h, lr_h, save_path=os.path.join(model_dir, 'train_loss_curve.png'))

        # 训练集画图
        pred_tr, tr_actu_aligned = model.nn_predict(Data, closed_loop=False)

        tr_error = model.error_indicator(tr_actu_aligned, pred_tr, N_out)
        model.AP_scatter(tr_actu_aligned, pred_tr, N_out, save_path=os.path.join(model_dir, 'train_scatter.png'))
        model.plot_comparison(tr_actu_aligned, pred_tr, N_out, title="Training Data",
                              save_path=os.path.join(model_dir, 'train_comparison.png'))
        np.savetxt(model_dir + '/out_training.csv', np.hstack((tr_actu_aligned, pred_tr)), fmt='%.10f', delimiter=',')
        np.savetxt(model_dir + '/out_training_error.csv', tr_error, fmt='%.10f', delimiter=',')
        # ================= 👇 新增：训练集闭环 (Train-Test) 评估 👇 =================
        print("[INFO] 生成训练集闭环 (Closed-loop) 预测报告...")

        # 强制开启 closed_loop=True，用实战模式跑训练集！
        pred_tr_closed, tr_actu_aligned_closed = model.nn_predict(Data, closed_loop=True)
        tr_closed_error = model.error_indicator(tr_actu_aligned_closed, pred_tr_closed, N_out)

        # 1. 保存散点图 (加上 _closed 后缀区分)
        model.AP_scatter(tr_actu_aligned_closed, pred_tr_closed, N_out,
                         save_path=os.path.join(model_dir, 'train_closed_scatter.png'))

        # 2. 保存对比图 (加上 _closed 后缀区分)
        model.plot_comparison(tr_actu_aligned_closed, pred_tr_closed, N_out,
                              title="Training Data (Closed-Loop)",
                              save_path=os.path.join(model_dir, 'train_closed_comparison.png'))

        # 3. 保存预测数据和误差数据的 CSV
        np.savetxt(os.path.join(model_dir, 'out_training_closed.csv'),
                   np.hstack((tr_actu_aligned_closed, pred_tr_closed)), fmt='%.10f', delimiter=',')
        np.savetxt(os.path.join(model_dir, 'out_training_closed_error.csv'),
                   tr_closed_error, fmt='%.10f', delimiter=',')
        # ================= 👆 新增结束 👆 =================
    else:
        print("[INFO] 加载模型进行推理...")
        model = ANNTorch(model_path=model_dir)

    print("[INFO] 开始进行测试集预测...")
    te_pred, te_actu_aligned = model.nn_predict(Te, closed_loop=True)
    te_error = model.error_indicator(te_actu_aligned, te_pred, N_out)

    model.plot_comparison(te_actu_aligned, te_pred, N_out, title="Test Data",
                          save_path=os.path.join(model_dir, 'test_comparison.png'))
    model.AP_scatter(te_actu_aligned, te_pred, N_out, save_path=os.path.join(model_dir, 'test_scatter.png'))

    np.savetxt(model_dir + '/out_testing.csv', np.hstack((te_actu_aligned, te_pred)), fmt='%.10f', delimiter=',')
    np.savetxt(model_dir + '/out_testing_error.csv', te_error, fmt='%.10f', delimiter=',')
    print("[INFO] 运行完成，所有评估图表与指标已保存！")
    # ================= 👇 新增：测试集前 2000 步专属测试与画图 👇 =================
    limit_steps = 2000
    # 如果测试集超过 2000 行，只取前 2000 行；否则取全部
    Te_sub = Te[:limit_steps] if len(Te) > limit_steps else Te

    print(f"\n[INFO] 开始进行测试集前 {len(Te_sub)} 步专项预测...")
    te_pred_sub, te_actu_aligned_sub = model.nn_predict(Te_sub, closed_loop=True)
    te_error_sub = model.error_indicator(te_actu_aligned_sub, te_pred_sub, N_out)

    # 画图并保存 (加上 _2000 后缀区分)
    model.plot_comparison(te_actu_aligned_sub, te_pred_sub, N_out,
                          title=f"Test Data (First {len(Te_sub)} Steps)",
                          save_path=os.path.join(model_dir, 'test_comparison_2000.png'))
    model.AP_scatter(te_actu_aligned_sub, te_pred_sub, N_out,
                     save_path=os.path.join(model_dir, 'test_scatter_2000.png'))

    # 保存 2000 步专属的数据 CSV
    np.savetxt(os.path.join(model_dir, 'out_testing_2000.csv'),
               np.hstack((te_actu_aligned_sub, te_pred_sub)), fmt='%.10f', delimiter=',')
    np.savetxt(os.path.join(model_dir, 'out_testing_error_2000.csv'),
               te_error_sub, fmt='%.10f', delimiter=',')
    print(f"[INFO] 前 {len(Te_sub)} 步专项测试完成，图表与数据已单独保存！")
    # ================= 👆 新增结束 👆 =================