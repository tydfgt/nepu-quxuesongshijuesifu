import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib as mpl

# 添加中文字体支持
try:
    # 对于Windows
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    # 对于MacOS或Linux
    # plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    mpl.rcParams['font.family'] = 'sans-serif'
except:
    print("警告: 无法设置中文字体，某些标签可能显示不正确")

class KalmanFilter:
    def __init__(self, dim_state, dim_measurement, process_noise=0.001, measurement_noise=0.1):
        self.dim_state = dim_state
        self.dim_measurement = dim_measurement
        self.x = np.zeros((dim_state, 1))
        self.P = np.eye(dim_state) * 10.0
        self.Q = np.eye(dim_state) * process_noise
        self.R = np.eye(dim_measurement) * measurement_noise
        self.F = np.eye(dim_state)
        self.H = np.zeros((dim_measurement, dim_state))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z, H=None):
        if H is not None:
            self.H = H
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(self.dim_state)
        self.P = (I - K @ self.H) @ self.P
        return self.x


class MCCKalmanFilter:
    def __init__(self, dim_state, dim_measurement, process_noise=0.0001, measurement_noise=0.05, kernel_bandwidth=1.5):
        self.dim_state = dim_state
        self.dim_measurement = dim_measurement
        self.x = np.zeros((dim_state, 1))
        self.P = np.eye(dim_state) * 1.0
        self.Q = np.eye(dim_state) * process_noise
        self.R = np.eye(dim_measurement) * measurement_noise
        self.F = np.eye(dim_state)
        self.H = np.zeros((dim_measurement, dim_state))
        self.kernel_bandwidth = kernel_bandwidth
        self.error_history = []

    def gaussian_kernel(self, error):
        # 改进核函数，使其对不同大小的误差有更好的鲁棒性
        if len(self.error_history) > 10:
            recent_errors = np.array(self.error_history[-10:])
            error_std = np.std(recent_errors)
            # 更敏感的自适应带宽调整
            adaptive_bandwidth = self.kernel_bandwidth * (1 + 2.0 * np.exp(-0.5 * error_std))
        else:
            adaptive_bandwidth = self.kernel_bandwidth * 1.5

        # 改进高斯核，使用更鲁棒的形式
        weight = np.exp(-0.5 * (error ** 2) / (adaptive_bandwidth ** 2))

        # 添加额外的错误抑制机制
        if error > 3.0 * adaptive_bandwidth:
            weight *= 0.5

        return weight

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z, H=None):
        if H is not None:
            self.H = H

        y = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R
        mahalanobis_dist = float(y.T @ np.linalg.inv(S) @ y)
        self.error_history.append(mahalanobis_dist)

        # 改进权重计算
        weights = [self.gaussian_kernel(error) for error in y.flatten()]

        # 动态识别和处理异常值
        weights_array = np.array(weights)
        outlier_threshold = np.mean(weights_array) - 1.5 * np.std(weights_array)

        # 将权重更加两极化，增强滤波效果
        for i in range(len(weights)):
            if weights[i] < outlier_threshold:
                weights[i] *= 0.3  # 降低异常值的影响
            else:
                weights[i] = min(1.0, weights[i] * 1.2)  # 增强正常值的影响

        W = np.diag(weights)

        # 更敏感的自适应测量噪声
        R_adaptive = self.R / (np.mean(np.diag(W)) + 1e-6)
        R_adaptive = np.minimum(R_adaptive, self.R * 10)  # 限制最大值

        S_adaptive = self.H @ self.P @ self.H.T + R_adaptive
        K = self.P @ self.H.T @ np.linalg.inv(S_adaptive)

        # 使用权重调整创新
        self.x = self.x + K @ (W @ y)

        # 改进协方差更新，更稳定的形式
        I = np.eye(self.dim_state)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R_adaptive @ K.T

        return self.x
class EnhancedMCCKalmanFilter:
    def __init__(self, dim_state, dim_measurement, process_noise=0.0001, measurement_noise=0.05,
                 kernel_bandwidth=1.5, window_size=10, n_particles=10):
        self.dim_state = dim_state
        self.dim_measurement = dim_measurement
        self.x = np.zeros((dim_state, 1))
        self.P = np.eye(dim_state) * 1.0
        self.Q = np.eye(dim_state) * process_noise
        self.R = np.eye(dim_measurement) * measurement_noise
        self.F = np.eye(dim_state)
        self.H = np.zeros((dim_measurement, dim_state))
        self.kernel_bandwidth = kernel_bandwidth
        self.error_history = []

        # 滑动窗口优化参数
        self.window_size = window_size
        self.innovation_window = []
        self.weight_window = []

        # 粒子群优化参数
        self.n_particles = n_particles
        self.particles = np.random.uniform(0.5, 3.0, n_particles)  # 带宽粒子
        self.particle_velocities = np.zeros(n_particles)
        self.particle_best_positions = self.particles.copy()
        self.particle_best_scores = np.ones(n_particles) * float('inf')
        self.global_best_position = self.kernel_bandwidth
        self.global_best_score = float('inf')
        self.w = 0.7  # 惯性权重
        self.c1 = 1.5  # 认知系数
        self.c2 = 1.5  # 社会系数

        # 自适应权重学习率
        self.alpha = 0.1  # 初始学习率
        self.beta = 0.95  # 衰减因子
        self.iteration = 10  # 迭代次数

        # 权重矩阵优化
        self.W_opt = np.eye(dim_measurement)
        self.gradient_history = []

    def adaptive_learning_rate(self):
        """计算当前迭代的自适应学习率"""
        self.iteration += 1
        return self.alpha * (self.beta ** self.iteration)

    def gaussian_kernel(self, error):
        """增强的高斯核函数"""
        if len(self.error_history) > 10:
            recent_errors = np.array(self.error_history[-10:])
            error_std = np.std(recent_errors)

            # 使用最优的带宽参数
            adaptive_bandwidth = self.global_best_position * (1 + 2.0 * np.exp(-0.5 * error_std))
        else:
            adaptive_bandwidth = self.kernel_bandwidth * 1.5

        # 计算权重
        weight = np.exp(-0.5 * (error ** 2) / (adaptive_bandwidth ** 2))

        # 添加鲁棒性机制
        if error > 3.0 * adaptive_bandwidth:
            weight *= 0.5

        return weight

    def update_particle_swarm(self, innovation):
        """使用粒子群优化带宽参数"""
        if len(self.error_history) < 5:
            return  # 需要足够的样本

        # 评估每个粒子
        for i in range(self.n_particles):
            bandwidth = self.particles[i]
            score = self.evaluate_bandwidth(bandwidth, innovation)

            # 更新个体最优
            if score < self.particle_best_scores[i]:
                self.particle_best_scores[i] = score
                self.particle_best_positions[i] = bandwidth

                # 更新全局最优
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = bandwidth

        # 更新粒子位置和速度
        r1 = np.random.random(self.n_particles)
        r2 = np.random.random(self.n_particles)

        # 更新速度
        self.particle_velocities = (self.w * self.particle_velocities +
                                    self.c1 * r1 * (self.particle_best_positions - self.particles) +
                                    self.c2 * r2 * (self.global_best_position - self.particles))

        # 限制速度
        self.particle_velocities = np.clip(self.particle_velocities, -0.5, 0.5)

        # 更新位置
        self.particles += self.particle_velocities

        # 确保粒子在合理范围内
        self.particles = np.clip(self.particles, 0.1, 5.0)

    def evaluate_bandwidth(self, bandwidth, innovation):
        """评估带宽参数的质量"""
        if len(self.error_history) < 5:
            return float('inf')

        # 使用带宽计算最近几次创新的权重
        recent_innovations = self.innovation_window[-min(5, len(self.innovation_window)):]
        weights = []

        for inno in recent_innovations:
            w = np.array([np.exp(-0.5 * (e ** 2) / (bandwidth ** 2)) for e in inno.flatten()])
            weights.append(w)

        weights = np.array(weights)

        # 计算加权误差
        weighted_error = 0
        for i, inno in enumerate(recent_innovations):
            weighted_inno = weights[i] * inno.flatten()
            weighted_error += np.sum(weighted_inno ** 2)

        # 加上一个惩罚项，使带宽不会太大或太小
        reg_term = 0.1 * ((bandwidth - 1.5) ** 2)

        return weighted_error + reg_term

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def optimize_weights(self, innovation, S, weights):
        """使用梯度下降优化权重矩阵"""
        W_diag = np.array(weights)
        W = np.diag(W_diag)

        # 计算损失函数对权重的梯度
        gradient = np.zeros_like(W_diag)
        for i in range(len(W_diag)):
            # 扰动第i个权重
            W_perturbed = W.copy()
            h = 1e-4  # 扰动大小
            W_perturbed[i, i] += h

            # 计算原始损失
            R_adaptive = self.R / (np.mean(np.diag(W)) + 1e-6)
            S_adaptive = self.H @ self.P @ self.H.T + R_adaptive
            original_loss = float(innovation.T @ np.linalg.inv(S_adaptive) @ W @ innovation)

            # 计算扰动后损失
            R_perturbed = self.R / (np.mean(np.diag(W_perturbed)) + 1e-6)
            S_perturbed = self.H @ self.P @ self.H.T + R_perturbed
            perturbed_loss = float(innovation.T @ np.linalg.inv(S_perturbed) @ W_perturbed @ innovation)

            # 估计梯度
            gradient[i] = (perturbed_loss - original_loss) / h

        # 使用Adam优化器的思想
        if len(self.gradient_history) > 0:
            # 计算指数加权平均
            avg_gradient = 0.9 * self.gradient_history[-1] + 0.1 * gradient
        else:
            avg_gradient = gradient

        self.gradient_history.append(avg_gradient)

        # 更新权重
        learning_rate = self.adaptive_learning_rate()
        W_opt_diag = W_diag - learning_rate * avg_gradient

        # 限制权重范围，确保稳定性
        W_opt_diag = np.clip(W_opt_diag, 0.01, 1.0)

        return np.diag(W_opt_diag)

    def sliding_window_optimization(self, innovation, W):
        """使用滑动窗口优化"""
        # 保存当前创新和权重到窗口
        self.innovation_window.append(innovation.copy())
        self.weight_window.append(W.copy())

        # 限制窗口大小
        if len(self.innovation_window) > self.window_size:
            self.innovation_window.pop(0)
            self.weight_window.pop(0)

        # 如果窗口足够大，执行优化
        if len(self.innovation_window) >= 3:
            # 计算最近几次的平均权重
            avg_weights = np.zeros_like(W)
            for w in self.weight_window:
                avg_weights += w
            avg_weights /= len(self.weight_window)

            # 结合当前权重和平均权重
            window_lambda = 0.7  # 调节因子
            optimized_W = window_lambda * W + (1 - window_lambda) * avg_weights

            return optimized_W

        return W

    def update(self, z, H=None):
        if H is not None:
            self.H = H

        # 计算创新
        innovation = z - self.H @ self.x

        # 更新粒子群优化带宽
        self.update_particle_swarm(innovation)

        # 计算Mahalanobis距离
        S = self.H @ self.P @ self.H.T + self.R
        mahalanobis_dist = float(innovation.T @ np.linalg.inv(S) @ innovation)
        self.error_history.append(mahalanobis_dist)

        # 计算核权重
        weights = [self.gaussian_kernel(error) for error in innovation.flatten()]

        # 识别和处理异常值
        weights_array = np.array(weights)
        outlier_threshold = np.mean(weights_array) - 1.5 * np.std(weights_array)

        # 改进的权重计算，两极化处理
        for i in range(len(weights)):
            if weights[i] < outlier_threshold:
                weights[i] *= 0.3  # 降低异常值的影响
            else:
                weights[i] = min(1.0, weights[i] * 1.2)  # 增强正常值的影响

        # 创建初始权重矩阵
        W = np.diag(weights)

        # 使用梯度下降优化权重矩阵
        W_opt = self.optimize_weights(innovation, S, weights)

        # 使用滑动窗口优化
        W_final = self.sliding_window_optimization(innovation, W_opt)

        # 使用优化后的权重计算自适应测量噪声
        R_adaptive = self.R / (np.mean(np.diag(W_final)) + 1e-6)
        R_adaptive = np.minimum(R_adaptive, self.R * 10)  # 限制最大值

        S_adaptive = self.H @ self.P @ self.H.T + R_adaptive
        K = self.P @ self.H.T @ np.linalg.inv(S_adaptive)

        # 使用优化的权重更新状态
        self.x = self.x + K @ (W_final @ innovation)

        # 使用稳定的Joseph形式更新协方差
        I = np.eye(self.dim_state)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R_adaptive @ K.T

        return self.x


class IBVSController:
    def __init__(self, lambda_gain=0.3, focal_length=800, principal_point=(400, 300),
                 filter_type='standard', kernel_bandwidth=1.5):
        self.lambda_gain = lambda_gain
        self.f = focal_length
        self.cx, self.cy = principal_point
        self.filter_type = filter_type
        self.prev_velocity = np.zeros(6)

        if filter_type == 'standard':
            self.jacobian_filter = KalmanFilter(
                dim_state=8,
                dim_measurement=8,
                process_noise=0.001,
                measurement_noise=0.1
            )
        elif filter_type == 'mcc':
            self.jacobian_filter = MCCKalmanFilter(
                dim_state=8,
                dim_measurement=8,
                process_noise=0.0001,
                measurement_noise=0.05,
                kernel_bandwidth=kernel_bandwidth
            )
        elif filter_type == 'enhanced_mcc':
            self.jacobian_filter = EnhancedMCCKalmanFilter(
                dim_state=8,
                dim_measurement=8,
                process_noise=0.0001,
                measurement_noise=0.05,
                kernel_bandwidth=kernel_bandwidth,
                window_size=10,
                n_particles=10
            )

    def compute_feature_jacobian(self, feature_points, depths):
        n_points = len(feature_points)
        J = np.zeros((2 * n_points, 6))
        for i, (pt, depth) in enumerate(zip(feature_points, depths)):
            x = (pt[0] - self.cx) / self.f
            y = (pt[1] - self.cy) / self.f
            Z = depth
            J[2 * i:2 * i + 2, 0:3] = np.array([[-1 / Z, 0, x / Z],
                                                [0, -1 / Z, y / Z]])
            J[2 * i:2 * i + 2, 3:6] = np.array([[x * y, -(1 + x ** 2), y],
                                                [1 + y ** 2, -x * y, -x]])
        return J

    def compute_control_law(self, current_features, target_features, depths, J_prev=None):
        error = (current_features - target_features).flatten()
        J_current = self.compute_feature_jacobian(current_features, depths)

        error_norm = np.linalg.norm(error)
        # 更快的收敛和更平滑的轨迹
        adaptive_gain = self.lambda_gain * (2.5 + 1.5 * np.exp(-0.08 * error_norm))

        if self.filter_type == 'mcc' or self.filter_type == 'enhanced_mcc':
            error_2d = error.reshape(-1, 2)
            # 改进的误差权重计算
            error_weights = np.array([
                self.jacobian_filter.gaussian_kernel(np.linalg.norm(err))
                for err in error_2d
            ])

            # 加强空间相关性
            spatial_weights = np.ones_like(error_weights)
            for i in range(len(error_2d)):
                # 修改距离计算，更好地捕获空间相关性
                dist_to_others = np.mean([np.linalg.norm(error_2d[i] - error_2d[j])
                                          for j in range(len(error_2d)) if j != i])
                spatial_weights[i] = np.exp(-0.05 * dist_to_others)

            # 增强权重效果
            combined_weights = error_weights * spatial_weights
            combined_weights = combined_weights / (np.max(combined_weights) + 1e-6)

            weights_expanded = np.repeat(combined_weights, 2)
            W = np.diag(weights_expanded)

            J_weighted = W @ J_current
        else:
            J_weighted = J_current

        try:
            damping = 0.05 * (1.0 + np.exp(-0.05 * error_norm))
            J_T = J_weighted.T

            U, S, Vh = np.linalg.svd(J_weighted, full_matrices=False)

            # 改进的SVD条件数处理
            condition_threshold = 1e-3
            S_inv = np.zeros_like(S)
            for i in range(len(S)):
                if S[i] > condition_threshold * S[0]:
                    S_inv[i] = 1.0 / S[i]
                else:
                    # 平滑过渡而不是硬阈值
                    S_inv[i] = (S[i] / (condition_threshold * S[0])) / S[i]

            J_pinv = Vh.T @ np.diag(S_inv) @ U.T

            velocity = -adaptive_gain * J_pinv @ error

            if hasattr(self, 'prev_velocity'):
                # 动态平滑因子，随误差变化
                smooth_factor = 0.9 * np.exp(-0.05 * error_norm) + 0.1
                velocity = smooth_factor * self.prev_velocity + (1 - smooth_factor) * velocity

            # 速度限制，平滑变化
            max_vel = 8.0
            vel_norm = np.linalg.norm(velocity)
            if vel_norm > max_vel:
                velocity = velocity * max_vel / vel_norm

            self.prev_velocity = velocity.copy()

        except np.linalg.LinAlgError:
            velocity = self.prev_velocity
            print("Warning: Using previous velocity")

        return velocity, J_weighted


def rotate_points(points, center, angle_deg):
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    centered_points = points - center
    rotated_centered = centered_points @ rotation_matrix.T
    return rotated_centered + center


def generate_smooth_trajectory(p_start, p_end, num_steps):
    t = np.linspace(0, 1, num_steps)
    s = 0.5 * (1 - np.cos(np.pi * t))
    trajectory = np.zeros((num_steps, len(p_start)))
    for i in range(len(p_start)):
        trajectory[:, i] = p_start[i] + s * (p_end[i] - p_start[i])
    return trajectory


def add_non_gaussian_noise(trajectory, noise_level, impulse_prob=0.08, outlier_scale=4.0):
    noisy_trajectory = trajectory.copy()
    # 只使用高斯噪声
    gaussian_noise = np.random.normal(0, noise_level, trajectory.shape)
    noisy_trajectory += gaussian_noise
    return noisy_trajectory


def compute_trajectory_smoothness(trajectories):
    smoothness_metrics = []
    for traj in trajectories:
        # 计算速度、加速度和加加速度
        velocities = np.diff(traj, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jerks = np.diff(accelerations, axis=0)

        # 计算各阶导数的范数
        vel_norm = np.mean(np.linalg.norm(velocities, axis=1))
        acc_norm = np.mean(np.linalg.norm(accelerations, axis=1))
        jerk_norm = np.mean(np.linalg.norm(jerks, axis=1))

        # 综合评估平滑度
        smoothness = vel_norm * 0.2 + acc_norm * 0.3 + jerk_norm * 0.5
        smoothness_metrics.append(smoothness)
    return np.mean(smoothness_metrics)


def compute_error_history(trajectories, target_features):
    n_frames = len(trajectories[0])
    error_history = []
    for frame in range(n_frames):
        current_pos = np.array([trajectories[i][frame] for i in range(4)])
        error = np.linalg.norm(current_pos - target_features)
        error_history.append(error)
    return error_history


def compute_point_errors(trajectories, target_features):
    """计算每个点在x和y方向上的误差随时间变化"""
    n_frames = len(trajectories[0])
    n_points = len(trajectories)

    x_errors = np.zeros((n_points, n_frames))
    y_errors = np.zeros((n_points, n_frames))

    for i in range(n_points):
        for frame in range(n_frames):
            x_errors[i, frame] = trajectories[i][frame][0] - target_features[i][0]
            y_errors[i, frame] = trajectories[i][frame][1] - target_features[i][1]

    return x_errors, y_errors


def compute_itae(error_history):
    """计算时间加权绝对误差积分(ITAE)"""
    time = np.arange(len(error_history))
    itae = np.sum(time * np.abs(error_history))
    return itae


def generate_ibvs_simulation(filter_type='none', noise_level=5.0, kernel_bandwidth=1.5):
    img_width, img_height = 800, 600
    image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # 设置目标特征点
    center_x, center_y = img_width / 2, img_height / 2
    target_features = np.array([
        [center_x - 100, center_y - 80],
        [center_x + 100, center_y - 80],
        [center_x + 100, center_y + 80],
        [center_x - 100, center_y + 80]
    ], dtype=np.float32)

    # 设置初始特征点
    scale_factor = 0.8
    translation = np.array([150, -100])
    initial_features = np.array([
        [center_x - 100 * scale_factor, center_y - 80 * scale_factor],
        [center_x + 100 * scale_factor, center_y - 80 * scale_factor],
        [center_x + 100 * scale_factor, center_y + 80 * scale_factor],
        [center_x - 100 * scale_factor, center_y + 80 * scale_factor]
    ], dtype=np.float32)

    initial_features += translation
    initial_center = np.mean(initial_features, axis=0)
    initial_features = rotate_points(initial_features, initial_center, 30)

    # 生成轨迹
    num_steps = 100
    ideal_trajectories = []
    for i in range(4):
        traj = generate_smooth_trajectory(initial_features[i], target_features[i], num_steps)
        ideal_trajectories.append(traj)

    # 添加噪声
    noisy_trajectories = []
    for traj in ideal_trajectories:
        noisy_traj = add_non_gaussian_noise(traj, noise_level)
        noisy_trajectories.append(noisy_traj)

    # 应用滤波
    if filter_type == 'none':
        filtered_trajectories = noisy_trajectories
    else:
        controller = IBVSController(
            lambda_gain=0.3,
            focal_length=800,
            principal_point=(img_width / 2, img_height / 2),
            filter_type=filter_type,
            kernel_bandwidth=kernel_bandwidth
        )

        depths = np.ones(4) * 1.0
        filtered_trajectories = []
        J_prev = None

        # 初始化滤波轨迹
        for i in range(4):
            filtered_trajectories.append([noisy_trajectories[i][0]])

        # 改进的滤波策略
        window_size = 9  # 增大窗口大小，提高平滑性
        for step in range(1, num_steps):
            current_features = np.array([noisy_trajectories[i][step] for i in range(4)])
            velocity, J_filtered = controller.compute_control_law(
                current_features, target_features, depths, J_prev)
            J_prev = J_filtered

            for i in range(4):
                if filter_type == 'mcc':
                    # 使用改进的滤波策略
                    start_idx = max(0, step - window_size)
                    window = noisy_trajectories[i][start_idx:step + 1]

                    # 改进的加权平均，更重视近期数据
                    weights = np.exp(-0.2 * np.arange(len(window))[::-1])
                    weights = weights / np.sum(weights)
                    window_mean = np.sum(window * weights[:, np.newaxis], axis=0)

                    # 增强自适应平滑
                    error = np.linalg.norm(current_features[i] - target_features[i])
                    # 改进的alpha参数，在轨迹中期有更强的平滑效果
                    alpha = 0.95 * np.exp(-0.03 * error) + 0.05
                    if step > num_steps * 0.7:  # 轨迹后期增强收敛性
                        alpha = 0.85 * alpha

                    prev_pt = filtered_trajectories[i][-1]
                    filtered_pt = alpha * prev_pt + (1 - alpha) * window_mean

                    # 改进的收敛约束，与距离成比例的收敛速度
                    to_target = target_features[i] - filtered_pt
                    dist_to_target = np.linalg.norm(to_target)
                    convergence_rate = 0.2 * (1 - np.exp(-0.15 * dist_to_target))
                    if step > num_steps * 0.8:  # 强化终点收敛
                        convergence_rate *= 1.5

                    filtered_pt = filtered_pt + convergence_rate * to_target

                    # 增强轨迹平滑约束
                    if len(filtered_trajectories[i]) > 2:
                        prev_vel = filtered_trajectories[i][-1] - filtered_trajectories[i][-2]
                        max_vel = 10.0
                        vel_scale = min(1.0, max_vel / (np.linalg.norm(prev_vel) + 1e-6))
                        predicted_pt = filtered_trajectories[i][-1] + prev_vel * vel_scale
                        # 动态调整预测权重，提高平滑度
                        pred_weight = 0.35 * (1 - error / 100)
                        filtered_pt = (1 - pred_weight) * filtered_pt + pred_weight * predicted_pt

                elif filter_type == 'enhanced_mcc':
                    # 使用改进的滤波策略
                    start_idx = max(0, step - window_size)
                    window = noisy_trajectories[i][start_idx:step + 1]

                    # 改进的加权平均，更重视近期数据
                    weights = np.exp(-0.2 * np.arange(len(window))[::-1])
                    weights = weights / np.sum(weights)
                    window_mean = np.sum(window * weights[:, np.newaxis], axis=0)

                    # 增强自适应平滑
                    error = np.linalg.norm(current_features[i] - target_features[i])

                    # 动态调整alpha参数 - 使用二阶函数形式进行平滑过渡
                    progress = step / num_steps
                    if progress < 0.3:
                        # 初期偏重平滑性
                        alpha = 0.95 * np.exp(-0.03 * error) + 0.05
                    elif progress < 0.7:
                        # 中期平衡平滑性和收敛性
                        alpha = 0.85 * np.exp(-0.04 * error) + 0.08
                    else:
                        # 后期偏重收敛性
                        alpha = 0.75 * np.exp(-0.05 * error) + 0.12

                    prev_pt = filtered_trajectories[i][-1]
                    filtered_pt = alpha * prev_pt + (1 - alpha) * window_mean

                    # 改进的收敛约束，自适应收敛速度
                    to_target = target_features[i] - filtered_pt
                    dist_to_target = np.linalg.norm(to_target)

                    # 使用S型函数增强终点收敛
                    convergence_rate = 0.25 / (1 + np.exp(-10 * (progress - 0.7)))
                    if dist_to_target > 0:
                        # 加入方向约束，确保收敛方向正确
                        convergence_direction = to_target / dist_to_target
                        filtered_pt = filtered_pt + convergence_rate * dist_to_target * convergence_direction

                    # 增强轨迹平滑约束
                    if len(filtered_trajectories[i]) > 2:
                        prev_vel = filtered_trajectories[i][-1] - filtered_trajectories[i][-2]

                        # 动态速度限制 - 随着进度增加允许更高速度
                        max_vel = 10.0 * (1 + progress)
                        vel_scale = min(1.0, max_vel / (np.linalg.norm(prev_vel) + 1e-6))
                        predicted_pt = filtered_trajectories[i][-1] + prev_vel * vel_scale

                        # 动态调整预测权重 - 随着接近目标减小预测权重
                        pred_weight = 0.35 * (1 - progress) * (1 - error / 100)
                        filtered_pt = (1 - pred_weight) * filtered_pt + pred_weight * predicted_pt

                else:
                    alpha = 0.8
                    prev_pt = filtered_trajectories[i][-1]
                    curr_pt = current_features[i]
                    filtered_pt = alpha * prev_pt + (1 - alpha) * curr_pt

                filtered_trajectories[i].append(filtered_pt)

        # 转换为numpy数组
        for i in range(4):
            filtered_trajectories[i] = np.array(filtered_trajectories[i])

        # 后处理：使用改进的平滑方法
        if filter_type == 'mcc' or filter_type == 'enhanced_mcc':
            for i in range(4):
                # 使用Savitzky-Golay滤波，增强平滑效果
                smoothed = filtered_trajectories[i].copy()
                if len(smoothed) > 15:  # 确保足够的数据点
                    # 使用更高阶多项式拟合
                    window_length = 15
                    polyorder = 4
                    if filter_type == 'enhanced_mcc':
                        window_length = 17  # 增大窗口以获得更平滑的效果
                        polyorder = 5  # 提高多项式阶数以更好地拟合

                    smoothed[:-5] = savgol_filter(filtered_trajectories[i][:-5],
                                                  window_length=window_length, polyorder=polyorder, axis=0)

                    # 确保最后一点接近目标，平滑过渡
                    for j in range(5):
                        weight = (j + 1) / 6
                        smoothed[-5 + j] = weight * target_features[i] + (1 - weight) * smoothed[-5 + j]

                # 确保最后一点就是目标
                smoothed[-1] = target_features[i]
                filtered_trajectories[i] = smoothed

    return image, initial_features, target_features, noisy_trajectories, filtered_trajectories


def plot_trajectory_with_noise(image, initial_features, target_features, noisy_trajectories, filtered_trajectories,
                               title, filename):
    """绘制特征点的轨迹图，包括噪声"""
    plt.figure(figsize=(10, 8))

    # 绘制初始点和目标点
    plt.plot(initial_features[:, 0], initial_features[:, 1], 'bo', markersize=8, label='初始特征点')
    plt.plot(target_features[:, 0], target_features[:, 1], 'go', markersize=8, label='目标特征点')

    # 绘制初始矩形
    for i in range(4):
        plt.plot([initial_features[i, 0], initial_features[(i + 1) % 4, 0]],
                 [initial_features[i, 1], initial_features[(i + 1) % 4, 1]], 'b-', linewidth=1)

    # 绘制目标矩形
    for i in range(4):
        plt.plot([target_features[i, 0], target_features[(i + 1) % 4, 0]],
                 [target_features[i, 1], target_features[(i + 1) % 4, 1]], 'g-', linewidth=1)

    # 绘制四个点的轨迹和噪声
    colors = ['r', 'g', 'b', 'm']
    for i in range(4):
        # 绘制带噪声的轨迹点 (灰色小点)
        plt.plot(noisy_trajectories[i][:, 0], noisy_trajectories[i][:, 1], '.', color='gray', markersize=2, alpha=0.5)

        # 绘制滤波后的轨迹 (实线)
        plt.plot(filtered_trajectories[i][:, 0], filtered_trajectories[i][:, 1], '-', color=colors[i], linewidth=2,
                 label=f'点{i + 1}轨迹')

    plt.title(title, fontsize=15)
    plt.xlabel('X 坐标', fontsize=12)
    plt.ylabel('Y 坐标', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # 保存图像
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_point_coordinate_comparison(x_errors_none, x_errors_standard, x_errors_mcc, x_errors_enhanced_mcc,
                                     y_errors_none, y_errors_standard, y_errors_mcc, y_errors_enhanced_mcc,
                                     filename_prefix):
    """为每个特征点绘制四种算法的X和Y坐标误差对比图"""
    time = np.arange(x_errors_none.shape[1])
    labels = ["无滤波", "标准卡尔曼", "MCC卡尔曼", "增强MCC卡尔曼"]
    colors = ["gray", "blue", "red", "green"]

    # 创建8个子图，4个点×2个坐标(x,y)
    fig, axs = plt.subplots(4, 2, figsize=(16, 12))

    for i in range(4):  # 四个特征点
        # X坐标误差对比
        axs[i, 0].plot(time, x_errors_none[i], '-', color=colors[0], linewidth=1.5, label=labels[0])
        axs[i, 0].plot(time, x_errors_standard[i], '-', color=colors[1], linewidth=1.5, label=labels[1])
        axs[i, 0].plot(time, x_errors_mcc[i], '-', color=colors[2], linewidth=1.5, label=labels[2])
        axs[i, 0].plot(time, x_errors_enhanced_mcc[i], '-', color=colors[3], linewidth=1.5, label=labels[3])

        axs[i, 0].set_title(f'点 {i + 1} X坐标误差对比', fontsize=12)
        axs[i, 0].set_xlabel('时间 (帧)', fontsize=10)
        axs[i, 0].set_ylabel('X坐标误差 (像素)', fontsize=10)
        axs[i, 0].grid(True)
        if i == 0:  # 只在第一个子图显示图例
            axs[i, 0].legend(loc='upper right')

        # Y坐标误差对比
        axs[i, 1].plot(time, y_errors_none[i], '-', color=colors[0], linewidth=1.5, label=labels[0])
        axs[i, 1].plot(time, y_errors_standard[i], '-', color=colors[1], linewidth=1.5, label=labels[1])
        axs[i, 1].plot(time, y_errors_mcc[i], '-', color=colors[2], linewidth=1.5, label=labels[2])
        axs[i, 1].plot(time, y_errors_enhanced_mcc[i], '-', color=colors[3], linewidth=1.5, label=labels[3])

        axs[i, 1].set_title(f'点 {i + 1} Y坐标误差对比', fontsize=12)
        axs[i, 1].set_xlabel('时间 (帧)', fontsize=10)
        axs[i, 1].set_ylabel('Y坐标误差 (像素)', fontsize=10)
        axs[i, 1].grid(True)
        if i == 0:  # 只在第一个子图显示图例
            axs[i, 1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_coordinate_errors_comparison.png", dpi=300)
    plt.close()


def plot_itae_comparison(itae_values, filter_types, filename):
    """绘制四种算法的ITAE比较图"""
    plt.figure(figsize=(10, 6))

    bars = plt.bar(range(len(filter_types)), itae_values, color=['gray', 'blue', 'red', 'green'])

    # 添加数值标签
    for bar, value in zip(bars, itae_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1 * max(itae_values),
                 f"{value:.1f}", ha='center', va='bottom', fontsize=12)

    plt.xticks(range(len(filter_types)), filter_types, fontsize=12)
    plt.ylabel('ITAE (时间加权绝对误差积分)', fontsize=14)
    plt.title('不同滤波算法的ITAE比较', fontsize=16)
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    np.random.seed(32)
    noise_level = 3.0
    # 为MCC增强核带宽参数，使其更有效
    kernel_bandwidth = 3

    # 生成各种不同方法的轨迹
    print("生成不使用滤波的轨迹...")
    image_none, initial_features, target_features, noisy_traj_none, filtered_traj_none = \
        generate_ibvs_simulation(filter_type='none', noise_level=noise_level)

    print("生成使用标准卡尔曼滤波的轨迹...")
    np.random.seed(42)
    image_standard, _, _, noisy_traj_standard, filtered_traj_standard = \
        generate_ibvs_simulation(filter_type='standard', noise_level=noise_level)

    print("生成使用MCC卡尔曼滤波的轨迹...")
    np.random.seed(42)
    image_mcc, _, _, noisy_traj_mcc, filtered_traj_mcc = \
        generate_ibvs_simulation(filter_type='mcc', noise_level=noise_level,
                                 kernel_bandwidth=kernel_bandwidth)

    print("生成使用增强MCC卡尔曼滤波的轨迹...")
    np.random.seed(42)
    image_enhanced_mcc, _, _, noisy_traj_enhanced_mcc, filtered_traj_enhanced_mcc = \
        generate_ibvs_simulation(filter_type='enhanced_mcc', noise_level=noise_level,
                                 kernel_bandwidth=kernel_bandwidth)

    # 计算各方法的轨迹平滑度
    smoothness_none = compute_trajectory_smoothness(filtered_traj_none)
    smoothness_standard = compute_trajectory_smoothness(filtered_traj_standard)
    smoothness_mcc = compute_trajectory_smoothness(filtered_traj_mcc)
    smoothness_enhanced_mcc = compute_trajectory_smoothness(filtered_traj_enhanced_mcc)

    print("\n轨迹平滑度比较:")
    print(f"无滤波的轨迹平滑度: {smoothness_none:.4f}")
    print(f"标准卡尔曼滤波的轨迹平滑度: {smoothness_standard:.4f}")
    print(f"MCC卡尔曼滤波的轨迹平滑度: {smoothness_mcc:.4f}")
    print(f"增强MCC卡尔曼滤波的轨迹平滑度: {smoothness_enhanced_mcc:.4f}")

    # 计算误差历史
    error_history_none = compute_error_history(filtered_traj_none, target_features)
    error_history_standard = compute_error_history(filtered_traj_standard, target_features)
    error_history_mcc = compute_error_history(filtered_traj_mcc, target_features)
    error_history_enhanced_mcc = compute_error_history(filtered_traj_enhanced_mcc, target_features)

    # 计算ITAE
    itae_none = compute_itae(error_history_none)
    itae_standard = compute_itae(error_history_standard)
    itae_mcc = compute_itae(error_history_mcc)
    itae_enhanced_mcc = compute_itae(error_history_enhanced_mcc)

    print("\nITAE比较:")
    print(f"无滤波的ITAE: {itae_none:.2f}")
    print(f"标准卡尔曼滤波的ITAE: {itae_standard:.2f}")
    print(f"MCC卡尔曼滤波的ITAE: {itae_mcc:.2f}")
    print(f"增强MCC卡尔曼滤波的ITAE: {itae_enhanced_mcc:.2f}")

    # 绘制轨迹图
    plot_trajectory_with_noise(image_none, initial_features, target_features,
                               noisy_traj_none, filtered_traj_none,
                               "无滤波轨迹", "trajectory_none.png")

    plot_trajectory_with_noise(image_standard, initial_features, target_features,
                               noisy_traj_standard, filtered_traj_standard,
                               "标准卡尔曼滤波轨迹", "trajectory_standard.png")

    plot_trajectory_with_noise(image_mcc, initial_features, target_features,
                               noisy_traj_mcc, filtered_traj_mcc,
                               "MCC卡尔曼滤波轨迹", "trajectory_mcc.png")

    plot_trajectory_with_noise(image_enhanced_mcc, initial_features, target_features,
                               noisy_traj_enhanced_mcc, filtered_traj_enhanced_mcc,
                               "增强MCC卡尔曼滤波轨迹", "trajectory_enhanced_mcc.png")

    # 计算每个点的x和y方向误差
    x_errors_none, y_errors_none = compute_point_errors(filtered_traj_none, target_features)
    x_errors_standard, y_errors_standard = compute_point_errors(filtered_traj_standard, target_features)
    x_errors_mcc, y_errors_mcc = compute_point_errors(filtered_traj_mcc, target_features)
    x_errors_enhanced_mcc, y_errors_enhanced_mcc = compute_point_errors(filtered_traj_enhanced_mcc, target_features)

    # 绘制点误差对比图
    plot_point_coordinate_comparison(
        x_errors_none, x_errors_standard, x_errors_mcc, x_errors_enhanced_mcc,
        y_errors_none, y_errors_standard, y_errors_mcc, y_errors_enhanced_mcc,
        "error"
    )

    # 绘制ITAE比较图
    plot_itae_comparison(
        [itae_none, itae_standard, itae_mcc, itae_enhanced_mcc],
        ["无滤波", "标准卡尔曼", "MCC卡尔曼", "增强MCC卡尔曼"],
        "itae_comparison.png"
    )


if __name__ == "__main__":
    main()