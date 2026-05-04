import numpy as np


def compute_L(sequence, K):
    """根据时间序列长度和窗口大小 K 动态计算列数 L"""
    return len(sequence) - K + 1


def build_hankel_matrix(sequence, K, L):
    """
    构建汉克尔矩阵
    :param sequence: 输入时间序列 (1D array)
    :param K: 窗口大小（矩阵行数）
    :param L: 矩阵列数
    :return: 汉克尔矩阵 (K, L)
    """
    if len(sequence) < K + L - 1:
        raise ValueError("序列长度不足以构建指定维度的汉克尔矩阵")
    hankel = np.zeros((K, L))
    for i in range(L):
        hankel[:, i] = sequence[i:i+K]
    return hankel


def dmd_decomposition(X, Y):
    """DMD分解"""
    # 确保X和Y列数一致
    assert X.shape[1] == Y.shape[1], "X和Y必须具有相同的列数"

    u, s, v = np.linalg.svd(X, full_matrices=False)
    r = np.sum(s > 1e-6)  # 初步计算有效秩
    r = min(r, X.shape[1])  # 关键修正：确保r不超过X的列数

    # 计算A_tilde
    A_tilde = u[:, :r].conj().T @ Y @ v[:r, :].conj().T @ np.diag(1.0 / s[:r])

    # 计算特征值和模态
    eigenvalues, modes = np.linalg.eig(A_tilde)
    Phi = u[:, :r] @ modes  # 修正模态计算公式

    # 按特征值幅值排序（可选）
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    Phi = Phi[:, idx]

    return Phi, eigenvalues


def reconstruct_error(Phi, eigenvalues, initial_error, K, original_length):
    """DMD重构误差（修复维度广播问题）"""
    # 确保输入维度正确
    initial_error = np.asarray(initial_error).reshape(-1, 1)  # (K, 1)
    b = np.linalg.pinv(Phi) @ initial_error  # (r, 1)
    r = len(eigenvalues)  # 模态数

    # 动态计算最大列数 L
    max_L = original_length - K + 1
    L = max_L if max_L > 0 else 1

    # 重构汉克尔矩阵
    hankel_reconstructed = np.zeros((K, L), dtype=complex)
    for i in range(L):
        # 关键修正：确保维度对齐
        eig_power = (eigenvalues ** i).reshape(-1, 1)  # (r, 1)
        modal_coeff = b * eig_power  # (r, 1)
        state_vector = Phi @ modal_coeff  # (K, 1)
        hankel_reconstructed[:, i] = state_vector.squeeze()

    # 反汉克尔变换
    full_sequence = np.zeros(original_length)
    count = np.zeros(original_length)
    for col in range(L):
        start = col
        end = start + K
        full_sequence[start:end] += np.real(hankel_reconstructed[:, col])
        count[start:end] += 1
    reconstructed_error = np.divide(full_sequence, count, where=count != 0)
    return reconstructed_error[:original_length]


def correct_predictions(original_predictions, reconstructed_error):
    """修正预测（带维度校验）"""
    assert len(original_predictions) == len(reconstructed_error), \
        f"维度不匹配：预测{len(original_predictions)} vs 误差{len(reconstructed_error)}"
    return original_predictions + reconstructed_error
