import numpy as np


def compute_L(sequence, K):
    return len(sequence) - K + 1


def build_hankel_matrix(sequence, K, L):
    if len(sequence) < K + L - 1:
        raise ValueError("序列长度不足以构建指定维度的汉克尔矩阵")
    hankel = np.zeros((K, L))
    for i in range(L):
        hankel[:, i] = sequence[i:i+K]
    return hankel


def dmd_decomposition(X, Y):
    assert X.shape[1] == Y.shape[1], "X和Y必须具有相同的列数"

    u, s, v = np.linalg.svd(X, full_matrices=False)
    r = np.sum(s > 1e-6)
    r = min(r, X.shape[1])

    A_tilde = u[:, :r].conj().T @ Y @ v[:r, :].conj().T @ np.diag(1.0 / s[:r])

    eigenvalues, modes = np.linalg.eig(A_tilde)
    Phi = u[:, :r] @ modes

    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    Phi = Phi[:, idx]

    return Phi, eigenvalues


def reconstruct_error(Phi, eigenvalues, initial_error, K, original_length):
    initial_error = np.asarray(initial_error).reshape(-1, 1)
    b = np.linalg.pinv(Phi) @ initial_error
    r = len(eigenvalues)

    max_L = original_length - K + 1
    L = max_L if max_L > 0 else 1

    hankel_reconstructed = np.zeros((K, L), dtype=complex)
    for i in range(L):
        eig_power = (eigenvalues ** i).reshape(-1, 1)
        modal_coeff = b * eig_power
        state_vector = Phi @ modal_coeff
        hankel_reconstructed[:, i] = state_vector.squeeze()

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
    assert len(original_predictions) == len(reconstructed_error), \
        f"维度不匹配：预测{len(original_predictions)} vs 误差{len(reconstructed_error)}"
    return original_predictions + reconstructed_error
