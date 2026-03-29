def normalize_activation(A_k):
    A_min, A_max = A_k.min(), A_k.max()
    return (A_k - A_min) / (A_max - A_min + 1e-8)
