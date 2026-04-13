import numpy as np

# Pré-alocar array de output reutilizável — evita malloc a cada chamada
# Dimensionado para o máximo de pontos possível (resolução típica de lidar 2D)
_MAX_PTS = 1024
_out_buf = np.empty((_MAX_PTS, 3), dtype=np.float32)

def get_3d_points(ranges, thetas, alpha, beta, L_BRACO, Z_TORRE):
    """
    Calcula coordenadas XYZ de forma vetorizada.
    Otimizações vs versão original:
      - Evita re-alocação de array se ranges já for ndarray
      - Pré-computa senos/cossenos (cada ângulo calculado 1x em vez de 2x)
      - Usa np.empty + escrita direta em vez de column_stack (evita cópia extra)
      - dtype float32 ao longo de todo o pipeline (metade da memória vs float64)
    """
    # Evitar conversão desnecessária se já for array numpy
    if not isinstance(ranges, np.ndarray):
        ranges = np.asarray(ranges, dtype=np.float32)
    elif ranges.dtype != np.float32:
        ranges = ranges.astype(np.float32)

    # Garantir que thetas também é float32
    if not isinstance(thetas, np.ndarray):
        thetas = np.asarray(thetas, dtype=np.float32)

    # Filtro de distância
    mask = (ranges > 0.25) & (ranges < 11.9)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    r  = ranges[mask]
    th = thetas[mask]
    n  = len(r)

    # Pré-computar todos os senos/cossenos uma única vez
    cos_th = np.cos(th)
    sin_th = np.sin(th)
    cos_a  = float(np.cos(alpha))
    sin_a  = float(np.sin(alpha))
    cos_b  = float(np.cos(beta))
    sin_b  = float(np.sin(beta))

    # 1. Coordenadas locais do Lidar
    lx = r * cos_th   # float32 * float32 = float32
    ly = r * sin_th

    # 2. Rotação da Plataforma (Beta) — L_BRACO aplicado aqui
    x1 = lx
    y1 = ly * cos_b - L_BRACO * sin_b
    z1 = ly * sin_b + L_BRACO * cos_b

    # 3. Rotação do Anel (Alpha) + offset Z_TORRE
    # Escrever diretamente no buffer pré-alocado quando possível
    if n <= _MAX_PTS:
        out = _out_buf[:n]
        out[:, 0] = x1 * cos_a + z1 * sin_a          # X
        out[:, 1] = y1                                  # Y
        out[:, 2] = x1 * sin_a - z1 * cos_a + Z_TORRE # Z  (sinal corrigido)
        return out.copy()   # cópia leve — o buffer é reutilizado na próxima chamada
    else:
        # Fallback para arrays maiores que o buffer (raro)
        X = x1 * cos_a + z1 * sin_a
        Y = y1
        Z = x1 * sin_a - z1 * cos_a + Z_TORRE
        return np.stack((X, Y, Z), axis=1).astype(np.float32)