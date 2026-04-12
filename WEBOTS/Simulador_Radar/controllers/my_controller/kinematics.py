import numpy as np

def get_3d_points(ranges, thetas, alpha, beta, L_BRACO, Z_TORRE):
    """Calcula coordenadas XYZ de forma vetorizada (Sem loops for)"""
    ranges = np.array(ranges)
    # Filtro básico de distância
    mask = (ranges < 11.9) & (ranges > 0.25)
    
    if not np.any(mask):
        return np.zeros((0, 3))
    
    r = ranges[mask]
    th = thetas[mask]
    
    # 1. Coordenadas locais do Lidar
    lx = r * np.cos(th)
    ly = r * np.sin(th)
    
    # 2. Rotação da Plataforma (Beta)
    x1 = lx
    y1 = ly * np.cos(beta) - L_BRACO * np.sin(beta)
    z1 = ly * np.sin(beta) + L_BRACO * np.cos(beta)
    
    # 3. Rotação do Anel (Alpha)
    X = x1 * np.cos(alpha) + z1 * np.sin(alpha)
    Y = y1
    Z = -(-x1 * np.sin(alpha) + z1 * np.cos(alpha)) + Z_TORRE
    
    return np.column_stack((X, Y, Z))