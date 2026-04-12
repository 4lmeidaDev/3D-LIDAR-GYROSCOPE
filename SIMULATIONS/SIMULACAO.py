import numpy as np
import matplotlib.pyplot as plt

def simular_rastreio(fov_deg, sample_rate, scan_freq, gimbal_freq, r=6.0):
    duration = 0.225  # Simular 1 segundo
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 1. Ângulo interno do LIDAR (360º)
    theta = 2 * np.pi * scan_freq * t
    
    # 2. Movimento do Gimbal (Oscilação senoidal desfasada pi/2)
    amp = np.radians(fov_deg / 2)
    alpha = amp * np.sin(2 * np.pi * gimbal_freq * t)          # Pitch (X)
    beta = amp * np.sin(2 * np.pi * gimbal_freq * t + np.pi/2) # Roll (Y)
    
    # 3. Coordenadas locais (LIDAR plano no horizonte)
    x_l = r * np.cos(theta)
    y_l = r * np.sin(theta)
    z_l = np.zeros_like(t)
    
    # 4. Aplicação das Matrizes de Rotação (Mapeamento 3D Real)
    # Rotação em X (Pitch)
    x1 = x_l
    y1 = y_l * np.cos(alpha) - z_l * np.sin(alpha)
    z1 = y_l * np.sin(alpha) + z_l * np.cos(alpha)
    
    # Rotação em Y (Roll)
    x_f = x1 * np.cos(beta) + z1 * np.sin(beta)
    y_f = y1
    z_f = -x1 * np.sin(beta) + z1 * np.cos(beta)
    
    return x_f, y_f, z_f

# --- Configuração da Visualização ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Simulação: FOV 45º, LD19 (4500Hz), Scan 10Hz, Gimbal 2.5Hz
x, y, z = simular_rastreio(45, 4500, 10, 2.5)

# Desenhar os pontos
scatter = ax.scatter(x, y, z, c=z, cmap='magma', s=1, alpha=0.5)

# Estética do Gráfico
ax.set_title("Simulação LIDAR 3D - Padrão de Onda (Gimbal 2-Eixos)")
ax.set_xlabel("X (metros)")
ax.set_ylabel("Y (metros)")
ax.set_zlabel("Z (metros)")

# Limites para manter a esfera centrada
limit = 7
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

plt.colorbar(scatter, label='Altura (Z)')
print("Gráfico gerado com sucesso!")
plt.show()