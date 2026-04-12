import numpy as np
import os
import sys
import math
import vispy
from vispy import app, scene
from kinematics import get_3d_points

# 1. FORÇAR BACKEND PYQT6 PARA COEXISTIR COM TKINTER
vispy.use('PyQt6')

# --- CONFIGURAÇÕES DE ENGENHARIA ---
# Devem ser idênticas ao scan.py para o filtro de ruído encaixar
Z_TORRE, L_BRACO = 0.130, 0.032
RAIO_MIN = 0.05  # Resolução de 5cm (para objetos de 10cm)

# Recuperar parâmetros da GUI (my_controller.py)
FREQ = float(os.getenv("PARAM_FREQ", 0.5))
FOV_G = float(os.getenv("PARAM_FOV", 1.047))
TEMPO_TOTAL = float(os.getenv("PARAM_TEMPO", 40.0))

# Decaimento: O ponto "morre" após 1 ciclo de oscilação (para não deixar rasto eterno)
TEMPO_VIDA = (1.0 / FREQ) * 1.1 

# --- 2. CARREGAMENTO DO BACKGROUND (FILTRO DE RUÍDO) ---
voxels_bg = set()
path_npy = "scan_otimizado.npy"

if os.path.exists(path_npy):
    print(f"A carregar mapa de referência: {path_npy}")
    data = np.load(path_npy)
    # Converter coordenadas reais para índices de grelha (Voxels)
    v_idx = np.floor(data / RAIO_MIN).astype(int)
    
    # PADDING: Bloqueamos o voxel e os 26 vizinhos à volta (Cubo 3x3x3)
    # Isto cria uma "armadura" de 5cm contra ruído do Lidar nas paredes.
    for v in v_idx:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    voxels_bg.add((v[0]+dx, v[1]+dy, v[2]+dz))
    print(f"Filtro de ruído blindado: {len(voxels_bg)} células protegidas.")
else:
    print("AVISO: scan_otimizado.npy não encontrado! O Search vai mostrar a sala toda.")

# --- 3. CONFIGURAÇÃO VISPY (GRÁFICO GPU) ---
canvas = scene.SceneCanvas(keys='interactive', show=True, title='RADAR SEARCH - DETEÇÃO DE INTRUSO')
view = canvas.central_widget.add_view()
view.camera = 'turntable'
scatter = scene.visuals.Markers()
view.add(scatter)
scene.visuals.XYZAxis(parent=view.scene)

# --- 4. DISPOSITIVOS E SENSORES ---
# robot e timestep são herdados do my_controller.py
lidar = robot.getDevice("lidar"); lidar.enable(timestep)
res_horiz = lidar.getHorizontalResolution()
fov_lidar = lidar.getFov()
thetas = np.linspace(-fov_lidar/2, fov_lidar/2, res_horiz)

motor_a = robot.getDevice("ANEL_INTERIOR_JOINT")
motor_b = robot.getDevice("PLATAFORMA_JOINT")
s_a = robot.getDevice("ANEL_INTERIOR_JOINT_sensor"); s_a.enable(timestep)
s_b = robot.getDevice("PLATAFORMA_JOINT_sensor"); s_b.enable(timestep)

# Memória dinâmica: { voxel_tup: [ [X,Y,Z], timestamp ] }
mapa_dinamico = {}
last_plot_time = 0
tempo_inicio_search = robot.getTime()

print("Search em tempo real ativo. Fecha a janela do gráfico para parar.")

# --- 5. LOOP DE OPERAÇÃO ---
while robot.step(timestep) != -1:
    t_atual = robot.getTime()
    t_relativo = t_atual - tempo_inicio_search
    
    if t_relativo > TEMPO_TOTAL or canvas._closed:
        break

    # MOVIMENTO SINUSOIDAL (ESTILO RADAR)
    pos_a = FOV_G * math.sin(2 * math.pi * FREQ * t_relativo)
    pos_b = FOV_G * math.sin(2 * math.pi * FREQ * t_relativo + (math.pi/2))
    motor_a.setPosition(pos_a)
    motor_b.setPosition(pos_b)

    # CAPTURA E CINEMÁTICA VETORIZADA
    ranges = lidar.getRangeImage()
    # Processamos 1 em cada 5 feixes (Equilíbrio entre detalhe de 10cm e performance)
    pts = get_3d_points(ranges[::5], thetas[::5], s_a.getValue(), s_b.getValue(), L_BRACO, Z_TORRE)

    if pts.size > 0:
        # Converter novos pontos para índices de voxel
        v_coords = np.floor(pts / RAIO_MIN).astype(int)
        
        for i in range(len(v_coords)):
            v_tup = (v_coords[i,0], v_coords[i,1], v_coords[i,2])
            
            # FILTRAGEM: Só aceita se NÃO estiver no Background (Cenário Original)
            if v_tup not in voxels_bg:
                mapa_dinamico[v_tup] = [pts[i], t_atual]

    # ATUALIZAÇÃO DO GRÁFICO (10 Hz para manter Simulação a 1.0x)
    if t_atual - last_plot_time > 0.1:
        # Limpeza Temporal (Decaimento)
        mapa_dinamico = {k: v for k, v in mapa_dinamico.items() if (t_atual - v[1]) <= TEMPO_VIDA}
        
        if mapa_dinamico:
            pts_display = np.array([v[0] for v in mapa_dinamico.values()])
            # Desenha os novos objetos em VERMELHO
            scatter.set_data(pts_display, edge_color=None, face_color=(1, 0, 0, 1), size=4)
        else:
            scatter.set_data(np.zeros((0, 3)))
        
        app.process_events()
        last_plot_time = t_atual

# LIMPEZA FINAL
canvas.close()
print(f"Search concluído. Janela fechada.")