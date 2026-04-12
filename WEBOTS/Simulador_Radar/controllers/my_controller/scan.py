import numpy as np
import os
import math
from kinematics import get_3d_points

# RAIO_MIN para detetar objetos de 10cm
Z_TORRE, L_BRACO, RAIO_MIN = 0.130, 0.032, 0.05

MODO = os.getenv("MODO_SCAN", "ORIGINAL")
TEMPO_TOTAL = float(os.getenv("PARAM_TEMPO", 40.0))
FREQ = float(os.getenv("PARAM_FREQ", 0.5))
FOV_G = float(os.getenv("PARAM_FOV", 1.047))

# Dispositivos (Assumindo que robot já existe no contexto global)
lidar = robot.getDevice("lidar"); lidar.enable(timestep)
thetas = np.linspace(-lidar.getFov()/2, lidar.getFov()/2, lidar.getHorizontalResolution())
motor_a = robot.getDevice("ANEL_INTERIOR_JOINT")
motor_b = robot.getDevice("PLATAFORMA_JOINT")
s_a = robot.getDevice("ANEL_INTERIOR_JOINT_sensor"); s_a.enable(timestep)
s_b = robot.getDevice("PLATAFORMA_JOINT_sensor"); s_b.enable(timestep)

all_pts = []
print("A gravar Background...")

tempo_ini = robot.getTime()
while robot.step(timestep) != -1:
    t = robot.getTime() - tempo_ini
    if t > TEMPO_TOTAL: break
    
    if MODO == "SINOSOIDAL":
        pos_a = FOV_G * math.sin(2*math.pi*FREQ*t)
        pos_b = FOV_G * math.sin(2*math.pi*FREQ*t + math.pi/2)
    else:
        prog = (t % (TEMPO_TOTAL/2)) / (TEMPO_TOTAL/2)
        pos_a, pos_b = ((-1.57 + prog*3.14, 0) if t < TEMPO_TOTAL/2 else (0, -1.57 + prog*3.14))
    
    motor_a.setPosition(pos_a); motor_b.setPosition(pos_b)
    
    pts = get_3d_points(lidar.getRangeImage(), thetas, s_a.getValue(), s_b.getValue(), L_BRACO, Z_TORRE)
    if pts.size > 0: all_pts.append(pts)

if all_pts:
    final = np.vstack(all_pts)
    coords = np.floor(final / RAIO_MIN).astype(int)
    _, idx = np.unique(coords, axis=0, return_index=True)
    np.save("scan_otimizado.npy", final[idx])
    print(f"Scan concluído: {len(idx)} voxels guardados.")