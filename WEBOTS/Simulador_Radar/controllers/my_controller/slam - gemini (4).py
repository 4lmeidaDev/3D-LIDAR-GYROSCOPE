"""
Odometria Inercial + Mapeamento LiDAR Direto
Mostra a posição (ponto laranja), orientação (seta azul) e constrói a point cloud
em tempo real baseada puramente na odometria inercial (IMU/Accel/Gyro).
"""

import numpy as np
import os, math, threading, queue
import vispy
from vispy import app, scene
from kinematics import get_3d_points

# ──────────────────────────────────────────────────────────────
# PARÂMETROS
# ──────────────────────────────────────────────────────────────
TEMPO_TOTAL   = float(os.getenv("PARAM_TEMPO", 60.0))
GRAVITY       = np.array([0.0, 0.0, -9.81]) # Z-up. Se Y-up, usar [0, -9.81, 0]
ACCEL_DEAD    = 0.15  # m/s² — threshold da dead-zone
REORTHO_EVERY = 50
ARROW_LEN     = 0.5   # comprimento da seta em metros
PLOT_INTERVAL = 1.0 / 20

# Parâmetros do LiDAR e Gimbal
Z_TORRE, L_BRACO = 0.130, 0.032
SUBSAMPLE   = 3
FREQ        = float(os.getenv("PARAM_FREQ",  0.5))
FOV_G       = float(os.getenv("PARAM_FOV",   1.047))
VOXEL_MAP   = 0.05  # Resolução do mapa (5cm) para não estourar a RAM

IMU_DEVICE   = "imu"
GYRO_DEVICE  = "gyro"
ACCEL_DEVICE = "accel"

# ──────────────────────────────────────────────────────────────
# MATEMÁTICA E FÍSICA
# ──────────────────────────────────────────────────────────────
def _rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [   -sp,             cp*sr,             cp*cr  ]
    ], dtype=np.float64)

def _skew(v):
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]], dtype=np.float64)

def _integrate_gyro(R, omega, dt):
    angle = float(np.linalg.norm(omega)) * dt
    if angle < 1e-10:
        return R
    ax = omega / np.linalg.norm(omega)
    K  = _skew(ax)
    dR = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return R @ dR

def _reortho(R):
    U, _, Vt = np.linalg.svd(R)
    Ro = U @ Vt
    if np.linalg.det(Ro) < 0:
        Vt[-1] *= -1; Ro = U @ Vt
    return Ro

def _update_pose(R, v, t, a_body, dt):
    a_world = R @ a_body + GRAVITY
    a_norm  = float(np.linalg.norm(a_world))
    speed   = float(np.linalg.norm(v))

    if a_norm < ACCEL_DEAD and speed < 0.10:
        v_new = np.zeros(3, dtype=np.float64)
    else:
        v_new = v + a_world * dt
    t_new = t + v_new * dt
    return v_new, t_new

def _voxel(pts, size):
    """Filtro para evitar pontos duplicados/muito densos e poupar memória."""
    if len(pts) == 0:
        return pts
    _, idx = np.unique(np.floor(pts / size).astype(np.int32), axis=0, return_index=True)
    return pts[idx]

# ──────────────────────────────────────────────────────────────
# VISUALIZAÇÃO (VISPY)
# ──────────────────────────────────────────────────────────────
_q    = queue.Queue(maxsize=2)
_stop = threading.Event()

def _vispy_thread():
    vispy.use('PyQt6')
    canvas = scene.SceneCanvas(keys='interactive', show=True,
                               title='Odometria Inercial + LiDAR', size=(1000, 800), bgcolor='#080810')
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=60, distance=5, elevation=30, azimuth=45)
    scene.visuals.XYZAxis(parent=view.scene)
    scene.visuals.GridLines(color=(.15, .15, .15, .5), parent=view.scene)

    # Elementos Visuais
    map_vis   = scene.visuals.Markers(parent=view.scene); map_vis.antialias = 0
    traj_vis  = scene.visuals.Line(parent=view.scene, color=(0, 1, .5, 1.), width=2, method='gl')
    robot_vis = scene.visuals.Markers(parent=view.scene); robot_vis.antialias = 0
    arrow_vis = scene.visuals.Line(parent=view.scene, color=(0, 1, 1, 1.), width=4, method='gl')
    arrow_tip = scene.visuals.Markers(parent=view.scene); arrow_tip.antialias = 0
    
    _cam_ok = [False]

    def _cor(pts):
        """Atribui cor à nuvem de pontos com base na altura (eixo Z)."""
        ranges = pts.max(0) - pts.min(0)
        ax = int(np.argmax(ranges))
        z  = pts[:, ax]; zt = (z - z.min()) / (z.max() - z.min() + 1e-9)
        r = np.clip(2*zt - .5, 0, 1).astype(np.float32)
        g = (np.clip(2*zt, 0, 1) * np.clip(2 - 2*zt, 0, 1)).astype(np.float32)
        b = np.clip(1 - 2*zt, 0, 1).astype(np.float32)
        return np.column_stack([r, g, b, np.ones(len(pts), np.float32)])

    def _update(ev):
        try:
            mapa, traj, pos_robot, heading_vec, yaw_deg = _q.get_nowait()
        except queue.Empty:
            return
            
        # Atualizar Point Cloud
        if len(mapa) > 0:
            step = max(1, len(mapa) // 100_000)
            vis  = mapa[::step]
            map_vis.set_data(vis, edge_color=None, face_color=_cor(vis), size=2)
            
        # Atualizar Trajetória
        if len(traj) > 1:
            traj_vis.set_data(np.array(traj, np.float32))
            
        # Atualizar Ponto Laranja (Robô)
        robot_vis.set_data(np.array([pos_robot], np.float32),
                           edge_color=(1, 1, 1, 1), face_color=(1, .3, 0, 1),
                           size=22, symbol='disc')
                           
        # Atualizar Seta Azul (Orientação)
        tip = (pos_robot + heading_vec).astype(np.float32)
        arrow_vis.set_data(np.array([pos_robot, tip], np.float32))
        arrow_tip.set_data(np.array([tip], np.float32),
                           edge_color=None, face_color=(0., 1., 1., 1.), size=10, symbol='disc')
                           
        # Seguir o robô com a câmara na primeira iteração
        if not _cam_ok[0] and len(mapa) > 0:
            view.camera.center = tuple(pos_robot)
            _cam_ok[0] = True

        canvas.title = (f'Odometria Inercial + LiDAR | mapa={len(mapa):,} pts | yaw={yaw_deg:+.1f}° | '
                        f'pos=({pos_robot[0]:.2f},{pos_robot[1]:.2f},{pos_robot[2]:.2f})')

    _timer = app.Timer(interval=1 / 20, connect=_update, start=True)

    @canvas.events.close.connect
    def _on_close(ev): _stop.set()

    app.run()

threading.Thread(target=_vispy_thread, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# DISPOSITIVOS WEBOTS
# ──────────────────────────────────────────────────────────────
# 1. LiDAR e Motores
lidar    = robot.getDevice("lidar"); lidar.enable(timestep)
fov_lid  = lidar.getFov()
thetas   = np.linspace(-fov_lid / 2, fov_lid / 2, lidar.getHorizontalResolution())
thetas_s = thetas[::SUBSAMPLE]

motor_a = robot.getDevice("ANEL_INTERIOR_JOINT")
motor_b = robot.getDevice("PLATAFORMA_JOINT")
s_a = robot.getDevice("ANEL_INTERIOR_JOINT_sensor"); s_a.enable(timestep)
s_b = robot.getDevice("PLATAFORMA_JOINT_sensor");    s_b.enable(timestep)

# 2. Sensores Inerciais
_imu = _gyro = _accel = None

try:
    _imu = robot.getDevice(IMU_DEVICE); _imu.enable(timestep)
    print(f"[NAV] ✓ InertialUnit '{IMU_DEVICE}'  → orientação ABSOLUTA")
except Exception:
    pass

try:
    _gyro = robot.getDevice(GYRO_DEVICE); _gyro.enable(timestep)
    print(f"[NAV] ✓ Gyroscope '{GYRO_DEVICE}'  → orientação")
except Exception:
    pass

try:
    _accel = robot.getDevice(ACCEL_DEVICE); _accel.enable(timestep)
    print(f"[NAV] ✓ Accelerometer '{ACCEL_DEVICE}'  → posição")
except Exception:
    print("[NAV] ✗ Accelerometer não encontrado. Posição não será atualizada.")

# ──────────────────────────────────────────────────────────────
# ESTADO INICIAL
# ──────────────────────────────────────────────────────────────
if _imu:
    rpy0  = np.array(_imu.getRollPitchYaw())
    R_imu = _rpy_to_R(rpy0[0], rpy0[1], rpy0[2])
else:
    R_imu = np.eye(3, dtype=np.float64)

t_imu = np.zeros(3, dtype=np.float64)
v_imu = np.zeros(3, dtype=np.float64)

_mapa          = np.empty((0, 3), dtype=np.float32)
_reortho_cnt   = 0
trajectory     = [t_imu.copy()]
t_inicio       = robot.getTime()
last_plot_time = t_inicio

print("-" * 50)
print("[NAV] A iniciar tracking com LiDAR. Fecha a janela para terminar.")
print("-" * 50)

# ──────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ──────────────────────────────────────────────────────────────
while robot.step(timestep) != -1:
    if _stop.is_set():
        break

    t_agora = robot.getTime()
    t_rel   = t_agora - t_inicio
    if t_rel > TEMPO_TOTAL:
        break

    dt = timestep / 1000.0   # ms → s

    # 1. Movimento do Gimbal (padrão sinusoidal)
    motor_a.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel))
    motor_b.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel + math.pi / 2))

    # 2. Atualizar Orientação (R_imu)
    if _imu:
        rpy   = np.array(_imu.getRollPitchYaw())
        R_imu = _rpy_to_R(rpy[0], rpy[1], rpy[2])
    elif _gyro:
        omega        = np.array(_gyro.getValues(), dtype=np.float64)
        R_imu        = _integrate_gyro(R_imu, omega, dt)
        _reortho_cnt += 1
        if _reortho_cnt >= REORTHO_EVERY:
            R_imu = _reortho(R_imu); _reortho_cnt = 0

    # 3. Atualizar Posição (t_imu) através do Acelerómetro
    if _accel:
        a_body       = np.array(_accel.getValues(), dtype=np.float64)
        v_imu, t_imu = _update_pose(R_imu, v_imu, t_imu, a_body, dt)
        
        # Guardar trajetória se estiver em movimento
        if np.linalg.norm(v_imu) > 0.01:
            trajectory.append(t_imu.copy())

    # 4. Processar LiDAR e construir Point Cloud
    pts_body = get_3d_points(
        lidar.getRangeImage()[::SUBSAMPLE], thetas_s,
        s_a.getValue(), s_b.getValue(), L_BRACO, Z_TORRE
    )
    
    if len(pts_body) > 0:
        # Inversão do eixo Y para corrigir o espelhamento
        pts_body[:, 1] = -pts_body[:, 1]
        
        # Rotação para o referencial do mundo (R_imu) + Translação da base (t_imu)
        pts_w = ((R_imu @ pts_body.T).T + t_imu).astype(np.float32)
        
        # Adicionar ao mapa global
        _mapa = np.vstack([_mapa, pts_w]) if len(_mapa) else pts_w.copy()
        
    # Voxel grid para manter a performance (simplifica o mapa a cada ciclo se estiver grande)
    if len(_mapa) > 20_000:
        _mapa = _voxel(_mapa, VOXEL_MAP)

    # 5. Enviar para a janela de visualização a cada PLOT_INTERVAL
    if t_agora - last_plot_time >= PLOT_INTERVAL:
        # A primeira coluna de R_imu representa o eixo X (frente) do robô no mundo
        heading_vec   = (R_imu[:, 0] * ARROW_LEN).astype(np.float32)
        yaw_atual_deg = math.degrees(math.atan2(R_imu[1, 0], R_imu[0, 0]))
        
        payload = (_mapa.copy(),
                   [p.copy() for p in trajectory],
                   t_imu.astype(np.float32),
                   heading_vec,
                   yaw_atual_deg)
                   
        if _q.full():
            try: _q.get_nowait()
            except queue.Empty: pass
        try: _q.put_nowait(payload)
        except queue.Full: pass
        
        last_plot_time = t_agora

_stop.set()

# Guardar a nuvem de pontos final
if len(_mapa) > 0:
    np.save("slam_map_simple.npy", _mapa)
    print(f"\n[NAV] ✓ slam_map_simple.npy  ({len(_mapa):,} pts guardados)")

print("[NAV] Terminado.")