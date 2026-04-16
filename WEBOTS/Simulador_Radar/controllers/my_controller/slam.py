"""
Odometria Inercial + LiDAR + ICP (LIO Simplificado)
Combina a fluidez do IMU/Acelerómetro em tempo real com a 
correção geométrica periódica do ICP e Filtragem de Redundância (KD-Tree).
"""

import numpy as np
import os, math, threading, queue
import vispy
from vispy import app, scene
from kinematics import get_3d_points
from sklearn.cluster import DBSCAN

try:
    from scipy.spatial import cKDTree as _KDTree
    _HAS_KDTREE = True
except ImportError:
    _HAS_KDTREE = False
    print("AVISO: Instala o scipy (pip install scipy) para o ICP e Filtros funcionarem!")

# ──────────────────────────────────────────────────────────────
# PARÂMETROS
# ──────────────────────────────────────────────────────────────
TEMPO_TOTAL   = float(os.getenv("PARAM_TEMPO", 60.0))
GRAVITY       = np.array([0.0, 0.0, -9.81]) 
ACCEL_DEAD    = 0.15  
REORTHO_EVERY = 50
ARROW_LEN     = 0.5   
PLOT_INTERVAL = 1.0 / 20

# Parâmetros do LiDAR e Gimbal
Z_TORRE, L_BRACO = 0.130, 0.032
SUBSAMPLE   = 3
FREQ        = float(os.getenv("PARAM_FREQ",  0.5))
FOV_G       = float(os.getenv("PARAM_FOV",   1.047))
VOXEL_MAP   = 0.05  

# Parâmetros ICP e Filtro
T_ICP        = max(1.0 / FREQ, 2.0) # Tempo entre ciclos de ICP (em segundos)
VOXEL_ICP    = 0.08                 # Simplificação da nuvem para o ICP ser rápido
MAX_ICP_ITER = 30
ICP_MAX_DIST = 0.4                  # Distância mais curta para evitar colar paredes erradas
TOLERANCIA_MAPA = 0.075              # 5 cm: Raio para considerar que um ponto já existe

# Variáveis globais para não correr o DBSCAN sempre que a câmara mexe
_cache_mapa_len = 0
_cache_cores = None


IMU_DEVICE   = "imu"
GYRO_DEVICE  = "gyro"
ACCEL_DEVICE = "accel"

# ──────────────────────────────────────────────────────────────
# MATEMÁTICA, FÍSICA E ICP
# ──────────────────────────────────────────────────────────────

#DEFINICAO PARA FAZER DBSCAN E IDENTIFICAR ENTIDADES DIFERENTES
def _cor(pts):
    global _cache_mapa_len, _cache_cores
    
    # Só volta a calcular os clusters se o mapa tiver crescido com pontos novos
    if len(pts) == _cache_mapa_len and _cache_cores is not None:
        return _cache_cores
        
    cores = np.ones((len(pts), 4), dtype=np.float32) # Base: branco/cinza
    
    # 1. IDENTIFICAR O CHÃO (Exemplo: tudo o que estiver abaixo do lidar - 10 cm)
    # Assumindo que o lidar está a Z_TORRE (0.13m), o chão deve estar perto de Z = -0.13m em relação ao lidar
    # Mas como estamos no referencial do mundo, vamos assumir que os pontos mais baixos são o chão
    z_min = np.min(pts[:, 2]) if len(pts) > 0 else 0
    mask_chao = pts[:, 2] < (z_min + 0.15) # Tudo nos 15 cm mais baixos é chão
    
    # Pintar o chão de cinzento escuro
    cores[mask_chao] = [0.3, 0.3, 0.3, 1.0] 
    
    # 2. IDENTIFICAR OBJETOS (DBSCAN apenas no que não é chão)
    mask_objetos = ~mask_chao
    pts_objetos = pts[mask_objetos]
    
    if len(pts_objetos) > 0:
        # eps: distância máxima para dois pontos serem do mesmo cluster (30 cm)
        # min_samples: pontos mínimos para ser considerado um objeto (e não lixo/ruído)
        clustering = DBSCAN(eps=0.30, min_samples=20, n_jobs=-1).fit(pts_objetos)
        labels = clustering.labels_
        
        # Atribuir cores aleatórias aos diferentes clusters encontrados
        cores_unicas = {}
        for i, label in enumerate(labels):
            if label == -1:
                # Ruído (pontos soltos) ficam a vermelho
                cores[mask_objetos][i] = [1.0, 0.0, 0.0, 1.0] 
            else:
                if label not in cores_unicas:
                    # Gera uma cor viva aleatória para cada novo objeto detetado
                    cores_unicas[label] = np.append(np.random.rand(3) * 0.8 + 0.2, 1.0)
                cores[np.where(mask_objetos)[0][i]] = cores_unicas[label]

    _cache_mapa_len = len(pts)
    _cache_cores = cores
    return cores
    

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
    if angle < 1e-10: return R
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
    if float(np.linalg.norm(a_world)) < ACCEL_DEAD and float(np.linalg.norm(v)) < 0.10:
        v_new = np.zeros(3, dtype=np.float64)
    else:
        v_new = v + a_world * dt
    return v_new, t + v_new * dt

def _voxel(pts, size):
    if len(pts) == 0: return pts
    _, idx = np.unique(np.floor(pts / size).astype(np.int32), axis=0, return_index=True)
    return pts[idx]

def _icp_6dof(src, tgt):
    if not _HAS_KDTREE or len(src) < 10 or len(tgt) < 10:
        return np.zeros(3), np.eye(3), False

    s = _voxel(src.astype(np.float32), VOXEL_ICP).astype(np.float64)
    t = _voxel(tgt.astype(np.float32), VOXEL_ICP).astype(np.float64)

    if len(s) > 2000: s = s[np.random.choice(len(s), 2000, replace=False)]
    if len(t) > 5000: t = t[np.random.choice(len(t), 5000, replace=False)]

    tree  = _KDTree(t)
    R_acc = np.eye(3, dtype=np.float64)
    tv    = np.zeros(3, dtype=np.float64)
    pts   = s.copy()

    for _ in range(MAX_ICP_ITER):
        dists, idxs = tree.query(pts, k=1, workers=1)
        mask = dists <= min(float(np.percentile(dists, 80)), ICP_MAX_DIST)
        if mask.sum() < 6: break

        src_m = pts[mask]
        tgt_m = t[idxs[mask]]

        mu_s = src_m.mean(0)
        mu_t = tgt_m.mean(0)
        H    = (src_m - mu_s).T @ (tgt_m - mu_t)
        U, _, Vt = np.linalg.svd(H)
        R_step = Vt.T @ U.T
        
        if np.linalg.det(R_step) < 0:
            Vt[-1] *= -1
            R_step = Vt.T @ U.T
            
        t_step = mu_t - R_step @ mu_s

        pts   = (R_step @ pts.T).T + t_step
        R_acc = R_step @ R_acc
        tv    = R_step @ tv + t_step

        if float(dists[mask].mean()) < 1e-4: 
            break

    return tv, R_acc, True

# ──────────────────────────────────────────────────────────────
# VISUALIZAÇÃO (VISPY)
# ──────────────────────────────────────────────────────────────
_q    = queue.Queue(maxsize=2)
_stop = threading.Event()

def _vispy_thread():
    vispy.use('PyQt6')
    canvas = scene.SceneCanvas(keys='interactive', show=True,
                               title='SLAM com ICP e Filtro Espacial', size=(1000, 800), bgcolor='#080810')
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=60, distance=5, elevation=30, azimuth=45)
    scene.visuals.XYZAxis(parent=view.scene)
    scene.visuals.GridLines(color=(.15, .15, .15, .5), parent=view.scene)

    map_vis   = scene.visuals.Markers(parent=view.scene); map_vis.antialias = 0
    buf_vis   = scene.visuals.Markers(parent=view.scene); buf_vis.antialias = 0
    traj_vis  = scene.visuals.Line(parent=view.scene, color=(0, 1, .5, 1.), width=2, method='gl')
    robot_vis = scene.visuals.Markers(parent=view.scene); robot_vis.antialias = 0
    arrow_vis = scene.visuals.Line(parent=view.scene, color=(0, 1, 1, 1.), width=4, method='gl')
    arrow_tip = scene.visuals.Markers(parent=view.scene); arrow_tip.antialias = 0
    
    _cam_ok = [False]

    def _cor(pts):
        ranges = pts.max(0) - pts.min(0)
        ax = int(np.argmax(ranges))
        z  = pts[:, ax]; zt = (z - z.min()) / (z.max() - z.min() + 1e-9)
        r = np.clip(2*zt - .5, 0, 1).astype(np.float32)
        g = (np.clip(2*zt, 0, 1) * np.clip(2 - 2*zt, 0, 1)).astype(np.float32)
        b = np.clip(1 - 2*zt, 0, 1).astype(np.float32)
        return np.column_stack([r, g, b, np.ones(len(pts), np.float32)])

    def _update(ev):
        try:
            mapa, buffer_pts, traj, pos_robot, heading_vec, yaw_deg = _q.get_nowait()
        except queue.Empty:
            return
            
        if len(mapa) > 0:
            step = max(1, len(mapa) // 100_000)
            vis  = mapa[::step]
            map_vis.set_data(vis, edge_color=None, face_color=_cor(vis), size=2)
            
        if len(buffer_pts) > 0:
            sb = max(1, len(buffer_pts) // 10_000)
            buf_vis.set_data(buffer_pts[::sb], edge_color=None, face_color=(1., .85, 0., .5), size=2)

        if len(traj) > 1:
            traj_vis.set_data(np.array(traj, np.float32))
            
        robot_vis.set_data(np.array([pos_robot], np.float32),
                           edge_color=(1, 1, 1, 1), face_color=(1, .3, 0, 1), size=22, symbol='disc')
                           
        tip = (pos_robot + heading_vec).astype(np.float32)
        arrow_vis.set_data(np.array([pos_robot, tip], np.float32))
        arrow_tip.set_data(np.array([tip], np.float32),
                           edge_color=None, face_color=(0., 1., 1., 1.), size=10, symbol='disc')
                           
        if not _cam_ok[0] and len(mapa) > 0:
            view.camera.center = tuple(pos_robot)
            _cam_ok[0] = True

        canvas.title = (f'SLAM | Mapa={len(mapa):,} pts | Buffer={len(buffer_pts):,} pts | '
                        f'pos=({pos_robot[0]:.2f},{pos_robot[1]:.2f},{pos_robot[2]:.2f})')

    _timer = app.Timer(interval=1 / 20, connect=_update, start=True)
    @canvas.events.close.connect
    def _on_close(ev): _stop.set()
    app.run()

threading.Thread(target=_vispy_thread, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# DISPOSITIVOS WEBOTS E ESTADO INICIAL
# ──────────────────────────────────────────────────────────────
lidar    = robot.getDevice("lidar"); lidar.enable(timestep)
fov_lid  = lidar.getFov()
thetas   = np.linspace(-fov_lid / 2, fov_lid / 2, lidar.getHorizontalResolution())
thetas_s = thetas[::SUBSAMPLE]

motor_a = robot.getDevice("ANEL_INTERIOR_JOINT")
motor_b = robot.getDevice("PLATAFORMA_JOINT")
s_a = robot.getDevice("ANEL_INTERIOR_JOINT_sensor"); s_a.enable(timestep)
s_b = robot.getDevice("PLATAFORMA_JOINT_sensor");    s_b.enable(timestep)

_imu = robot.getDevice(IMU_DEVICE)
_gyro = robot.getDevice(GYRO_DEVICE)
_accel = robot.getDevice(ACCEL_DEVICE)

if _imu is not None: _imu.enable(timestep)
if _gyro is not None: _gyro.enable(timestep)
if _accel is not None: _accel.enable(timestep)

if _imu is not None:
    rpy0  = np.array(_imu.getRollPitchYaw())
    R_imu = _rpy_to_R(rpy0[0], rpy0[1], rpy0[2])
else:
    R_imu = np.eye(3, dtype=np.float64)

t_imu = np.zeros(3, dtype=np.float64)
v_imu = np.zeros(3, dtype=np.float64)

_mapa          = np.empty((0, 3), dtype=np.float32)
_buf           = np.empty((0, 3), dtype=np.float32)
_reortho_cnt   = 0
trajectory     = [t_imu.copy()]
t_inicio       = robot.getTime()
last_plot_time = t_inicio
last_icp_time  = t_inicio

print("-" * 50)
print("[SLAM] A iniciar tracking com ICP e Filtro Inteligente. Fecha a janela para terminar.")
print("-" * 50)

# ──────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ──────────────────────────────────────────────────────────────
while robot.step(timestep) != -1:
    if _stop.is_set(): break

    t_agora = robot.getTime()
    t_rel   = t_agora - t_inicio
    if t_rel > TEMPO_TOTAL: break
    dt = timestep / 1000.0   

    motor_a.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel))
    motor_b.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel + math.pi / 2))

    # 1. Odometria Inercial Rápida
    if _imu is not None:
        rpy   = np.array(_imu.getRollPitchYaw())
        R_imu = _rpy_to_R(rpy[0], rpy[1], rpy[2])
    elif _gyro is not None:
        omega        = np.array(_gyro.getValues(), dtype=np.float64)
        R_imu        = _integrate_gyro(R_imu, omega, dt)
        _reortho_cnt += 1
        if _reortho_cnt >= REORTHO_EVERY:
            R_imu = _reortho(R_imu); _reortho_cnt = 0

    if _accel is not None:
        a_body       = np.array(_accel.getValues(), dtype=np.float64)
        v_imu, t_imu = _update_pose(R_imu, v_imu, t_imu, a_body, dt)
        if np.linalg.norm(v_imu) > 0.01:
            trajectory.append(t_imu.copy())

    # 2. Leitura do LiDAR
    pts_body = get_3d_points(
        lidar.getRangeImage()[::SUBSAMPLE], thetas_s,
        s_a.getValue(), s_b.getValue(), L_BRACO, Z_TORRE
    )
    
    if len(pts_body) > 0:
        pts_body[:, 1] = -pts_body[:, 1]
        pts_w = ((R_imu @ pts_body.T).T + t_imu).astype(np.float32)
        _buf  = np.vstack([_buf, pts_w]) if len(_buf) else pts_w.copy()
        
    if len(_buf) > 30_000:
        _buf = _voxel(_buf, VOXEL_MAP)

    # 3. ICP Periódico e Filtragem de Redundância
    if t_agora - last_icp_time >= T_ICP and len(_buf) > 50:
        
        if len(_mapa) == 0:
            _mapa = _buf.copy()
            print(f"[ICP] Bootstrap: Mapa criado com {len(_mapa)} pontos.")
        else:
            distancias_sq = np.sum((_mapa - t_imu)**2, axis=1)
            mapa_local = _mapa[distancias_sq < 100.0] 
            
            if len(mapa_local) < 50: 
                mapa_local = _mapa

            t_corr, R_corr, success = _icp_6dof(_buf, mapa_local)
            
            if success:
                t_imu = t_imu + t_corr
                if _imu is None: 
                    R_imu = _reortho(R_corr @ R_imu)
                
                buf_corr = ((R_corr @ _buf.T).T + t_corr).astype(np.float32)
                
                # --- FILTRO DE REDUNDÂNCIA ---
                if _HAS_KDTREE:
                    arvore_mapa = _KDTree(_mapa)
                    distancias, _ = arvore_mapa.query(buf_corr, k=1, workers=1)
                    pontos_ineditos = buf_corr[distancias > TOLERANCIA_MAPA]
                    
                    _mapa = np.vstack([_mapa, pontos_ineditos])
                    
                    tr_val = np.clip((np.trace(R_corr) - 1.0) / 2.0, -1.0, 1.0)
                    rot_deg = math.degrees(math.acos(tr_val))
                    print(f"[ICP] ✓ Ajuste | Filtro: +{len(pontos_ineditos)} pts (Rejeitou {len(buf_corr) - len(pontos_ineditos)})")
                else:
                    _mapa = np.vstack([_mapa, buf_corr])
                    print("[ICP] ✓ Ajuste aplicado (Sem filtro KD-Tree).")
            else:
                # Se falhar a convergência, aplica o filtro na mesma com a odometria
                if _HAS_KDTREE:
                    arvore_mapa = _KDTree(_mapa)
                    distancias, _ = arvore_mapa.query(_buf, k=1, workers=1)
                    pontos_ineditos = _buf[distancias > TOLERANCIA_MAPA]
                    _mapa = np.vstack([_mapa, pontos_ineditos])
                    print(f"[ICP] ✗ Falhou | Filtro (Odometria): +{len(pontos_ineditos)} pts")
                else:
                    _mapa = np.vstack([_mapa, _buf])
                    print("[ICP] ✗ Convergência falhou. A confiar apenas na odometria inercial.")

            _mapa = _voxel(_mapa, VOXEL_MAP) 

        _buf = np.empty((0, 3), dtype=np.float32)
        last_icp_time = t_agora

    # 4. Enviar dados para a janela Vispy
    if t_agora - last_plot_time >= PLOT_INTERVAL:
        heading_vec   = (R_imu[:, 0] * ARROW_LEN).astype(np.float32)
        yaw_atual_deg = math.degrees(math.atan2(R_imu[1, 0], R_imu[0, 0]))
        
        buf_snap = _buf.copy() if len(_buf) else np.empty((0,3), np.float32)
        
        payload = (_mapa.copy(), buf_snap,
                   [p.copy() for p in trajectory],
                   t_imu.astype(np.float32), heading_vec, yaw_atual_deg)
                   
        if _q.full():
            try: _q.get_nowait()
            except queue.Empty: pass
        try: _q.put_nowait(payload)
        except queue.Full: pass
        
        last_plot_time = t_agora

_stop.set()
if len(_mapa) > 0: np.save("slam_map_icp.npy", _mapa)
print("[SLAM] Terminado.")