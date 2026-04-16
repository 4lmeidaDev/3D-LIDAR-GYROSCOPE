"""
slam.py  —  LIO: LiDAR Inertial Odometry  (v2 — melhorias aplicadas)
=========================================================

MELHORIAS v2:
  1. ICP 6-DOF (SVD point-to-point) — estima R_corr + t_corr juntos.
     Corrige o drift de orientação que o gyro acumula ao longo do tempo.
     A rotação já não depende só do IMU/gyro entre ciclos ICP.

  2. Loop closure simples — mantém keyframes (pose + centróide do scan).
     Quando o ICP converge perto de um keyframe anterior com scan similar,
     aplica uma correção extra à pose e regista o evento.
     Parâmetros: LC_DIST_M (raio de busca) e LC_SCAN_SIM (limiar RMSE).

  3. Sliding window para referência ICP — _mapa_icp_ref() devolve apenas
     os pontos adicionados nas últimas SLIDING_WINDOW iterações ICP.
     O mapa global continua completo para visualização e exportação,
     mas o ICP corre contra dados recentes → mais rápido e mais preciso
     em zonas com geometria repetitiva.

  4. Validação do ICP melhorada — _icp_valido agora verifica também a
     rotação estimada: se o ângulo de correção for > ICP_MAX_ROT_DEG,
     o resultado é rejeitado (provavelmente um mau alinhamento).

FLUXO:
  ┌─ CADA TIMESTEP ───────────────────────────────────────────────┐
  │  1. IMU  → R_imu  (orientação absoluta, sem drift)            │
  │  2. Accel→ remove gravidade → integra → v_imu, t_imu          │
  │  3. LiDAR → pts_body → pts_world  (com R_imu e t_imu actuais) │
  │  4. Guardar em _buf                                            │
  └───────────────────────────────────────────────────────────────┘

  ┌─ CADA T_ICP s ────────────────────────────────────────────────┐
  │  5. ICP 6-DOF: alinhar _buf ao mapa de janela deslizante      │
  │  6. t_corr, R_corr → corrigir t_imu, R_imu, v_imu            │
  │  7. Verificar loop closure contra keyframes anteriores         │
  │  8. Confirmar buffer no mapa global                            │
  └───────────────────────────────────────────────────────────────┘

SENSORES (na BASE, não no gimbal):
  • InertialUnit  "imu"   → orientação absoluta ZYX  [PRIMÁRIO]
  • Gyroscope     "gyro"  → fallback se sem IMU
  • Accelerometer "accel" → tradução em tempo real

GRAVIDADE:
  Webots Z-up → GRAVITY = [0, 0, -9.81]
  Webots Y-up → GRAVITY = [0, -9.81, 0]

NOTA SOBRE O ACELERÓMETRO:
  A dupla integração (a → v → t) acumula erro como t².
  Um bias de 0.001 m/s² produz ~5 m de drift em 100s.
  O ICP 6-DOF a cada T_ICP segundos é a única fonte de posição fiável.
  O acelerómetro serve para interpolar a pose entre ciclos ICP.
"""

import numpy as np
import os, math, threading, queue, json
import vispy
from vispy import app, scene
from kinematics import get_3d_points

try:
    from scipy.spatial import cKDTree as _KDTree
    _HAS_KDTREE = True
except ImportError:
    _HAS_KDTREE = False

# ──────────────────────────────────────────────────────────────
# PARÂMETROS
# ──────────────────────────────────────────────────────────────
Z_TORRE, L_BRACO = 0.130, 0.032
RAIO_MIN    = 0.05
SUBSAMPLE   = 3
FREQ        = float(os.getenv("PARAM_FREQ",  0.5))
FOV_G       = float(os.getenv("PARAM_FOV",   1.047))
TEMPO_TOTAL = float(os.getenv("PARAM_TEMPO", 60.0))

T_ICP     = max(1.0 / FREQ, 2.0)
VOXEL_MAP = RAIO_MIN

if os.path.exists("scan_params.json"):
    with open("scan_params.json") as f:
        _p = json.load(f)
    FOV_G, RAIO_MIN, Z_TORRE, L_BRACO, SUBSAMPLE = (
        _p["FOV_G"], _p["RAIO_MIN"], _p["Z_TORRE"], _p["L_BRACO"], _p["SUBSAMPLE"])
    T_ICP     = max(1.0 / FREQ, 2.0)
    VOXEL_MAP = RAIO_MIN
    print(f"[LIO] scan_params: FOV={math.degrees(FOV_G):.0f}°  "
          f"voxel={RAIO_MIN*100:.0f}cm  T_ICP={T_ICP:.1f}s")

IMU_DEVICE   = "imu"
GYRO_DEVICE  = "gyro"
ACCEL_DEVICE = "accel"

# Gravidade no referencial MUNDO.
# Webots Y-up (padrão): [0, -9.81, 0]
# Se o teu mundo for Z-up: [0, 0, -9.81]
GRAVITY = np.array([0.0, 0.0, -9.81])

REORTHO_EVERY = 50
MIN_BUF_PTS   = 40
ACCEL_DEAD    = 0.05   # m/s² — abaixo disto é ruído → amortece v

VOXEL_ICP       = 0.15
MAX_ICP_PTS     = 3000
ICP_MAX_ITER    = 50
ICP_MAX_DIST    = 5.0
ICP_TOL         = 1e-6
ICP_RMSE_MAX    = 0.40
MAX_STEP_M      = 5.0
MIN_MOVE_M      = 0.03
ICP_MAX_ROT_DEG = 15.0   # [NOVO] rejeitar ICP se correcção de rotação > N graus

# [NOVO] Sliding window: referência ICP usa só as últimas N iterações do mapa
SLIDING_WINDOW  = 10     # número de blocos ICP a manter na janela de referência

# [NOVO] Loop closure
LC_DIST_M   = 1.5    # raio máximo para considerar revisita (metros)
LC_SCAN_SIM = 0.12   # RMSE máximo entre scan actual e keyframe (metros)
LC_MIN_SEP  = 5      # número mínimo de ciclos ICP entre dois loop closures

PLOT_INTERVAL = 1.0 / 20

# ──────────────────────────────────────────────────────────────
# ORIENTAÇÃO  —  InertialUnit → R  (ZYX, convenção aeroespacial)
# ──────────────────────────────────────────────────────────────
def _rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [   -sp,            cp*sr,             cp*cr  ]
    ], dtype=np.float64)

# ──────────────────────────────────────────────────────────────
# ORIENTAÇÃO  —  Gyroscope → R  (fallback, Rodrigues)
# ──────────────────────────────────────────────────────────────
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
    return R @ (np.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K))

def _reortho(R):
    U, _, Vt = np.linalg.svd(R)
    Ro = U @ Vt
    if np.linalg.det(Ro) < 0:
        Vt[-1] *= -1; Ro = U @ Vt
    return Ro

# ──────────────────────────────────────────────────────────────
# POSIÇÃO  —  Acelerómetro → velocidade → posição
# ──────────────────────────────────────────────────────────────
def _update_pose(R, v, t, a_body, dt):
    """
    Física do acelerómetro:
      a_medido_body = R.T @ (a_real_mundo - GRAVITY)
      ⟹  a_real_mundo  = R @ a_medido_body + GRAVITY

    Dead-zone: se |a_world| < ACCEL_DEAD E velocidade baixa → amortece v.
    """
    a_world = R @ a_body + GRAVITY
    speed   = float(np.linalg.norm(v))
    if np.linalg.norm(a_world) < ACCEL_DEAD and speed < 0.10:
        v_new = v * math.exp(-dt * 10.0)
    else:
        v_new = v + a_world * dt
    t_new = t + v_new * dt
    return v_new, t_new

# ──────────────────────────────────────────────────────────────
# VOXEL
# ──────────────────────────────────────────────────────────────
def _voxel(pts, size):
    if len(pts) == 0:
        return pts
    _, idx = np.unique(np.floor(pts / size).astype(np.int32), axis=0, return_index=True)
    return pts[idx]

# ──────────────────────────────────────────────────────────────
# [NOVO] ICP 6-DOF  —  SVD point-to-point
#
# Estima R_corr E t_corr em simultâneo (vs versão anterior que só
# estimava translação e delegava rotação ao IMU).
# Benefício: corrige o bias acumulado do gyro a cada ciclo ICP.
# ──────────────────────────────────────────────────────────────
def _icp_6dof(src, tgt):
    """
    ICP point-to-point com estimativa conjunta de R e t via SVD.
    Retorna: (t_total, R_total, rmse, converged)
    """
    s = _voxel(src.astype(np.float32), VOXEL_ICP).astype(np.float64)
    t = _voxel(tgt.astype(np.float32), VOXEL_ICP).astype(np.float64)
    if len(s) > MAX_ICP_PTS: s = s[np.random.choice(len(s), MAX_ICP_PTS, replace=False)]
    if len(t) > MAX_ICP_PTS: t = t[np.random.choice(len(t), MAX_ICP_PTS, replace=False)]
    if len(s) < 10 or len(t) < 10:
        return np.zeros(3), np.eye(3), np.inf, False

    if not _HAS_KDTREE:
        # fallback sem scipy: só translação (comportamento antigo)
        tv   = np.zeros(3, dtype=np.float64)
        pts  = s.copy()
        prev = np.inf
        for _ in range(ICP_MAX_ITER):
            d2    = np.sum((t[None] - pts[:, None])**2, axis=2)
            idxs  = d2.argmin(axis=1)
            dists = np.sqrt(d2[np.arange(len(pts)), idxs])
            mask  = dists <= min(float(np.percentile(dists, 80)), ICP_MAX_DIST)
            if mask.sum() < 6: break
            step  = t[idxs[mask]].mean(0) - pts[mask].mean(0)
            pts  += step; tv += step
            rmse  = float(dists[mask].mean())
            if abs(prev - rmse) < ICP_TOL:
                return tv, np.eye(3), rmse, True
            prev = rmse
        return tv, np.eye(3), prev, False

    tree  = _KDTree(t)
    R_acc = np.eye(3,  dtype=np.float64)
    tv    = np.zeros(3, dtype=np.float64)
    pts   = s.copy()
    prev  = np.inf

    for _ in range(ICP_MAX_ITER):
        dists, idxs = tree.query(pts, k=1, workers=1)
        mask = dists <= min(float(np.percentile(dists, 80)), ICP_MAX_DIST)
        if mask.sum() < 6:
            break

        src_m = pts[mask]
        tgt_m = t[idxs[mask]]

        # SVD — estima R e t óptimos para este par de correspondências
        mu_s = src_m.mean(0)
        mu_t = tgt_m.mean(0)
        H    = (src_m - mu_s).T @ (tgt_m - mu_t)
        U, _, Vt = np.linalg.svd(H)
        R_step = Vt.T @ U.T
        # Garantir matriz de rotação própria (det = +1)
        if np.linalg.det(R_step) < 0:
            Vt[-1] *= -1
            R_step  = Vt.T @ U.T
        t_step = mu_t - R_step @ mu_s

        pts    = (R_step @ pts.T).T + t_step
        R_acc  = R_step @ R_acc
        tv     = R_step @ tv + t_step

        rmse = float(dists[mask].mean())
        if abs(prev - rmse) < ICP_TOL:
            return tv, R_acc, rmse, True
        prev = rmse

    return tv, R_acc, prev, False


def _rot_angle_deg(R):
    """Ângulo de rotação de uma matriz R (graus). Útil para validação."""
    cos_a = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))


def _icp_valido(tv, R, rmse):
    """
    [ACTUALIZADO] Valida resultado ICP.
    Agora verifica também o ângulo de rotação estimado:
    correções > ICP_MAX_ROT_DEG são provavelmente maus alinhamentos.
    """
    if rmse > ICP_RMSE_MAX:
        return False, f"rmse={rmse:.3f}"
    if np.linalg.norm(tv) > MAX_STEP_M:
        return False, f"|t|={np.linalg.norm(tv):.2f}m"
    rot_deg = _rot_angle_deg(R)
    if rot_deg > ICP_MAX_ROT_DEG:
        return False, f"rot={rot_deg:.1f}°"
    return True, "ok"

# ──────────────────────────────────────────────────────────────
# MAPA GLOBAL  (pontos confirmados pelo ICP)
# ──────────────────────────────────────────────────────────────
_mapa     = np.empty((0, 3), dtype=np.float32)
_mapa_vox = {}

# [NOVO] Sliding window: lista de blocos por iteração ICP
# Cada entrada é um array de pontos adicionados nessa iteração.
_mapa_blocos = []   # deque lógica; máx SLIDING_WINDOW entradas

def _add_mapa(pts):
    """Adiciona pts ao mapa global (com dedup voxel) e ao bloco da janela."""
    global _mapa
    if len(pts) == 0:
        return
    keys  = np.floor(pts / VOXEL_MAP).astype(np.int32)
    novos = [i for i, k in enumerate(map(tuple, keys.tolist())) if k not in _mapa_vox]
    if not novos:
        _mapa_blocos.append(np.empty((0, 3), dtype=np.float32))
        return
    for i in novos:
        _mapa_vox[tuple(keys[i].tolist())] = True
    blk   = pts[novos].astype(np.float32)
    _mapa = np.vstack([_mapa, blk]) if len(_mapa) else blk
    _mapa_blocos.append(blk)
    # Manter só SLIDING_WINDOW blocos
    if len(_mapa_blocos) > SLIDING_WINDOW:
        _mapa_blocos.pop(0)


def _mapa_icp_ref():
    """
    [ACTUALIZADO] Referência para o ICP = sliding window (últimos
    SLIDING_WINDOW blocos confirmados), não o mapa global inteiro.
    Mais rápido e mais preciso em ambientes com geometria repetitiva.
    Cai-back para mapa global se a janela for pequena (bootstrap).
    """
    if len(_mapa) == 0:
        return np.empty((0, 3), np.float64)
    if len(_mapa_blocos) >= 3:
        win = np.vstack([b for b in _mapa_blocos if len(b) > 0]) \
              if any(len(b) > 0 for b in _mapa_blocos) \
              else _mapa
    else:
        win = _mapa
    m = _voxel(win, VOXEL_ICP * 2)
    if len(m) > MAX_ICP_PTS * 4:
        m = m[np.random.choice(len(m), MAX_ICP_PTS * 4, replace=False)]
    return m.astype(np.float64)

# ──────────────────────────────────────────────────────────────
# [NOVO] LOOP CLOSURE
#
# Guarda keyframes (pose + centróide do scan) a cada ciclo ICP.
# Quando o robot regressa a uma zona já vista, detecta a revisita
# e aplica uma micro-correção à pose actual.
# ──────────────────────────────────────────────────────────────
_keyframes   = []   # lista de {'pos': np.array(3), 'scan': np.array(N,3)}
_lc_count    = 0    # total de loop closures detectados
_last_lc_icp = -LC_MIN_SEP   # iteração ICP do último loop closure

def _lc_adicionar_keyframe(pos, scan):
    """Guarda um keyframe se a posição for suficientemente diferente do último."""
    if len(_keyframes) == 0 or \
       np.linalg.norm(pos - _keyframes[-1]['pos']) > LC_DIST_M * 0.5:
        kf_scan = _voxel(scan.astype(np.float32), VOXEL_ICP) if len(scan) > 0 \
                  else np.empty((0, 3), np.float32)
        _keyframes.append({'pos': pos.copy(), 'scan': kf_scan})


def _lc_verificar(pos, scan_atual, n_icp_atual):
    """
    Procura keyframes próximos e verifica se o scan actual é similar.
    Retorna (t_corr, encontrado).
    Só activa se passaram LC_MIN_SEP iterações ICP desde o último LC.
    """
    global _lc_count, _last_lc_icp
    if not _HAS_KDTREE:
        return np.zeros(3), False
    if n_icp_atual - _last_lc_icp < LC_MIN_SEP:
        return np.zeros(3), False
    if len(_keyframes) < 3:
        return np.zeros(3), False

    scan_d = _voxel(scan_atual.astype(np.float32), VOXEL_ICP).astype(np.float64) \
             if len(scan_atual) > 0 else np.empty((0, 3))
    if len(scan_d) < 10:
        return np.zeros(3), False

    # Procurar keyframes dentro do raio LC_DIST_M
    # (excluir os últimos 3 para evitar false-positives com poses recentes)
    candidatos = [(i, kf) for i, kf in enumerate(_keyframes[:-3])
                  if np.linalg.norm(pos - kf['pos']) < LC_DIST_M]

    for _, kf in candidatos:
        if len(kf['scan']) < 10:
            continue
        tree_kf = _KDTree(kf['scan'].astype(np.float64))
        dists, _ = tree_kf.query(scan_d, k=1, workers=1)
        mask = dists <= ICP_MAX_DIST
        if mask.sum() < 6:
            continue
        rmse_lc = float(dists[mask].mean())
        if rmse_lc < LC_SCAN_SIM:
            # Scan similar encontrado — calcular deslocamento residual
            t_lc, R_lc, rmse_lc2, conv_lc = _icp_6dof(scan_atual, kf['scan'])
            if conv_lc and rmse_lc2 < LC_SCAN_SIM and \
               np.linalg.norm(t_lc) < LC_DIST_M and \
               _rot_angle_deg(R_lc) < ICP_MAX_ROT_DEG:
                _lc_count      += 1
                _last_lc_icp    = n_icp_atual
                print(f"[LIO] ★ Loop closure #{_lc_count}  "
                      f"rmse={rmse_lc2:.3f}  Δt={np.linalg.norm(t_lc):.3f}m  "
                      f"ΔR={_rot_angle_deg(R_lc):.1f}°")
                return t_lc, True

    return np.zeros(3), False

# ──────────────────────────────────────────────────────────────
# VISPY
# ──────────────────────────────────────────────────────────────
_q    = queue.Queue(maxsize=2)
_stop = threading.Event()

def _vispy_thread():
    vispy.use('PyQt6')
    canvas = scene.SceneCanvas(keys='interactive', show=True,
                               title='LIO SLAM v2', size=(1200, 800), bgcolor='#080810')
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=60, distance=8, elevation=30, azimuth=45)
    scene.visuals.XYZAxis(parent=view.scene)
    scene.visuals.GridLines(color=(.15, .15, .15, .5), parent=view.scene)

    map_vis   = scene.visuals.Markers(parent=view.scene); map_vis.antialias   = 0
    buf_vis   = scene.visuals.Markers(parent=view.scene); buf_vis.antialias   = 0
    traj_vis  = scene.visuals.Line(parent=view.scene, color=(0, 1, .5, 1.), width=2, method='gl')
    robot_vis = scene.visuals.Markers(parent=view.scene); robot_vis.antialias = 0
    lc_vis    = scene.visuals.Markers(parent=view.scene); lc_vis.antialias    = 0  # [NOVO] LC markers
    _cam_ok   = [False]

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
            mapa, buf, traj, pos_lidar, modo, lc_pts = _q.get_nowait()
        except queue.Empty:
            return
        if len(mapa) > 1:
            step = max(1, len(mapa) // 100_000)
            vis  = mapa[::step]
            map_vis.set_data(vis, edge_color=None, face_color=_cor(vis), size=2)
            if not _cam_ok[0]:
                c  = vis.mean(0); sp = float(np.linalg.norm(vis.max(0) - vis.min(0)))
                view.camera.center   = tuple(c)
                view.camera.distance = max(sp * .9, 3.)
                _cam_ok[0] = True
        if len(buf) > 1:
            sb = max(1, len(buf) // 20_000)
            buf_vis.set_data(buf[::sb], edge_color=None, face_color=(1., .85, 0., .6), size=2)
        if len(traj) > 1:
            traj_vis.set_data(np.array(traj, np.float32))
        robot_vis.set_data(np.array([pos_lidar], np.float32),
                           edge_color=(1, 1, 1, 1), face_color=(1, .3, 0, 1),
                           size=22, symbol='disc')
        # [NOVO] Keyframes de loop closure a magenta
        if len(lc_pts) > 0:
            lc_vis.set_data(np.array(lc_pts, np.float32),
                            edge_color=None, face_color=(1., 0., 1., 0.9), size=14, symbol='star')
        canvas.title = (f'LIO v2 [{modo}]  mapa={len(mapa):,}  buf={len(buf):,}  '
                        f'LC={_lc_count} | '
                        f'pos=({pos_lidar[0]:.2f},{pos_lidar[1]:.2f},{pos_lidar[2]:.2f})')

    _timer = app.Timer(interval=1 / 20, connect=_update, start=True)

    @canvas.events.close.connect
    def _on_close(ev): _stop.set()

    app.run()

threading.Thread(target=_vispy_thread, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# DISPOSITIVOS WEBOTS
# ──────────────────────────────────────────────────────────────
lidar    = robot.getDevice("lidar"); lidar.enable(timestep)
fov_lid  = lidar.getFov()
thetas   = np.linspace(-fov_lid / 2, fov_lid / 2, lidar.getHorizontalResolution())
thetas_s = thetas[::SUBSAMPLE]

motor_a = robot.getDevice("ANEL_INTERIOR_JOINT")
motor_b = robot.getDevice("PLATAFORMA_JOINT")
s_a = robot.getDevice("ANEL_INTERIOR_JOINT_sensor"); s_a.enable(timestep)
s_b = robot.getDevice("PLATAFORMA_JOINT_sensor");    s_b.enable(timestep)

_imu = _gyro = _accel = None

try:
    _imu = robot.getDevice(IMU_DEVICE); _imu.enable(timestep)
    print(f"[LIO] ✓ InertialUnit '{IMU_DEVICE}'  → orientação ABSOLUTA")
except Exception:
    print(f"[LIO] ✗ InertialUnit '{IMU_DEVICE}' não encontrado")

try:
    _gyro = robot.getDevice(GYRO_DEVICE); _gyro.enable(timestep)
    print(f"[LIO] ✓ Gyroscope '{GYRO_DEVICE}'  → {'validação' if _imu else 'FALLBACK orientação'}")
except Exception:
    print(f"[LIO] ✗ Gyroscope '{GYRO_DEVICE}' não encontrado")

try:
    _accel = robot.getDevice(ACCEL_DEVICE); _accel.enable(timestep)
    print(f"[LIO] ✓ Accelerometer '{ACCEL_DEVICE}'  → posição em tempo real")
except Exception:
    print(f"[LIO] ✗ Accelerometer '{ACCEL_DEVICE}' — SEM posição em tempo real!")
    print(f"[LIO]   Robot móvel sem accel = pontos na posição velha até ICP correr!")

if not _imu and not _gyro:
    print("[LIO] AVISO: Sem IMU nem Gyro — orientação fixa (R=I)")
if not _accel:
    print("[LIO] AVISO: Sem Accel — t_imu só actualiza a cada T_ICP s (ICP-only posição)")

MODO = ("IMU+Accel" if (_imu and _accel) else
        "IMU-only"  if _imu              else
        "Gyro+Accel"if (_gyro and _accel)else
        "ICP-only")

print(f"[LIO] Modo: {MODO} | T_ICP={T_ICP:.1f}s | GRAVITY={GRAVITY.tolist()}")
print(f"[LIO] Z_TORRE={Z_TORRE:.3f}m  L_BRACO={L_BRACO:.3f}m")
print(f"[LIO] ICP 6-DOF activo | sliding_window={SLIDING_WINDOW} | LC_dist={LC_DIST_M}m")

# ──────────────────────────────────────────────────────────────
# ESTADO
# ──────────────────────────────────────────────────────────────
if _imu:
    rpy0  = np.array(_imu.getRollPitchYaw())
    R_imu = _rpy_to_R(rpy0[0], rpy0[1], rpy0[2])
    print(f"[LIO] Orientação inicial: r={math.degrees(rpy0[0]):.1f}°  "
          f"p={math.degrees(rpy0[1]):.1f}°  y={math.degrees(rpy0[2]):.1f}°")
else:
    R_imu = np.eye(3, dtype=np.float64)

t_imu = np.zeros(3, dtype=np.float64)
v_imu = np.zeros(3, dtype=np.float64)

_reortho_cnt  = 0
_buf          = np.empty((0, 3), dtype=np.float32)
_bootstrapped = False
trajectory    = [t_imu.copy()]
lc_poses      = []   # [NOVO] posições onde ocorreu loop closure (para vispy)
n_icp         = 0

t_inicio       = robot.getTime()
last_icp_time  = t_inicio
last_plot_time = t_inicio

print("-" * 60)
print("[LIO] A correr. Move o robot. Fecha a janela para terminar.")
print("-" * 60)

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

    # ── 1. Gimbal sinusoidal ──────────────────────────────────
    motor_a.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel))
    motor_b.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel + math.pi / 2))

    # ── 2. Orientação: R_imu ─────────────────────────────────
    if _imu:
        rpy   = np.array(_imu.getRollPitchYaw())
        R_imu = _rpy_to_R(rpy[0], rpy[1], rpy[2])
    elif _gyro:
        omega        = np.array(_gyro.getValues(), dtype=np.float64)
        R_imu        = _integrate_gyro(R_imu, omega, dt)
        _reortho_cnt += 1
        if _reortho_cnt >= REORTHO_EVERY:
            R_imu = _reortho(R_imu); _reortho_cnt = 0

    # ── 3. Posição: t_imu (acelerómetro, tempo real) ─────────
    if _accel:
        a_body        = np.array(_accel.getValues(), dtype=np.float64)
        v_imu, t_imu  = _update_pose(R_imu, v_imu, t_imu, a_body, dt)

    # ── 4. LiDAR → referencial mundo ─────────────────────────
    pts_body = get_3d_points(
        lidar.getRangeImage()[::SUBSAMPLE], thetas_s,
        s_a.getValue(), s_b.getValue(), L_BRACO, Z_TORRE
    )
    if len(pts_body) > 0:
        pts_w = ((R_imu @ pts_body.T).T + t_imu).astype(np.float32)
        _buf  = np.vstack([_buf, pts_w]) if len(_buf) else pts_w.copy()

    if len(_buf) > 30_000:
        _buf = _voxel(_buf, VOXEL_MAP)

    # ── 5. ICP periódico ─────────────────────────────────────
    if t_agora - last_icp_time >= T_ICP and len(_buf) >= MIN_BUF_PTS:

        # Bootstrap: 1.º ciclo → mapa inicial
        if not _bootstrapped:
            _add_mapa(_buf.copy())
            _bootstrapped = True
            trajectory.append(t_imu.copy())
            _lc_adicionar_keyframe(t_imu, _buf)
            print(f"[LIO] Bootstrap: mapa inicial = {len(_mapa):,} pts")
            _buf          = np.empty((0, 3), dtype=np.float32)
            last_icp_time = t_agora

        elif len(_mapa) >= MIN_BUF_PTS:
            mapa_ref               = _mapa_icp_ref()
            # [NOVO] ICP 6-DOF em vez de _icp_trans
            t_corr, R_corr, rmse, conv = _icp_6dof(_buf, mapa_ref)
            ok, motivo             = _icp_valido(t_corr, R_corr, rmse)
            d                      = float(np.linalg.norm(t_corr))
            rot_deg                = _rot_angle_deg(R_corr)

            if ok:
                # Aplicar correcção de translação
                t_imu = t_imu + t_corr
                # [NOVO] Aplicar correcção de rotação ao R_imu
                # (só se o IMU não for absoluto — se há InertialUnit, R_imu
                #  já é absoluto e a correção de rotação do ICP é residual;
                #  aplicamos apenas se a correção for < 5° para não destabilizar)
                if not _imu or rot_deg < 5.0:
                    R_imu = _reortho(R_corr @ R_imu)

                if d < MIN_MOVE_M:
                    v_imu = np.zeros(3)
                else:
                    v_imu = t_corr / T_ICP

                if d > MIN_MOVE_M:
                    trajectory.append(t_imu.copy())

                buf_corr = (_buf + t_corr).astype(np.float32)
                _add_mapa(buf_corr)
                n_icp += 1

                # [NOVO] Verificar loop closure
                t_lc, lc_found = _lc_verificar(t_imu, buf_corr, n_icp)
                if lc_found:
                    t_imu += t_lc
                    lc_poses.append(t_imu.copy())

                # Guardar keyframe
                _lc_adicionar_keyframe(t_imu, buf_corr)

                print(f"[LIO] #{n_icp:3d}  rmse={rmse:.3f} {'✓' if conv else '~'}  "
                      f"ΔR={rot_deg:.1f}°  "
                      f"{'Δ='+f'{d:.3f}m' if d > MIN_MOVE_M else 'estático'}  "
                      f"pos=({t_imu[0]:.2f},{t_imu[1]:.2f},{t_imu[2]:.2f})  "
                      f"mapa={len(_mapa):,}  LC={_lc_count}")

            elif d < MIN_MOVE_M:
                _add_mapa(_buf.astype(np.float32))
                v_imu = np.zeros(3)
                n_icp += 1
                print(f"[LIO] #{n_icp:3d}  rotação detectada (rmse={rmse:.3f}, |t|≈0) "
                      f"→ buffer adicionado  mapa={len(_mapa):,}")

            else:
                if d < MIN_MOVE_M * 3:
                    v_imu = np.zeros(3)
                print(f"[LIO] #{n_icp:3d}  descartado ({motivo})  mapa={len(_mapa):,}")

            _buf          = np.empty((0, 3), dtype=np.float32)
            last_icp_time = t_agora

    # ── 6. Vispy ─────────────────────────────────────────────
    if t_agora - last_plot_time >= PLOT_INTERVAL:
        buf_snap  = _buf.copy() if len(_buf) else np.empty((0, 3), np.float32)
        lidar_pos = (R_imu @ np.array([0., 0., Z_TORRE]) + t_imu).astype(np.float32)
        payload   = (_mapa.copy(), buf_snap,
                     [p.copy() for p in trajectory],
                     lidar_pos, MODO,
                     lc_poses.copy())   # [NOVO] posições de LC para vispy
        if _q.full():
            try: _q.get_nowait()
            except queue.Empty: pass
        try: _q.put_nowait(payload)
        except queue.Full: pass
        last_plot_time = t_agora

# ──────────────────────────────────────────────────────────────
# GUARDAR
# ──────────────────────────────────────────────────────────────
_stop.set()
final = np.vstack([_mapa, _buf]) if len(_buf) else _mapa
if len(final) > 0:
    np.save("slam_map.npy",        final)
    np.save("slam_trajectory.npy", np.array(trajectory, np.float32))
    print(f"\n[LIO] ✓ slam_map.npy  ({len(final):,} pts)")
    print(f"[LIO] ✓ slam_trajectory.npy  ({len(trajectory)} poses)")
if len(lc_poses) > 0:
    np.save("slam_loop_closures.npy", np.array(lc_poses, np.float32))
    print(f"[LIO] ✓ slam_loop_closures.npy  ({len(lc_poses)} eventos)")
print(f"[LIO] Loop closures totais: {_lc_count}")
print("[LIO] Concluído.")