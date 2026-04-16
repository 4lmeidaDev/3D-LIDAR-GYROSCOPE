"""
slam.py — LIO: Gyro-ICP SLAM
=======================================================================

PORQUÊ NÃO INTEGRAMOS O ACELERÓMETRO PARA POSIÇÃO:
  A dupla integração (accel → velocidade → posição) acumula erro como t².
  Um bias de 0.001 m/s² produz 0.05m de drift em 10s, 5m em 100s.
  Sem GPS ou referência absoluta, o robot "voa" sozinho até ao infinito.
  Esta é uma limitação física, não um bug de código.

ARQUITECTURA CORRECTA:
  • Gyroscope  → integra velocidade angular (Rodrigues) → R por timestep
                 Integração SIMPLES = deriva lenta e controlável pelo ICP
  • ICP        → ÚNICA fonte de posição (t). Sem drift acumulado.
                 Corre a cada T_ICP segundos e corrige também R
  • Cada ponto LiDAR é transformado com o R ACTUAL do giroscópio →
    orientação correcta mesmo em movimento rotacional rápido
  • Acelerómetro → reservado para futura deteção de movimento rápido
                   (NÃO usado para integração de posição)

PARA HELICÓPTERO RÁPIDO (futuro):
  Quando o robot se move mais de ICP_MAX_DIST entre ciclos ICP,
  o acelerómetro pode fornecer uma estimativa inicial da posição
  (dead-reckoning de curto prazo: ~0.5s). Mas é opcional e requer
  fusão cuidadosa. Para já, ICP_MAX_DIST=5m a cada 2s cobre até 2.5m/s.

SENSORES (todos na BASE do robot, não no gimbal):
  • Gyroscope     name="gyro"   — velocidade angular [rad/s]
  • Accelerometer name="accel"  — reservado (deteção de movimento futuro)
  • InertialUnit  name="imu"    — reservado
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

# ==============================================================================
# PARÂMETROS
# ==============================================================================
Z_TORRE, L_BRACO = 0.130, 0.032
RAIO_MIN    = 0.05
SUBSAMPLE   = 3
FREQ        = float(os.getenv("PARAM_FREQ",  0.5))
FOV_G       = float(os.getenv("PARAM_FOV",   1.047))
TEMPO_TOTAL = float(os.getenv("PARAM_TEMPO", 60.0))

if os.path.exists("scan_params.json"):
    with open("scan_params.json") as f:
        _p = json.load(f)
    FOV_G, RAIO_MIN, Z_TORRE, L_BRACO, SUBSAMPLE = (
        _p["FOV_G"], _p["RAIO_MIN"], _p["Z_TORRE"], _p["L_BRACO"], _p["SUBSAMPLE"])
    T_ICP    = max(1.0 / FREQ, 2.0)   # recalcular com FREQ real do scan_params
    VOXEL_MAP = RAIO_MIN
    print(f"[LIO] scan_params: FOV={math.degrees(FOV_G):.0f}° voxel={RAIO_MIN*100:.0f}cm T_ICP={T_ICP:.1f}s")

# Nomes dos sensores Webots
GYRO_DEVICE  = "gyro"
ACCEL_DEVICE = "accel"   # reservado — não usado para posição
IMU_DEVICE   = "imu"     # reservado

# Re-ortogonalização da matriz R (evita deriva numérica da integração gyro)
REORTHO_EVERY = 50   # a cada N timesteps

# ICP periódico — ÚNICA fonte de posição
# Deve cobrir pelo menos 1 ciclo completo do gimbal para sobreposição suficiente.
# Calculado depois de ler scan_params (depende de FREQ).
T_ICP        = max(1.0 / FREQ, 2.0)   # [s] ≥ 1 ciclo, mínimo 2s
MIN_BUF_PTS  = 50    # pontos mínimos no buffer para tentar ICP

# Mapa
VOXEL_MAP    = RAIO_MIN

# ICP
VOXEL_ICP    = 0.15
MAX_ICP_PTS  = 3000
ICP_MAX_ITER = 50
ICP_MAX_DIST = 5.0    # [m] amplo — percentil 80% faz a filtragem real
ICP_TOL      = 1e-6

# Validação ICP
ICP_RMSE_MAX = 0.40
MAX_STEP_M   = 5.0
MAX_STEP_DEG = 30.0
MIN_MOVE_M   = 0.03

PLOT_INTERVAL = 1.0 / 15

# ==============================================================================
# INTEGRAÇÃO GIROSCÓPIO
# ==============================================================================
def _skew(v):
    return np.array([[ 0.0,  -v[2],  v[1]],
                     [ v[2],  0.0,  -v[0]],
                     [-v[1],  v[0],  0.0 ]], dtype=np.float64)

def _integrate_gyro(R, omega, dt):
    """
    Rodrigues: integra velocidade angular omega [rad/s] durante dt [s].
    Retorna nova R (rotação body → mundo).
    Integração SIMPLES: erro cresce linearmente, não quadraticamente.
    """
    angle = float(np.linalg.norm(omega)) * dt
    if angle < 1e-10:
        return R
    axis = omega / np.linalg.norm(omega)
    K    = _skew(axis)
    dR   = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return R @ dR

def _reortho(R):
    """SVD re-ortogonalização — mantém R como rotação válida."""
    U, _, Vt = np.linalg.svd(R)
    Ro = U @ Vt
    if np.linalg.det(Ro) < 0:
        Vt[-1] *= -1
        Ro = U @ Vt
    return Ro

# ==============================================================================
# VOXEL DOWNSAMPLE
# ==============================================================================
def _voxel(pts, size):
    if len(pts) == 0:
        return pts
    _, idx = np.unique(np.floor(pts / size).astype(np.int32), axis=0, return_index=True)
    return pts[idx]

# ==============================================================================
# ICP — translação apenas (sem rotação)
# ==============================================================================
# PORQUÊ SÓ TRANSLAÇÃO:
#   O giroscópio já dá a rotação com precisão — não precisamos do ICP para isso.
#   ICP com R+t em scans parciais (arcos) confunde a mudança de perspetiva
#   com rotação → produz rotações espúrias mesmo em movimento linear puro.
#   Separando responsabilidades:
#     Gyro → R (rotação)      ICP → t (translação)
# ==============================================================================
def _icp_trans(src, tgt):
    """
    ICP que encontra APENAS translação.
    Em cada iteração: tv = centróide(tgt_vizinhos) - centróide(src_pts)
    Sem SVD, sem rotação — gyro trata da orientação.
    Retorna (t_vec 3, rmse, convergiu).
    """
    s = _voxel(src.astype(np.float32), VOXEL_ICP).astype(np.float64)
    t = _voxel(tgt.astype(np.float32), VOXEL_ICP).astype(np.float64)
    if len(s) > MAX_ICP_PTS:
        s = s[np.random.choice(len(s), MAX_ICP_PTS, replace=False)]
    if len(t) > MAX_ICP_PTS:
        t = t[np.random.choice(len(t), MAX_ICP_PTS, replace=False)]
    if len(s) < 10 or len(t) < 10:
        return np.zeros(3), np.inf, False

    tree      = _KDTree(t) if _HAS_KDTREE else None
    tv_acc    = np.zeros(3, dtype=np.float64)
    pts       = s.copy()
    prev_rmse = np.inf

    for _ in range(ICP_MAX_ITER):
        if tree:
            dists, idxs = tree.query(pts, k=1, workers=1)
        else:
            d2    = np.sum((t[None] - pts[:, None]) ** 2, axis=2)
            idxs  = d2.argmin(axis=1)
            dists = np.sqrt(d2[np.arange(len(pts)), idxs])

        mask = dists <= min(float(np.percentile(dists, 80)), ICP_MAX_DIST)
        if mask.sum() < 6:
            break

        # Translação pura: deslocar para o centróide dos vizinhos
        tv     = t[idxs[mask]].mean(0) - pts[mask].mean(0)
        pts    = pts + tv
        tv_acc = tv_acc + tv

        rmse = float(dists[mask].mean())
        if abs(prev_rmse - rmse) < ICP_TOL:
            return tv_acc, rmse, True
        prev_rmse = rmse

    return tv_acc, prev_rmse, False


def _icp_valido(t_vec, rmse):
    if rmse > ICP_RMSE_MAX:
        return False, f"rmse={rmse:.3f}>{ICP_RMSE_MAX}"
    d = float(np.linalg.norm(t_vec))
    if d > MAX_STEP_M:
        return False, f"|t|={d:.2f}>{MAX_STEP_M}m"
    return True, "ok"

# ==============================================================================
# MAPA GLOBAL
# ==============================================================================
_mapa     = np.empty((0, 3), dtype=np.float32)
_mapa_vox = {}

def _add_mapa(pts_world):
    global _mapa
    if len(pts_world) == 0:
        return
    keys  = np.floor(pts_world / VOXEL_MAP).astype(np.int32)
    novos = [i for i, k in enumerate(map(tuple, keys.tolist())) if k not in _mapa_vox]
    if not novos:
        return
    for i in novos:
        _mapa_vox[tuple(keys[i].tolist())] = True
    blk   = pts_world[novos].astype(np.float32)
    _mapa = np.vstack([_mapa, blk]) if len(_mapa) else blk

def _mapa_para_icp():
    if len(_mapa) == 0:
        return np.empty((0, 3), dtype=np.float64)
    m = _voxel(_mapa, VOXEL_ICP * 2)
    if len(m) > MAX_ICP_PTS * 4:
        m = m[np.random.choice(len(m), MAX_ICP_PTS * 4, replace=False)]
    return m.astype(np.float64)

# ==============================================================================
# THREAD VISPY
# ==============================================================================
_q    = queue.Queue(maxsize=2)
_stop = threading.Event()

def _vispy_thread():
    vispy.use('PyQt6')
    canvas = scene.SceneCanvas(keys='interactive', show=True,
                               title='LIO SLAM', size=(1200, 800), bgcolor='#080810')
    view   = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=60, distance=8, elevation=30, azimuth=45)
    scene.visuals.XYZAxis(parent=view.scene)
    scene.visuals.GridLines(color=(.15, .15, .15, .5), parent=view.scene)

    map_vis   = scene.visuals.Markers(parent=view.scene); map_vis.antialias = 0
    buf_vis   = scene.visuals.Markers(parent=view.scene); buf_vis.antialias = 0
    traj_vis  = scene.visuals.Line(parent=view.scene, color=(0, 1, .5, 1.), width=2, method='gl')
    robot_vis = scene.visuals.Markers(parent=view.scene); robot_vis.antialias = 0
    _cam_ok   = [False]

    def _cor_altura(pts):
        ranges = pts.max(0) - pts.min(0)
        ax     = int(np.argmax(ranges))
        z      = pts[:, ax]
        zt     = (z - z.min()) / (z.max() - z.min() + 1e-9)
        r = np.clip(2*zt - .5, 0, 1).astype(np.float32)
        g = (np.clip(2*zt, 0, 1) * np.clip(2 - 2*zt, 0, 1)).astype(np.float32)
        b = np.clip(1 - 2*zt, 0, 1).astype(np.float32)
        return np.column_stack([r, g, b, np.ones(len(pts), np.float32)])

    def _update(ev):
        try:
            mapa, buf, traj, cur_pos = _q.get_nowait()
        except queue.Empty:
            return

        if len(mapa) > 1:
            step = max(1, len(mapa) // 100_000)
            vis  = mapa[::step]
            map_vis.set_data(vis, edge_color=None, face_color=_cor_altura(vis), size=2)
            if not _cam_ok[0]:
                c  = vis.mean(0)
                sp = float(np.linalg.norm(vis.max(0) - vis.min(0)))
                view.camera.center   = tuple(c)
                view.camera.distance = max(sp * 0.9, 3.0)
                _cam_ok[0] = True

        if len(buf) > 1:
            step_b = max(1, len(buf) // 20_000)
            buf_vis.set_data(buf[::step_b], edge_color=None,
                             face_color=(1.0, 0.85, 0.0, 0.5), size=2)

        if len(traj) > 1:
            traj_vis.set_data(np.array(traj, np.float32))

        robot_vis.set_data(
            np.array([cur_pos], np.float32),
            edge_color=(1.0, 1.0, 1.0, 1.0),
            face_color=(1.0, 0.3, 0.0, 1.0),
            size=22, symbol='disc',
        )
        n_total = len(mapa) + len(buf)
        canvas.title = (f'LIO SLAM — {n_total:,} pts '
                        f'({len(mapa):,} mapa + {len(buf):,} buffer) | '
                        f'pos=({cur_pos[0]:.2f}, {cur_pos[1]:.2f}, {cur_pos[2]:.2f})')

    # Guardar referência — evita garbage collection (bug crítico)
    _timer = app.Timer(interval=1/20, connect=_update, start=True)

    @canvas.events.close.connect
    def _close(ev): _stop.set()

    app.run()

threading.Thread(target=_vispy_thread, daemon=True).start()

# ==============================================================================
# DISPOSITIVOS WEBOTS
# ==============================================================================
lidar     = robot.getDevice("lidar"); lidar.enable(timestep)
fov_lidar = lidar.getFov()
thetas    = np.linspace(-fov_lidar/2, fov_lidar/2, lidar.getHorizontalResolution())
thetas_s  = thetas[::SUBSAMPLE]

motor_a = robot.getDevice("ANEL_INTERIOR_JOINT")
motor_b = robot.getDevice("PLATAFORMA_JOINT")
s_a = robot.getDevice("ANEL_INTERIOR_JOINT_sensor"); s_a.enable(timestep)
s_b = robot.getDevice("PLATAFORMA_JOINT_sensor");    s_b.enable(timestep)

# Gyroscope — fonte principal de orientação
_gyro = None
try:
    _gyro = robot.getDevice(GYRO_DEVICE); _gyro.enable(timestep)
    print(f"[LIO] ✓ Gyroscope '{GYRO_DEVICE}' — rotação activa")
except Exception:
    print(f"[LIO] ✗ Gyroscope '{GYRO_DEVICE}' não encontrado — orientação fixa")

# Acelerómetro — ligado mas NÃO integrado (evitar drift quadrático)
_accel = None
try:
    _accel = robot.getDevice(ACCEL_DEVICE); _accel.enable(timestep)
    print(f"[LIO] ✓ Accelerometer '{ACCEL_DEVICE}' (reservado — não integrado para posição)")
except Exception:
    pass  # opcional

MODO = "Gyro+ICP" if _gyro else "ICP-only"
print(f"[LIO] Modo: {MODO} | T_ICP={T_ICP}s | voxel={VOXEL_MAP*100:.0f}cm")
print(f"[LIO] POSIÇÃO vem exclusivamente do ICP — sem drift do acelerómetro")

# ==============================================================================
# ESTADO
# ==============================================================================
R_imu = np.eye(3, dtype=np.float64)    # rotação body→mundo (gyro, por timestep)
t_imu = np.zeros(3, dtype=np.float64)  # posição no mundo   (ICP, por ciclo)

_reortho_cnt = 0
_icp_buf     = np.empty((0, 3), dtype=np.float32)
trajectory   = [t_imu.copy()]
n_icp        = 0

t_inicio       = robot.getTime()
last_icp_time  = t_inicio
last_plot_time = t_inicio

print("-" * 60)
print("[LIO] Pronto. Move o robot. Fecha a janela para parar.")
print("-" * 60)

# ==============================================================================
# LOOP PRINCIPAL
# ==============================================================================
while robot.step(timestep) != -1:
    if _stop.is_set():
        break

    t_atual = robot.getTime()
    t_rel   = t_atual - t_inicio
    if t_rel > TEMPO_TOTAL:
        break

    dt = timestep / 1000.0

    # ── 1. Gimbal sinusoidal ────────────────────────────────────────────────
    motor_a.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel))
    motor_b.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel + math.pi / 2))

    # ── 2. Giroscópio → R_imu ──────────────────────────────────────────────
    # Integração simples: erro linear no tempo (não quadrático como accel)
    # Corrigido pelo ICP a cada T_ICP segundos
    if _gyro:
        omega        = np.array(_gyro.getValues(), dtype=np.float64)
        R_imu        = _integrate_gyro(R_imu, omega, dt)
        _reortho_cnt += 1
        if _reortho_cnt >= REORTHO_EVERY:
            R_imu        = _reortho(R_imu)
            _reortho_cnt = 0

    # ── 3. LiDAR → referencial mundo ────────────────────────────────────────
    # R_imu = orientação actual (do gyro, precisa)
    # t_imu = última posição confirmada pelo ICP (estável, sem drift)
    pts_body = get_3d_points(
        lidar.getRangeImage()[::SUBSAMPLE], thetas_s,
        s_a.getValue(), s_b.getValue(), L_BRACO, Z_TORRE
    )

    if len(pts_body) > 0:
        pts_world = ((R_imu @ pts_body.T).T + t_imu).astype(np.float32)
        _icp_buf  = np.vstack([_icp_buf, pts_world]) if len(_icp_buf) else pts_world.copy()

        # Bootstrap: primeiros pontos vão directo ao mapa
        if len(_mapa) < MIN_BUF_PTS:
            _add_mapa(pts_world)

    # Downsample do buffer para limitar memória
    if len(_icp_buf) > 30_000:
        _icp_buf = _voxel(_icp_buf, VOXEL_MAP)

    # ── 4. ICP periódico — ÚNICA fonte de posição ──────────────────────────
    if (t_atual - last_icp_time >= T_ICP and
            len(_mapa) >= MIN_BUF_PTS and
            len(_icp_buf) >= MIN_BUF_PTS):

        mapa_ref           = _mapa_para_icp()
        t_corr, rmse, conv = _icp_trans(_icp_buf, mapa_ref)
        ok, motivo         = _icp_valido(t_corr, rmse)
        d                  = float(np.linalg.norm(t_corr))

        if ok:
            # Só translação — gyro mantém a rotação, ICP corrige posição
            t_imu = t_imu + t_corr

            if d > MIN_MOVE_M:
                trajectory.append(t_imu.copy())

            # Confirmar buffer no mapa com translação corrigida
            buf_corr = (_icp_buf + t_corr).astype(np.float32)
            _add_mapa(buf_corr)

            n_icp += 1
            estado = f"Δ={d:.3f}m" if d > MIN_MOVE_M else "estático"
            print(f"[LIO] ICP #{n_icp:3d} | rmse={rmse:.3f} {'✓' if conv else '~'} | "
                  f"{estado} | "
                  f"pos=({t_imu[0]:.2f}, {t_imu[1]:.2f}, {t_imu[2]:.2f}) | "
                  f"mapa={len(_mapa):,}")
        else:
            # ICP falhou: adicionar buffer sem correcção (manter t_imu)
            _add_mapa(_icp_buf.astype(np.float32))
            print(f"[LIO] ICP rejeitado ({motivo}) — posição mantida | mapa={len(_mapa):,}")

        _icp_buf      = np.empty((0, 3), dtype=np.float32)
        last_icp_time = t_atual

    # ── 5. Vispy ────────────────────────────────────────────────────────────
    if t_atual - last_plot_time >= PLOT_INTERVAL:
        buf_snap = _icp_buf.copy() if len(_icp_buf) else np.empty((0, 3), np.float32)
        payload  = (_mapa.copy(), buf_snap, [p.copy() for p in trajectory], t_imu.copy())
        if _q.full():
            try:    _q.get_nowait()
            except queue.Empty: pass
        try:    _q.put_nowait(payload)
        except queue.Full: pass
        last_plot_time = t_atual

# ==============================================================================
# GUARDAR
# ==============================================================================
_stop.set()
mapa_final = np.vstack([_mapa, _icp_buf]) if len(_icp_buf) else _mapa
if len(mapa_final) > 0:
    np.save("slam_map.npy",        mapa_final)
    np.save("slam_trajectory.npy", np.array(trajectory, dtype=np.float32))
    print(f"\n[LIO] ✓ slam_map.npy        ({len(mapa_final):,} pts)")
    print(f"[LIO] ✓ slam_trajectory.npy ({len(trajectory)} poses)")
print("[LIO] Concluído.")
