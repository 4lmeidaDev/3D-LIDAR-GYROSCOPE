"""
slam.py — LiDAR SLAM  (ICP scan-to-MAP + deteção de movimento via IMU)
=======================================================================

ARQUITECTURA:
  • IMU  → detector de movimento durante acumulação do scan
           Se o robot se moveu a meio de um ciclo → keyframe descartado
           (evita misturar pontos de duas posições diferentes)
  • ICP  → estima R + t completos por alinhamento scan-to-MAP
           (mais robusto que scan-to-scan: usa todo o mapa como referência)

IMU (InertialUnit) — único sensor extra necessário:
  1. Selecciona o robot node no editor Webots
  2. Add child node → InertialUnit → name "imu"
  3. Grava e reinicia

USO:
  • Corre ▶ SLAM
  • Arrasta lentamente o robot pelo mundo Webots
  • Quanto mais devagar arrastas, melhor o ICP alinha
  • Fecha a janela para parar
"""

import numpy as np
import os
import math
import threading
import queue
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
RAIO_MIN         = 0.05
SUBSAMPLE        = 3
FREQ             = float(os.getenv("PARAM_FREQ",  0.5))
FOV_G            = float(os.getenv("PARAM_FOV",   1.047))
TEMPO_TOTAL      = float(os.getenv("PARAM_TEMPO", 60.0))

import json
if os.path.exists("scan_params.json"):
    with open("scan_params.json") as f:
        _p = json.load(f)
    FOV_G, RAIO_MIN, Z_TORRE, L_BRACO, SUBSAMPLE = (
        _p["FOV_G"], _p["RAIO_MIN"], _p["Z_TORRE"], _p["L_BRACO"], _p["SUBSAMPLE"])
    print(f"[SLAM] scan_params: FOV={math.degrees(FOV_G):.0f}° voxel={RAIO_MIN*100:.0f}cm")

IMU_DEVICE = "imu"      # InertialUnit no Webots — só para detetar movimento

# --- Keyframe ---
# FIXO em 0.5s independente do FREQ do gimbal.
# Mais curto = menos chance de mover durante o ciclo = ICP mais fiável.
T_KEYFRAME       = 0.5   # [s]
MIN_KF_PTS       = 30    # descartar se tiver menos pontos

# Limiar de deteção de movimento durante acumulação (via IMU)
# Se algum eixo do IMU mudou mais do que isto → keyframe descartado
IMU_MOVE_THRESH  = math.radians(3)   # 3 graus

# --- Mapa ---
VOXEL_MAP  = RAIO_MIN    # [m] resolução do mapa final (5 cm)

# --- ICP ---
VOXEL_ICP     = 0.15     # [m] downsample agressivo para ICP
MAX_ICP_PTS   = 2000     # cap de pontos (src e tgt)
ICP_MAX_ITER  = 30
ICP_MAX_DIST  = 0.50     # [m] correspondências acima disto ignoradas
ICP_TOL       = 1e-6

# Validação do resultado ICP
ICP_RMSE_MAX  = 0.20     # [m] rejeitar se RMSE > isto
MAX_STEP_M    = 1.5      # [m] translação máxima plausível por keyframe
MAX_STEP_DEG  = 60.0     # [°] rotação máxima plausível por keyframe
MIN_MOVE_M    = 0.01     # [m] abaixo → robot estático

PLOT_INTERVAL = 1.0 / 15

# ==============================================================================
# VOXEL DOWNSAMPLE
# ==============================================================================
def _voxel(pts, size):
    if len(pts) == 0:
        return pts
    _, idx = np.unique(np.floor(pts / size).astype(np.int32), axis=0, return_index=True)
    return pts[idx]

# ==============================================================================
# ICP — point-to-point, puro numpy/scipy
# ==============================================================================
def _icp(src, tgt):
    """
    Alinha src (N×3) a tgt (M×3).
    Retorna (R 3×3, t 3, rmse, convergiu).
    """
    s = _voxel(src.astype(np.float32), VOXEL_ICP).astype(np.float64)
    t = _voxel(tgt.astype(np.float32), VOXEL_ICP).astype(np.float64)
    if len(s) > MAX_ICP_PTS:
        s = s[np.random.choice(len(s), MAX_ICP_PTS, replace=False)]
    if len(t) > MAX_ICP_PTS:
        t = t[np.random.choice(len(t), MAX_ICP_PTS, replace=False)]
    if len(s) < 10 or len(t) < 10:
        return np.eye(3), np.zeros(3), np.inf, False

    tree      = _KDTree(t) if _HAS_KDTREE else None
    R_acc     = np.eye(3, dtype=np.float64)
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

        mask = dists <= min(np.percentile(dists, 80), ICP_MAX_DIST)
        if mask.sum() < 6:
            break

        si, ti    = pts[mask], t[idxs[mask]]
        cs, ct    = si.mean(0), ti.mean(0)
        U, _, Vt  = np.linalg.svd((si - cs).T @ (ti - ct))
        R         = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = Vt.T @ U.T
        tv     = ct - R @ cs
        pts    = (R @ pts.T).T + tv
        R_acc  = R @ R_acc
        tv_acc = R @ tv_acc + tv

        rmse = float(dists[mask].mean())
        if abs(prev_rmse - rmse) < ICP_TOL:
            return R_acc, tv_acc, rmse, True
        prev_rmse = rmse

    return R_acc, tv_acc, prev_rmse, False


def _icp_valido(R, t, rmse):
    if rmse > ICP_RMSE_MAX:
        return False, f"rmse={rmse:.3f}>{ICP_RMSE_MAX}"
    d = float(np.linalg.norm(t))
    if d > MAX_STEP_M:
        return False, f"|t|={d:.2f}>{MAX_STEP_M}m"
    ang = math.degrees(math.acos(max(-1., min(1., (np.trace(R) - 1) / 2))))
    if ang > MAX_STEP_DEG:
        return False, f"rot={ang:.1f}°>{MAX_STEP_DEG}°"
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
                               title='SLAM', size=(1200, 800), bgcolor='#080810')
    view   = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=60, distance=8, elevation=30, azimuth=45)
    scene.visuals.XYZAxis(parent=view.scene)
    scene.visuals.GridLines(color=(.15, .15, .15, .5), parent=view.scene)

    pts_vis   = scene.visuals.Markers(parent=view.scene); pts_vis.antialias = 0
    traj_vis  = scene.visuals.Line(parent=view.scene, color=(0, 1, .5, 1.), width=3, method='gl')
    robot_vis = scene.visuals.Markers(parent=view.scene)
    _cam_ok   = [False]

    def _cor(pts):
        z  = pts[:, 2]; zt = (z - z.min()) / (z.max() - z.min() + 1e-9)
        r  = np.clip(2*zt - .5, 0, 1).astype(np.float32)
        g  = (np.clip(2*zt, 0, 1) * np.clip(2 - 2*zt, 0, 1)).astype(np.float32)
        b  = np.clip(1 - 2*zt, 0, 1).astype(np.float32)
        return np.column_stack([r, g, b, np.ones(len(pts), np.float32)])

    def _update(ev):
        try:
            mapa, traj, n_kf, modo = _q.get_nowait()
        except queue.Empty:
            return
        if len(mapa) > 1:
            step = max(1, len(mapa) // 100_000)
            vis  = mapa[::step]
            pts_vis.set_data(vis, edge_color=None, face_color=_cor(vis), size=3)
            if not _cam_ok[0]:
                c  = vis.mean(0); sp = float(np.linalg.norm(vis.max(0) - vis.min(0)))
                view.camera.center = tuple(c); view.camera.distance = sp * 0.9
                _cam_ok[0] = True
        if len(traj) > 1:
            arr = np.array(traj, np.float32)
            traj_vis.set_data(arr)
            robot_vis.set_data(arr[[-1]], edge_color=None,
                               face_color=(1, .8, 0, 1), size=14)
        canvas.title = (f'SLAM [{modo}] — {len(mapa):,} pts | '
                        f'{n_kf} KF | '
                        f'pos=({traj[-1][0]:.2f},{traj[-1][1]:.2f},{traj[-1][2]:.2f})')

    # Guardar referência — evita garbage collection (bug crítico)
    _timer = app.Timer(interval=1/20, connect=_update, start=True)

    @canvas.events.close.connect
    def _close(ev): _stop.set()

    app.run()

threading.Thread(target=_vispy_thread, daemon=True).start()

# ==============================================================================
# DISPOSITIVOS
# ==============================================================================
lidar     = robot.getDevice("lidar"); lidar.enable(timestep)
fov_lidar = lidar.getFov()
thetas    = np.linspace(-fov_lidar/2, fov_lidar/2, lidar.getHorizontalResolution())
thetas_s  = thetas[::SUBSAMPLE]

motor_a = robot.getDevice("ANEL_INTERIOR_JOINT")
motor_b = robot.getDevice("PLATAFORMA_JOINT")
s_a = robot.getDevice("ANEL_INTERIOR_JOINT_sensor"); s_a.enable(timestep)
s_b = robot.getDevice("PLATAFORMA_JOINT_sensor");    s_b.enable(timestep)

_imu = None
try:
    _imu = robot.getDevice(IMU_DEVICE)
    _imu.enable(timestep)
    print(f"[SLAM] ✓ IMU '{IMU_DEVICE}' — deteção de movimento activa")
    MODO = "IMU+ICP"
except Exception:
    print(f"[SLAM] ✗ IMU '{IMU_DEVICE}' não encontrado — sem deteção de movimento")
    print(f"[SLAM]   Adiciona InertialUnit name='{IMU_DEVICE}' ao robot para melhor qualidade")
    MODO = "ICP"

print(f"[SLAM] T_keyframe={T_KEYFRAME}s | voxel={VOXEL_MAP*100:.0f}cm | modo={MODO}")
print(f"[SLAM] Arrasta o robot DEVAGAR — cada {T_KEYFRAME:.1f}s processa um scan")

# ==============================================================================
# ESTADO DO SLAM
# ==============================================================================
pose_R     = np.eye(3, dtype=np.float64)
pose_t     = np.zeros(3, dtype=np.float64)
trajectory = [pose_t.copy()]
n_kf       = 0

# Buffer do keyframe
_kf_vox      = {}
_kf_pts      = np.empty((0, 3), dtype=np.float32)
_imu_at_start = None   # leitura IMU no início da acumulação atual

def _kf_insert(pts):
    global _kf_pts
    if pts.size == 0:
        return
    keys  = np.floor(pts / VOXEL_MAP).astype(np.int32)
    novos = [i for i, k in enumerate(map(tuple, keys.tolist())) if k not in _kf_vox]
    if not novos:
        return
    for i in novos:
        _kf_vox[tuple(keys[i].tolist())] = True
    _kf_pts = np.vstack([_kf_pts, pts[novos]]) if len(_kf_pts) else pts[novos].copy()

def _kf_reset():
    global _kf_pts, _imu_at_start
    _kf_vox.clear()
    _kf_pts = np.empty((0, 3), dtype=np.float32)
    _imu_at_start = np.array(_imu.getRollPitchYaw()) if _imu else None

def _imu_moveu():
    """Verdadeiro se o robot se moveu (rodou) durante a acumulação do keyframe."""
    if _imu is None or _imu_at_start is None:
        return False
    curr    = np.array(_imu.getRollPitchYaw())
    delta   = np.abs(curr - _imu_at_start)
    # Corrigir wrap-around para ângulos perto de ±π
    delta   = np.minimum(delta, 2 * math.pi - delta)
    return float(delta.max()) > IMU_MOVE_THRESH

# ==============================================================================
# LOOP PRINCIPAL
# ==============================================================================
t_inicio       = robot.getTime()
last_kf_time   = t_inicio
last_plot_time = t_inicio

_kf_reset()   # inicializar IMU at start

print("-" * 60)
print("[SLAM] Pronto. Fecha a janela para parar e guardar o mapa.")
print("-" * 60)

while robot.step(timestep) != -1:
    if _stop.is_set():
        break

    t_atual = robot.getTime()
    t_rel   = t_atual - t_inicio
    if t_rel > TEMPO_TOTAL:
        break

    # Gimbal sinusoidal (igual ao scan/search)
    motor_a.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel))
    motor_b.setPosition(FOV_G * math.sin(2 * math.pi * FREQ * t_rel + math.pi / 2))

    # Acumular pontos no referencial do corpo do robot
    pts_body = get_3d_points(
        lidar.getRangeImage()[::SUBSAMPLE], thetas_s,
        s_a.getValue(), s_b.getValue(), L_BRACO, Z_TORRE
    )
    _kf_insert(pts_body)

    # ── Processar keyframe ──────────────────────────────────────────────────
    if (t_atual - last_kf_time) < T_KEYFRAME:
        pass  # ainda a acumular

    else:
        # Verificar se o robot se mexeu A MEIO do ciclo (keyframe inválido)
        if _imu_moveu():
            print(f"[SLAM] KF descartado — robot moveu-se durante acumulação "
                  f"(ciclo={T_KEYFRAME}s) → tenta arrastar mais devagar")
            _kf_reset()
            last_kf_time = t_atual
        else:
            kf_body = _kf_pts.copy()
            _kf_reset()
            last_kf_time = t_atual

            if len(kf_body) < MIN_KF_PTS:
                print(f"[SLAM] KF ignorado — {len(kf_body)} pts < {MIN_KF_PTS}")

            else:
                # Colocar keyframe na pose estimada actual → referencial mundo
                kf_world_est = (pose_R @ kf_body.T).T + pose_t

                mapa_ref = _mapa_para_icp()

                if len(mapa_ref) < 10:
                    # ── Primeiro keyframe ──────────────────────────────────
                    _add_mapa(kf_world_est.astype(np.float32))
                    trajectory.append(pose_t.copy())
                    n_kf += 1
                    print(f"[SLAM] KF {n_kf:3d} | {len(kf_body):5d} pts | "
                          f"referência inicial | mapa={len(_mapa):,}")

                else:
                    # ── ICP scan-to-MAP ────────────────────────────────────
                    R_corr, t_corr, rmse, conv = _icp(kf_world_est, mapa_ref)
                    ok, motivo = _icp_valido(R_corr, t_corr, rmse)

                    if not ok:
                        # ICP falhou: adicionar na pose actual, sem actualizar pose
                        _add_mapa(kf_world_est.astype(np.float32))
                        n_kf += 1
                        print(f"[SLAM] KF {n_kf:3d} | {len(kf_body):5d} pts | "
                              f"⚠ ICP rejeitado ({motivo}) | pose mantida | mapa={len(_mapa):,}")

                    else:
                        d = float(np.linalg.norm(t_corr))

                        # Actualizar pose: compor com correcção ICP
                        # kf_world_correcto = R_corr @ kf_world_est + t_corr
                        #                   = R_corr @ (pose_R @ kf_body + pose_t) + t_corr
                        # → pose_R_new = R_corr @ pose_R
                        # → pose_t_new = R_corr @ pose_t + t_corr
                        pose_R = R_corr @ pose_R
                        pose_t = R_corr @ pose_t + t_corr

                        # Pontos finais no mapa com pose corrigida
                        kf_world_final = (pose_R @ kf_body.T).T + pose_t
                        _add_mapa(kf_world_final.astype(np.float32))

                        if d > MIN_MOVE_M:
                            trajectory.append(pose_t.copy())

                        n_kf += 1
                        estado = f"Δ={d:.3f}m" if d > MIN_MOVE_M else "estático"
                        print(f"[SLAM] KF {n_kf:3d} | {len(kf_body):5d} pts | "
                              f"rmse={rmse:.3f} {'✓' if conv else '~'} | "
                              f"{estado} | "
                              f"pos=({pose_t[0]:.2f},{pose_t[1]:.2f},{pose_t[2]:.2f}) | "
                              f"mapa={len(_mapa):,}")

    # ── Vispy ───────────────────────────────────────────────────────────────
    if t_atual - last_plot_time >= PLOT_INTERVAL and len(_mapa) > 0:
        payload = (_mapa.copy(), [p.copy() for p in trajectory], n_kf, MODO)
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
if len(_mapa) > 0:
    np.save("slam_map.npy",        _mapa)
    np.save("slam_trajectory.npy", np.array(trajectory, dtype=np.float32))
    print(f"\n[SLAM] ✓ slam_map.npy        ({len(_mapa):,} pts)")
    print(f"[SLAM] ✓ slam_trajectory.npy ({len(trajectory)} poses)")
print("[SLAM] Concluído.")
