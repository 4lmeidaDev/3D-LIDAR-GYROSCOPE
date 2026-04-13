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
# CONFIGURAÇÕES — os parâmetros de MOVIMENTO podem ser diferentes do scan
# mas os parâmetros de CINEMÁTICA têm de ser IGUAIS ao scan.
# São carregados automaticamente do scan_params.json gerado pelo scan.py.
# ==============================================================================
Z_TORRE, L_BRACO = 0.130, 0.032
RAIO_MIN         = 0.05
SUBSAMPLE        = 3

FREQ        = float(os.getenv("PARAM_FREQ",  0.5))
FOV_G       = float(os.getenv("PARAM_FOV",   1.047))
TEMPO_TOTAL = float(os.getenv("PARAM_TEMPO", 40.0))

PLOT_INTERVAL = 1.0 / 20

# Carregar parâmetros críticos de cinemática do scan — sobrepõem a GUI
import json
path_params = "scan_params.json"
if os.path.exists(path_params):
    with open(path_params) as f:
        p = json.load(f)
    FOV_G_scan = p["FOV_G"]
    RAIO_MIN   = p["RAIO_MIN"]
    Z_TORRE    = p["Z_TORRE"]
    L_BRACO    = p["L_BRACO"]
    SUBSAMPLE  = p["SUBSAMPLE"]

    if abs(FOV_G - FOV_G_scan) > 0.01:
        print(f"[SEARCH DBSCAN] AVISO: FOV da GUI ({math.degrees(FOV_G):.1f}°) é diferente do scan ({math.degrees(FOV_G_scan):.1f}°).")
        print(f"[SEARCH DBSCAN]        A usar FOV do scan para garantir cobertura correta.")
    FOV_G = FOV_G_scan
    print(f"[SEARCH DBSCAN] Parâmetros do scan carregados: FOV={math.degrees(FOV_G):.1f}° | RAIO={RAIO_MIN*100:.0f}cm | SUBSAMPLE=1/{int(SUBSAMPLE)}")
else:
    print(f"[SEARCH DBSCAN] AVISO: scan_params.json não encontrado — a usar valores padrão.")
    print(f"[SEARCH DBSCAN]        Garante que scan.py foi corrido nesta sessão.")

T_CICLO    = 1.0 / FREQ
TEMPO_VIDA = T_CICLO

VOTOS_MINIMOS = 2

# ==============================================================================
# DBSCAN — separação de entidades distintas
# ==============================================================================
DBSCAN_EPS         = 0.30   # [m] distância máxima entre pontos do mesmo cluster
DBSCAN_MIN_SAMPLES = 5      # pontos mínimos para formar um cluster

# Paleta de cores por cluster (RGB float)
_PALETTE = [
    (1.00, 0.20, 0.20),  # vermelho
    (0.20, 0.90, 0.20),  # verde
    (0.20, 0.50, 1.00),  # azul
    (1.00, 0.80, 0.00),  # amarelo
    (1.00, 0.40, 0.00),  # laranja
    (0.80, 0.20, 1.00),  # roxo
    (0.00, 0.90, 0.90),  # ciano
    (1.00, 0.00, 0.60),  # rosa
]

def _aplicar_dbscan(pts):
    """
    DBSCAN sem sklearn.
    Usa scipy.spatial.cKDTree se disponível (rápido, O(N log N)).
    Fallback: numpy puro (O(N²) — aceitável para poucos milhares de pontos).
    Retorna labels: -1 = ruído, 0..K = cluster.
    """
    n = len(pts)
    if n < DBSCAN_MIN_SAMPLES:
        return np.full(n, -1, dtype=np.int32)

    # --- Cálculo de vizinhanças ---
    if _HAS_KDTREE:
        neigh = _KDTree(pts).query_ball_point(pts, DBSCAN_EPS)
    else:
        # O(N²) vetorizado em chunks para limitar RAM
        eps2  = DBSCAN_EPS ** 2
        neigh = [None] * n
        CHUNK = 256
        for i in range(0, n, CHUNK):
            blk  = pts[i:i + CHUNK]
            d2   = ((blk[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
            for j in range(len(blk)):
                neigh[i + j] = np.where(d2[j] <= eps2)[0]

    # --- Expansão dos clusters ---
    labels  = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cid     = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = neigh[i]
        if len(nb) < DBSCAN_MIN_SAMPLES:
            continue                          # ruído (por agora)
        labels[i] = cid
        fila  = list(nb)
        qi    = 0
        while qi < len(fila):
            j = int(fila[qi])
            if not visited[j]:
                visited[j] = True
                nbj = neigh[j]
                if len(nbj) >= DBSCAN_MIN_SAMPLES:
                    fila.extend(nbj)
            if labels[j] < 0:
                labels[j] = cid
            qi += 1
        cid += 1

    return labels

# ==============================================================================
# CARREGAR BACKGROUND — scan_otimizado.npy gerado pelo scan.py
# ==============================================================================
voxels_bg = set()
bg_pts_display = None
path_npy = "scan_otimizado.npy"

if os.path.exists(path_npy):
    print(f"[SEARCH DBSCAN] A carregar background: {path_npy}")
    data  = np.load(path_npy).astype(np.float32)
    bg_pts_display = data

    v_idx = np.floor(data / RAIO_MIN).astype(np.int32)

    offsets = np.array([[dx, dy, dz]
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)], dtype=np.int32)
    padded = (v_idx[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    voxels_bg = set(map(tuple, padded.tolist()))

    print(f"[SEARCH DBSCAN] Background carregado: {len(data):,} voxels")
    print(f"[SEARCH DBSCAN] Zona protegida (com padding 3x3x3): {len(voxels_bg):,} células")
    print(f"[SEARCH DBSCAN] RAIO_MIN={RAIO_MIN*100:.0f}cm | VOTOS={VOTOS_MINIMOS} | TEMPO_VIDA={TEMPO_VIDA:.1f}s")
    print(f"[SEARCH DBSCAN] DBSCAN eps={DBSCAN_EPS}m | min_samples={DBSCAN_MIN_SAMPLES}")
else:
    print("[SEARCH DBSCAN] AVISO: scan_otimizado.npy não encontrado!")
    print("[SEARCH DBSCAN]        Corre o scan.py primeiro para mapear o background.")

# ==============================================================================
# THREAD VISPY — janela 3D independente, não bloqueia o Webots
# ==============================================================================
_data_queue = queue.Queue(maxsize=2)
_stop_event = threading.Event()

def _vispy_thread():
    vispy.use('PyQt6')

    canvas = scene.SceneCanvas(
        keys='interactive',
        show=True,
        title='RADAR SEARCH DBSCAN — Deteção de Entidades',
        size=(1024, 768),
        bgcolor='#0d0d1a',
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        fov=60, distance=5.0, elevation=25, azimuth=45,
    )
    scene.visuals.XYZAxis(parent=view.scene)
    scene.visuals.GridLines(color=(0.25, 0.25, 0.25, 0.4), parent=view.scene)

    # Background a cinzento muito suave — contexto visual da sala
    scatter_bg = scene.visuals.Markers(parent=view.scene)
    scatter_bg.antialias = 0
    if bg_pts_display is not None and len(bg_pts_display) > 0:
        step = max(1, len(bg_pts_display) // 40_000)
        pts_bg = bg_pts_display[::step]
        scatter_bg.set_data(
            pts_bg,
            edge_color=None,
            face_color=(0.18, 0.18, 0.28, 0.25),
            size=2,
        )
        centro = pts_bg.mean(axis=0)
        span   = np.linalg.norm(pts_bg.max(axis=0) - pts_bg.min(axis=0))
        view.camera.center   = tuple(centro)
        view.camera.distance = float(span) * 0.8

    # Pontos detetados — coloridos por cluster DBSCAN
    scatter = scene.visuals.Markers(parent=view.scene)
    scatter.antialias = 0

    def _colorir_por_cluster(pts, labels):
        cores = np.zeros((len(pts), 4), dtype=np.float32)
        for i, lbl in enumerate(labels):
            if lbl < 0:
                cores[i] = (0.45, 0.45, 0.45, 0.30)   # ruído — cinzento transparente
            else:
                r, g, b = _PALETTE[lbl % len(_PALETTE)]
                cores[i] = (r, g, b, 1.0)
        return cores

    def _update(ev):
        try:
            payload = _data_queue.get_nowait()
        except queue.Empty:
            return
        if payload is None:
            scatter.set_data(np.zeros((1, 3), dtype=np.float32),
                             edge_color=None, face_color=(0, 0, 0, 0), size=1)
            return
        pts, labels = payload
        n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
        scatter.set_data(
            pts.astype(np.float32),
            edge_color=None,
            face_color=_colorir_por_cluster(pts, labels),
            size=6,
        )
        canvas.title = f'RADAR SEARCH DBSCAN — {n_clusters} entidade(s) | {len(pts)} pts'

    # Guardar referência para evitar garbage collection (bug crítico!)
    _timer = app.Timer(interval=1/30, connect=_update, start=True)

    @canvas.events.close.connect
    def _on_close(ev):
        _stop_event.set()

    app.run()

_t = threading.Thread(target=_vispy_thread, daemon=True)
_t.start()

# ==============================================================================
# DISPOSITIVOS E SENSORES  (robot e timestep herdados do my_controller.py)
# ==============================================================================
lidar      = robot.getDevice("lidar"); lidar.enable(timestep)
fov_lidar  = lidar.getFov()
thetas     = np.linspace(-fov_lidar / 2, fov_lidar / 2, lidar.getHorizontalResolution())
thetas_sub = thetas[::SUBSAMPLE]

motor_a = robot.getDevice("ANEL_INTERIOR_JOINT")
motor_b = robot.getDevice("PLATAFORMA_JOINT")
s_a     = robot.getDevice("ANEL_INTERIOR_JOINT_sensor"); s_a.enable(timestep)
s_b     = robot.getDevice("PLATAFORMA_JOINT_sensor");    s_b.enable(timestep)

# ==============================================================================
# BUFFER DE PONTOS COM SISTEMA DE VOTOS
# ==============================================================================
MAX_PONTOS = 20_000
vox_keys   = np.empty((MAX_PONTOS, 3), dtype=np.int32)
vox_pts    = np.empty((MAX_PONTOS, 3), dtype=np.float32)
vox_times  = np.full(MAX_PONTOS, -9999.0, dtype=np.float32)
vox_votos  = np.zeros(MAX_PONTOS, dtype=np.int16)
n_pontos   = 0
vox_index  = {}   # { (ix,iy,iz): índice }

def _inserir(new_pts, new_vkeys, t):
    global n_pontos
    for i in range(len(new_pts)):
        k = (int(new_vkeys[i, 0]), int(new_vkeys[i, 1]), int(new_vkeys[i, 2]))
        if k in vox_index:
            idx = vox_index[k]
            if vox_times[idx] < (t - TEMPO_VIDA):
                vox_pts[idx]   = new_pts[i]
                vox_votos[idx] = 1
            else:
                vox_pts[idx]   = vox_pts[idx] * 0.7 + new_pts[i] * 0.3
                if vox_votos[idx] < 32767:
                    vox_votos[idx] += 1
            vox_times[idx] = t
        elif n_pontos < MAX_PONTOS:
            vox_index[k]        = n_pontos
            vox_keys[n_pontos]  = new_vkeys[i]
            vox_pts[n_pontos]   = new_pts[i]
            vox_times[n_pontos] = t
            vox_votos[n_pontos] = 1
            n_pontos           += 1

def _limpar(t):
    global n_pontos, vox_index
    if n_pontos == 0:
        return
    vivos = vox_times[:n_pontos] >= (t - TEMPO_VIDA)
    if np.all(vivos):
        return
    idx_v  = np.where(vivos)[0]
    novo_n = len(idx_v)
    vox_keys[:novo_n]  = vox_keys[idx_v]
    vox_pts[:novo_n]   = vox_pts[idx_v]
    vox_times[:novo_n] = vox_times[idx_v]
    vox_votos[:novo_n] = vox_votos[idx_v]
    vox_times[novo_n:] = -9999.0
    vox_votos[novo_n:] = 0
    vox_index = {
        (int(vox_keys[i, 0]), int(vox_keys[i, 1]), int(vox_keys[i, 2])): i
        for i in range(novo_n)
    }
    n_pontos = novo_n

def _pontos_confiaveis():
    if n_pontos == 0:
        return None
    mask = vox_votos[:n_pontos] >= VOTOS_MINIMOS
    if not np.any(mask):
        return None
    return vox_pts[:n_pontos][mask].copy()

# ==============================================================================
# LOOP PRINCIPAL DO SEARCH DBSCAN
# ==============================================================================
last_plot_time  = 0.0
last_report     = 0.0
REPORT_INTERVAL = 10.0

t_inicio = robot.getTime()
print(f"[SEARCH DBSCAN] A iniciar deteção com clustering em tempo real.")
print(f"[SEARCH DBSCAN] Fecha a janela 3D para parar.")
print("-" * 52)

while robot.step(timestep) != -1:

    if _stop_event.is_set():
        break

    t_atual = robot.getTime()
    t_rel   = t_atual - t_inicio
    if t_rel > TEMPO_TOTAL:
        break

    # --- MOVIMENTO SINUSOIDAL --- IGUAL ao scan.py ---
    pos_a = FOV_G * math.sin(2 * math.pi * FREQ * t_rel)
    pos_b = FOV_G * math.sin(2 * math.pi * FREQ * t_rel + math.pi / 2)
    motor_a.setPosition(pos_a)
    motor_b.setPosition(pos_b)

    # --- CAPTURA E CINEMÁTICA ---
    alpha  = s_a.getValue()
    beta   = s_b.getValue()
    ranges = lidar.getRangeImage()
    pts    = get_3d_points(
        ranges[::SUBSAMPLE], thetas_sub,
        alpha, beta,
        L_BRACO, Z_TORRE
    )

    # --- FILTRAGEM DE BACKGROUND ---
    if pts.size > 0:
        v_coords = np.floor(pts / RAIO_MIN).astype(np.int32)

        novos_mask = np.array(
            [tuple(v_coords[i].tolist()) not in voxels_bg
             for i in range(len(v_coords))],
            dtype=bool
        )

        pts_novos   = pts[novos_mask]
        vkeys_novos = v_coords[novos_mask]

        if len(pts_novos) > 0:
            _inserir(pts_novos, vkeys_novos, t_atual)

    # --- LIMPEZA CONTÍNUA ---
    _limpar(t_atual)

    # --- PROGRESSO NA CONSOLA ---
    if t_atual - last_report >= REPORT_INTERVAL:
        confiaveis  = int(np.sum(vox_votos[:n_pontos] >= VOTOS_MINIMOS)) if n_pontos > 0 else 0
        snap_report = _pontos_confiaveis()
        if snap_report is not None:
            lbl_report = _aplicar_dbscan(snap_report)
            n_ent      = int(lbl_report.max()) + 1 if lbl_report.max() >= 0 else 0
        else:
            n_ent = 0
        print(f"[SEARCH DBSCAN] t={t_rel:.0f}s | confirmados={confiaveis} | entidades={n_ent}")
        last_report = t_atual

    # --- DBSCAN + ENVIO PARA VISPY ---
    if t_atual - last_plot_time > PLOT_INTERVAL:
        snap    = _pontos_confiaveis()
        payload = None
        if snap is not None:
            labels  = _aplicar_dbscan(snap)
            payload = (snap, labels)

        if _data_queue.full():
            try:    _data_queue.get_nowait()
            except queue.Empty: pass
        try:    _data_queue.put_nowait(payload)
        except queue.Full: pass

        last_plot_time = t_atual

# ==============================================================================
# LIMPEZA FINAL
# ==============================================================================
_stop_event.set()
_t.join(timeout=2.0)
print("[SEARCH DBSCAN] Concluído.")
