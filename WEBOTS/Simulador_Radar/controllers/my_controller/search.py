import numpy as np
import os
import math
import threading
import queue
import vispy
from vispy import app, scene
from kinematics import get_3d_points

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
        print(f"[SEARCH] AVISO: FOV da GUI ({math.degrees(FOV_G):.1f}°) é diferente do scan ({math.degrees(FOV_G_scan):.1f}°).")
        print(f"[SEARCH]        A usar FOV do scan para garantir cobertura correta.")
    FOV_G = FOV_G_scan
    print(f"[SEARCH] Parâmetros do scan carregados: FOV={math.degrees(FOV_G):.1f}° | RAIO={RAIO_MIN*100:.0f}cm | SUBSAMPLE=1/{int(SUBSAMPLE)}")
else:
    print(f"[SEARCH] AVISO: scan_params.json não encontrado — a usar valores padrão.")
    print(f"[SEARCH]        Garante que scan.py foi corrido nesta sessão.")

T_CICLO    = 1.0 / FREQ
TEMPO_VIDA = T_CICLO

VOTOS_MINIMOS = 2

# ==============================================================================
# CARREGAR BACKGROUND — scan_otimizado.npy gerado pelo scan.py
#
# Construímos um set de voxels com padding 3x3x3 (cubo de 15cm à volta de cada
# ponto do background). Qualquer ponto novo que caia dentro deste cubo é
# considerado background e ignorado. Só o que está FORA é "objeto novo".
# ==============================================================================
voxels_bg = set()
bg_pts_display = None
path_npy = "scan_otimizado.npy"

if os.path.exists(path_npy):
    print(f"[SEARCH] A carregar background: {path_npy}")
    data  = np.load(path_npy).astype(np.float32)
    bg_pts_display = data

    # Converter para índices de voxel
    v_idx = np.floor(data / RAIO_MIN).astype(np.int32)

    # Padding 3x3x3 vetorizado — sem loops Python
    offsets = np.array([[dx, dy, dz]
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)], dtype=np.int32)          # (27, 3)
    padded = (v_idx[:, None, :] + offsets[None, :, :]).reshape(-1, 3)   # (N*27, 3)
    voxels_bg = set(map(tuple, padded.tolist()))

    print(f"[SEARCH] Background carregado: {len(data):,} voxels")
    print(f"[SEARCH] Zona protegida (com padding 3x3x3): {len(voxels_bg):,} células")
    print(f"[SEARCH] RAIO_MIN={RAIO_MIN*100:.0f}cm | VOTOS={VOTOS_MINIMOS} | TEMPO_VIDA={TEMPO_VIDA:.1f}s")
else:
    print("[SEARCH] AVISO: scan_otimizado.npy não encontrado!")
    print("[SEARCH]        Corre o scan.py primeiro para mapear o background.")

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
        title='RADAR SEARCH — Deteção de Intruso',
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
        # Ajustar câmara ao tamanho real da sala
        centro = pts_bg.mean(axis=0)
        span   = np.linalg.norm(pts_bg.max(axis=0) - pts_bg.min(axis=0))
        view.camera.center   = tuple(centro)
        view.camera.distance = float(span) * 0.8

    # Pontos detetados (objetos novos) — coloridos por altura
    scatter = scene.visuals.Markers(parent=view.scene)
    scatter.antialias = 0

    def _colorir_por_altura(pts):
        z = pts[:, 2]
        t = (z - z.min()) / (z.max() - z.min() + 1e-9)
        r = np.clip(2.0 * t - 0.5, 0, 1).astype(np.float32)
        g = (np.clip(2.0 * t, 0, 1) * np.clip(2.0 - 2.0 * t, 0, 1)).astype(np.float32)
        b = np.clip(1.0 - 2.0 * t, 0, 1).astype(np.float32)
        a = np.ones(len(pts), dtype=np.float32)
        return np.column_stack((r, g, b, a))

    def _update(ev):
        try:
            payload = _data_queue.get_nowait()
        except queue.Empty:
            return
        if payload is None or len(payload) == 0:
            scatter.set_data(np.zeros((1, 3), dtype=np.float32),
                             edge_color=None, face_color=(0, 0, 0, 0), size=1)
            return
        scatter.set_data(
            payload.astype(np.float32),
            edge_color=None,
            face_color=_colorir_por_altura(payload),
            size=6,
        )
        canvas.title = f'RADAR SEARCH — {len(payload)} objetos detetados'

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
#
# Um voxel só aparece no gráfico depois de ser "visto" VOTOS_MINIMOS vezes.
# Isto elimina reflexos e ruído pontual do lidar.
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
            # Se o ponto já expirou (o lidar completou 1 ciclo e voltou),
            # trata-o como novo: reset de votos e posição.
            if vox_times[idx] < (t - TEMPO_VIDA):
                vox_pts[idx]   = new_pts[i]
                vox_votos[idx] = 1
            else:
                # Ainda dentro do ciclo: suavizar posição e acumular votos
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
# LOOP PRINCIPAL DO SEARCH
# ==============================================================================
last_plot_time  = 0.0
last_report     = 0.0
REPORT_INTERVAL = 10.0

t_inicio = robot.getTime()
print(f"[SEARCH] A iniciar deteção em tempo real.")
print(f"[SEARCH] Fecha a janela 3D para parar.")
print("-" * 52)

while robot.step(timestep) != -1:

    if _stop_event.is_set():
        break

    if globals().get('_ctrl_stop') and _ctrl_stop.is_set():
        print("[SEARCH] Parado pelo controlador.")
        break

    t_atual = robot.getTime()
    t_rel   = t_atual - t_inicio

    # Live settings — atualiza FREQ e TEMPO (FOV mantém-se do scan_params.json)
    _ls = globals().get('_live_settings')
    if _ls:
        FREQ        = _ls.get("FREQ",  FREQ)
        TEMPO_TOTAL = _ls.get("TEMPO", TEMPO_TOTAL)

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

    # --- FILTRAGEM DE BACKGROUND — 100% no mesmo referencial do scan.py ---
    # Os pontos XYZ são calculados com a MESMA função get_3d_points e os MESMOS
    # parâmetros, logo estão no mesmo referencial. A comparação por voxel é direta.
    if pts.size > 0:
        v_coords = np.floor(pts / RAIO_MIN).astype(np.int32)

        # Verificar quais voxels NÃO estão no background (são objetos novos)
        novos_mask = np.array(
            [tuple(v_coords[i].tolist()) not in voxels_bg
             for i in range(len(v_coords))],
            dtype=bool
        )

        pts_novos   = pts[novos_mask]
        vkeys_novos = v_coords[novos_mask]

        if len(pts_novos) > 0:
            _inserir(pts_novos, vkeys_novos, t_atual)

    # --- LIMPEZA CONTÍNUA — corre cada frame para garantir precisão de 1 ciclo ---
    _limpar(t_atual)

    # --- PROGRESSO NA CONSOLA ---
    if t_atual - last_report >= REPORT_INTERVAL:
        confiaveis = int(np.sum(vox_votos[:n_pontos] >= VOTOS_MINIMOS)) if n_pontos > 0 else 0
        print(f"[SEARCH] t={t_rel:.0f}s | voxels totais={n_pontos} | confirmados={confiaveis}")
        last_report = t_atual

    # --- ENVIO PARA VISPY ---
    if t_atual - last_plot_time > PLOT_INTERVAL:
        snap = _pontos_confiaveis()

        if _data_queue.full():
            try:    _data_queue.get_nowait()
            except queue.Empty: pass
        try:    _data_queue.put_nowait(snap)
        except queue.Full: pass

        last_plot_time = t_atual

# ==============================================================================
# LIMPEZA FINAL
# ==============================================================================
_stop_event.set()
_t.join(timeout=2.0)
print("[SEARCH] Concluído.")
