import numpy as np
import os
import math
import threading
import queue
import vispy
from vispy import app, scene
from kinematics import get_3d_points

# ==============================================================================
# CONFIGURAÇÕES — idênticas ao search.py para o filtro encaixar na perfeição
# ==============================================================================
Z_TORRE, L_BRACO = 0.130, 0.032
RAIO_MIN         = 0.05   # Resolução de 5cm — deteção de objetos de 10cm

FREQ        = float(os.getenv("PARAM_FREQ",  0.5))
FOV_G       = float(os.getenv("PARAM_FOV",   1.047))
TEMPO_TOTAL = float(os.getenv("PARAM_TEMPO", 40.0))

# Subsampling: 1 em cada 3 feixes — mais denso que o search (mais detalhe no bg)
SUBSAMPLE     = 3
PLOT_INTERVAL = 1.0 / 15   # Atualizar gráfico a 15 Hz

# ==============================================================================
# BUFFER DE VOXELS — deduplicação incremental, frame a frame
#
# Em vez de guardar todos os pontos brutos numa lista e fazer np.vstack no final
# (que explode em RAM com milhões de pontos), mantemos apenas 1 ponto por célula
# de 5cm durante todo o scan. RAM constante desde o primeiro frame.
# ==============================================================================
MAX_VOXELS = 500_000   # ~sala grande com resolução de 5cm tem normalmente <200k voxels
vox_pts    = np.empty((MAX_VOXELS, 3), dtype=np.float32)
n_voxels   = 0
vox_set    = {}   # { (ix, iy, iz): índice_no_array } — lookup O(1)

def _inserir_frame(pts):
    """
    Recebe pontos de um frame e insere apenas os que caem em células novas.
    Pontos em células já conhecidas são descartados (o 1º ponto por célula vence).
    """
    global n_voxels
    if pts.size == 0:
        return

    v_coords = np.floor(pts / RAIO_MIN).astype(np.int32)
    for i in range(len(v_coords)):
        k = (int(v_coords[i, 0]), int(v_coords[i, 1]), int(v_coords[i, 2]))
        if k not in vox_set and n_voxels < MAX_VOXELS:
            vox_set[k]        = n_voxels
            vox_pts[n_voxels] = pts[i]
            n_voxels         += 1

# ==============================================================================
# THREAD VISPY — visualização em tempo real do background a crescer
# ==============================================================================
_data_queue = queue.Queue(maxsize=2)
_stop_event = threading.Event()

def _vispy_thread():
    vispy.use('PyQt6')

    canvas = scene.SceneCanvas(
        keys='interactive',
        show=True,
        title='SCAN — A construir Background...',
        size=(1024, 768),
        bgcolor='#0d0d1a',
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        fov=60, distance=5.0, elevation=25, azimuth=45,
    )
    scene.visuals.XYZAxis(parent=view.scene)
    scene.visuals.GridLines(color=(0.3, 0.3, 0.3, 0.5), parent=view.scene)

    scatter = scene.visuals.Markers(parent=view.scene)
    scatter.antialias = 0   # mais rápido sem antialiasing

    def _colorir_por_altura(pts):
        """Gradiente de cor por altura Z: azul (baixo) → verde → vermelho (cima)."""
        if len(pts) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        z = pts[:, 2]
        t = (z - z.min()) / (z.max() - z.min() + 1e-9)
        r = np.clip(2.0 * t - 0.5, 0, 1).astype(np.float32)
        g = (np.clip(2.0 * t, 0, 1) * np.clip(2.0 - 2.0 * t, 0, 1)).astype(np.float32)
        b = np.clip(1.0 - 2.0 * t, 0, 1).astype(np.float32)
        a = np.ones(len(pts), dtype=np.float32)
        return np.column_stack((r, g, b, a))

    _camera_ajustada = [False]   # flag para ajustar câmara uma vez com dados reais

    def _update(ev):
        try:
            payload = _data_queue.get_nowait()
        except queue.Empty:
            return
        if payload is None or len(payload) == 0:
            return

        cores = _colorir_por_altura(payload)
        scatter.set_data(
            payload.astype(np.float32),
            edge_color=None,
            face_color=cores,
            size=3,
        )
        canvas.title = f'SCAN — Background: {len(payload):,} voxels visíveis'

        # Ajustar câmara automaticamente na primeira vez que chegam pontos reais
        if not _camera_ajustada[0]:
            centro = payload.mean(axis=0)
            span   = np.linalg.norm(payload.max(axis=0) - payload.min(axis=0))
            view.camera.center   = tuple(centro)
            view.camera.distance = float(span) * 0.8
            _camera_ajustada[0]  = True

    # Guardar referência para evitar garbage collection
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
# LOOP PRINCIPAL DO SCAN
# ==============================================================================
last_plot_time   = 0.0
last_report_time = 0.0
REPORT_INTERVAL  = 5.0   # Imprimir progresso na consola a cada 5 segundos

tempo_ini = robot.getTime()
print(f"[SCAN] A iniciar varrimento sinusoidal")
print(f"[SCAN] Duração={TEMPO_TOTAL:.0f}s | FOV={math.degrees(FOV_G):.0f}° | FREQ={FREQ}Hz")
print(f"[SCAN] Resolução={RAIO_MIN*100:.0f}cm | Subsampling=1/{SUBSAMPLE} feixes")
print("-" * 52)

while robot.step(timestep) != -1:

    if _stop_event.is_set():
        print("[SCAN] Janela fechada — scan interrompido.")
        break

    t = robot.getTime() - tempo_ini

    if t > TEMPO_TOTAL:
        break

    # --- MOVIMENTO SINUSOIDAL ---
    # Alpha e Beta desfasados 90° garantem cobertura esférica completa
    pos_a = FOV_G * math.sin(2 * math.pi * FREQ * t)
    pos_b = FOV_G * math.sin(2 * math.pi * FREQ * t + math.pi / 2)
    motor_a.setPosition(pos_a)
    motor_b.setPosition(pos_b)

    # --- CAPTURA + CINEMÁTICA + DEDUPLICAÇÃO INCREMENTAL ---
    ranges = lidar.getRangeImage()
    pts    = get_3d_points(
        ranges[::SUBSAMPLE], thetas_sub,
        s_a.getValue(), s_b.getValue(),
        L_BRACO, Z_TORRE
    )
    _inserir_frame(pts)

    # --- PROGRESSO NA CONSOLA (a cada 5s) ---
    if t - last_report_time >= REPORT_INTERVAL:
        pct      = (t / TEMPO_TOTAL) * 100
        restante = TEMPO_TOTAL - t
        ciclos   = t * FREQ
        bar_len  = 30
        filled   = int(bar_len * pct / 100)
        bar      = "█" * filled + "░" * (bar_len - filled)
        print(f"[SCAN] [{bar}] {pct:5.1f}% | {t:5.1f}s/{TEMPO_TOTAL:.0f}s "
              f"| {n_voxels:>7,} voxels | {ciclos:.1f} ciclos | restam {restante:.0f}s")
        last_report_time = t

    # --- ENVIO PARA VISPY (não bloqueante) ---
    if t - last_plot_time >= PLOT_INTERVAL:
        if n_voxels > 0:
            # Subsampling visual: nunca enviamos mais de 80k pts para o gráfico
            step     = max(1, n_voxels // 80_000)
            snapshot = vox_pts[:n_voxels:step].copy()
        else:
            snapshot = None

        if _data_queue.full():
            try:    _data_queue.get_nowait()
            except queue.Empty: pass
        try:    _data_queue.put_nowait(snapshot)
        except queue.Full: pass

        last_plot_time = t

# ==============================================================================
# GUARDAR FICHEIRO DE BACKGROUND
# ==============================================================================
print("-" * 52)
if n_voxels > 0:
    final = vox_pts[:n_voxels].copy()
    np.save("scan_otimizado.npy", final)
    print(f"[SCAN] ✓ Guardado:  scan_otimizado.npy")
    print(f"[SCAN] ✓ Voxels únicos:  {n_voxels:,}")
    print(f"[SCAN] ✓ Resolução:      {RAIO_MIN*100:.0f} cm por célula")
    print(f"[SCAN] ✓ Tamanho:        {final.nbytes / 1024:.1f} KB")
    print(f"[SCAN] ✓ Pronto para o search.py")
else:
    print("[SCAN] AVISO: Nenhum ponto capturado — verifica os sensores.")

_stop_event.set()
_t.join(timeout=2.0)
print("[SCAN] Janela fechada.")