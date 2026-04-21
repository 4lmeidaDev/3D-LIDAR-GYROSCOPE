import tkinter as tk
from tkinter import ttk
import os
import math
import threading
import time as _time
from controller import Robot

# ==============================================================================
# INICIALIZAÇÃO DO ROBÔ
# ==============================================================================
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

# ==============================================================================
# SETTINGS PARTILHADAS — dict mutável lido pelos scripts em runtime
# Quando um script está em execução, lê daqui em vez de os.environ,
# o que permite o botão "Atualizar" alterar parâmetros em tempo real.
# ==============================================================================
_live_settings = {
    "FREQ":  0.3,
    "FOV_G": math.radians(60.0),
    "TEMPO": 60.0,
    "MODO":  "SINOSOIDAL",
}

# ==============================================================================
# PRÉ-COMPILAÇÃO DOS SCRIPTS
# ==============================================================================
_bytecode_cache = {}

def _compilar(nome_ficheiro):
    """Lê e compila o ficheiro para bytecode. Guarda em cache."""
    if nome_ficheiro not in _bytecode_cache:
        try:
            with open(nome_ficheiro, encoding="utf-8") as f:
                source = f.read()
            _bytecode_cache[nome_ficheiro] = compile(source, nome_ficheiro, "exec")
            print(f"[CTRL] '{nome_ficheiro}' compilado e em cache.")
        except FileNotFoundError:
            print(f"[CTRL] ERRO: '{nome_ficheiro}' não encontrado.")
            return None
        except SyntaxError as e:
            print(f"[CTRL] ERRO de sintaxe em '{nome_ficheiro}': {e}")
            return None
    return _bytecode_cache[nome_ficheiro]

def _invalidar_cache(nome_ficheiro):
    _bytecode_cache.pop(nome_ficheiro, None)

for _f in ("scan.py", "search.py", "searchDBSCAN.py", "slam.py"):
    _compilar(_f)

# ==============================================================================
# CONTROLO DE EXECUÇÃO — não-bloqueante via threading
# ==============================================================================
_script_owns_step = threading.Event()  # Set enquanto um script está no robot.step()
_active_thread    = None
_ctrl_stop        = threading.Event()  # Sinal de paragem partilhado com os scripts
_action_btns      = []


def _atualizar_settings():
    """Lê os campos da GUI, atualiza _live_settings e os.environ."""
    try:
        freq    = float(entry_freq.get())
        fov_deg = float(entry_fov.get())
        tempo   = float(entry_tempo.get())
        modo    = combo_modo.get()
    except ValueError as e:
        print(f"[CTRL] ERRO nas settings: {e}")
        lbl_status.config(text="ERRO nas settings!", foreground="#e74c3c")
        return

    _live_settings["FREQ"]  = freq
    _live_settings["FOV_G"] = fov_deg * math.pi / 180.0
    _live_settings["TEMPO"] = tempo
    _live_settings["MODO"]  = modo

    os.environ["MODO_SCAN"]   = modo
    os.environ["PARAM_TEMPO"] = str(tempo)
    os.environ["PARAM_FREQ"]  = str(freq)
    os.environ["PARAM_FOV"]   = str(fov_deg * math.pi / 180.0)

    print(f"[CTRL] Settings: FREQ={freq} Hz | FOV={fov_deg}° | TEMPO={tempo} s | MODO={modo}")
    lbl_status.config(
        text=f"Atualizado: {freq} Hz | {fov_deg}° | {tempo} s",
        foreground="#27ae60",
    )


def executar_script(nome_ficheiro):
    global _active_thread, _ctrl_stop

    # Parar o script corrente (se existir) antes de lançar um novo
    if _active_thread and _active_thread.is_alive():
        _ctrl_stop.set()
        _active_thread.join(timeout=3.0)

    codigo = _compilar(nome_ficheiro)
    if codigo is None:
        return

    # Garantir que as settings estão sincronizadas antes de lançar
    _atualizar_settings()

    # Novo evento de paragem para este script
    _ctrl_stop = threading.Event()

    # Desativar botões (chamado do main thread pelo handler do botão)
    for b in _action_btns:
        b.config(state="disabled")
    lbl_status.config(text=f"A executar: {nome_ficheiro}…", foreground="#e67e22")

    def _run():
        _script_owns_step.set()
        print(f"\n[CTRL] A executar: {nome_ficheiro}...")
        try:
            exec(codigo, globals())
        except SystemExit:
            pass
        except Exception:
            import traceback
            print(f"[CTRL] ERRO em '{nome_ficheiro}':")
            traceback.print_exc()
        finally:
            _script_owns_step.clear()
            root.after(0, _restaurar_ui)
            print(f"[CTRL] '{nome_ficheiro}' terminou.\n")

    _active_thread = threading.Thread(target=_run, daemon=True)
    _active_thread.start()


def _restaurar_ui():
    for b in _action_btns:
        b.config(state="normal")
    lbl_status.config(text="Pronto.", foreground="gray")


# ==============================================================================
# INTERFACE GRÁFICA
# ==============================================================================
root = tk.Tk()
root.title("MASTER CONTROL")
root.geometry("320x680")
root.attributes('-topmost', True)
root.resizable(False, False)

style = ttk.Style()
style.configure("Cfg.TLabelframe.Label",    foreground="#8e44ad", font=('Arial', 9, 'bold'))
style.configure("Static.TLabelframe.Label", foreground="#2980b9", font=('Arial', 9, 'bold'))
style.configure("Mobil.TLabelframe.Label",  foreground="#27ae60", font=('Arial', 9, 'bold'))


def _campo(parent, label, default):
    ttk.Label(parent, text=label).pack(anchor='w')
    e = ttk.Entry(parent)
    e.insert(0, default)
    e.pack(fill='x', pady=2)
    return e


# --- Secção CONFIGURAÇÕES ---
frame_cfg = ttk.LabelFrame(root, text=" CONFIGURAÇÕES ", style="Cfg.TLabelframe", padding=8)
frame_cfg.pack(padx=14, pady=(10, 4), fill='x')

entry_fov   = _campo(frame_cfg, "Ângulo FOV (Graus):", "60")
entry_freq  = _campo(frame_cfg, "Frequência (Hz):",    "0.3")
entry_tempo = _campo(frame_cfg, "Tempo (s):",          "60")

ttk.Label(frame_cfg, text="Tipo de Movimento:").pack(anchor='w')
combo_modo = ttk.Combobox(frame_cfg, values=["SINOSOIDAL", "ORIGINAL"], state="readonly")
combo_modo.set("SINOSOIDAL")
combo_modo.pack(fill='x', pady=2)

ttk.Button(
    frame_cfg, text="✔  Atualizar",
    command=_atualizar_settings,
).pack(fill='x', pady=(6, 0))

# --- Secção STATIC ---
frame_static = ttk.LabelFrame(root, text=" STATIC ", style="Static.TLabelframe", padding=8)
frame_static.pack(padx=14, pady=(10, 4), fill='x')

btn_scan = ttk.Button(
    frame_static, text="▶  SCAN",
    command=lambda: executar_script("scan.py"),
)
btn_scan.pack(fill='x', pady=2)

btn_search = ttk.Button(
    frame_static, text="▶  SEARCH",
    command=lambda: executar_script("search.py"),
)
btn_search.pack(fill='x', pady=2)

btn_search_dbscan = ttk.Button(
    frame_static, text="▶  SEARCH DBSCAN",
    command=lambda: executar_script("searchDBSCAN.py"),
)
btn_search_dbscan.pack(fill='x', pady=2)

ttk.Button(
    frame_static, text="↺  Recarregar Scripts",
    command=lambda: [
        [_invalidar_cache(f) for f in ("scan.py", "search.py", "searchDBSCAN.py", "slam.py")],
        [_compilar(f)        for f in ("scan.py", "search.py", "searchDBSCAN.py", "slam.py")],
        print("[CTRL] Scripts recarregados."),
    ],
).pack(fill='x', pady=(6, 0))

# --- Secção MOBIL ---
frame_mobil = ttk.LabelFrame(root, text=" MOVING ", style="Mobil.TLabelframe", padding=8)
frame_mobil.pack(padx=14, pady=(10, 4), fill='x')

btn_slam = ttk.Button(
    frame_mobil, text="▶  SLAM",
    command=lambda: executar_script("slam.py"),
)
btn_slam.pack(fill='x', pady=2)

btn_doppler = ttk.Button(
    frame_mobil, text="▶  DOPPLER",
    command=lambda: executar_script("doppler.py"),
)
btn_doppler.pack(fill='x', pady=2)

_action_btns.extend([btn_scan, btn_search, btn_search_dbscan, btn_slam, btn_doppler])

# --- Status bar ---
ttk.Separator(root, orient='horizontal').pack(fill='x', pady=8, padx=10)
lbl_status = ttk.Label(root, text="Pronto.", foreground="gray")
lbl_status.pack()

# ==============================================================================
# LOOP PRINCIPAL DO WEBOTS
# Quando nenhum script está a correr, este loop avança a simulação.
# Quando um script está a correr (noutra thread), apenas atualiza o Tkinter.
# ==============================================================================
while True:
    try:
        if not _script_owns_step.is_set():
            if robot.step(timestep) == -1:
                break
        root.update()
        if _script_owns_step.is_set():
            _time.sleep(0.01)
    except tk.TclError:
        break
