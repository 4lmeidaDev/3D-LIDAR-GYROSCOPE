import tkinter as tk
from tkinter import ttk
import os
import math
from controller import Robot

# ==============================================================================
# INICIALIZAÇÃO DO ROBÔ
# ==============================================================================
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

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
# ESTADO DE EXECUÇÃO
# ==============================================================================
_a_executar  = False
_action_btns = []   # preenchido depois da GUI — evita repetir config() por botão

def executar_script(nome_ficheiro):
    global _a_executar

    if _a_executar:
        print(f"[CTRL] Já existe um script em execução. Aguarda que termine.")
        return

    os.environ["MODO_SCAN"]   = combo_modo.get()
    os.environ["PARAM_TEMPO"] = entry_tempo.get()
    os.environ["PARAM_FREQ"]  = entry_freq.get()
    try:
        fov_rad = str(float(entry_fov.get()) * (math.pi / 180))
        os.environ["PARAM_FOV"] = fov_rad
    except ValueError:
        os.environ["PARAM_FOV"] = "1.047"

    codigo = _compilar(nome_ficheiro)
    if codigo is None:
        return

    print(f"\n[CTRL] A executar: {nome_ficheiro}...")
    _a_executar = True
    for b in _action_btns:
        b.config(state="disabled")

    try:
        exec(codigo, globals())
    except SystemExit:
        pass
    except Exception:
        import traceback
        print(f"[CTRL] ERRO em '{nome_ficheiro}':")
        traceback.print_exc()
    finally:
        _a_executar = False
        for b in _action_btns:
            b.config(state="normal")
        print(f"[CTRL] '{nome_ficheiro}' terminou.\n")

# ==============================================================================
# INTERFACE GRÁFICA
# ==============================================================================
root = tk.Tk()
root.title("MASTER CONTROL")
root.geometry("320x600")
root.attributes('-topmost', True)
root.resizable(False, False)

style = ttk.Style()
style.configure("Static.TLabelframe.Label", foreground="#2980b9", font=('Arial', 9, 'bold'))
style.configure("Mobil.TLabelframe.Label",  foreground="#27ae60", font=('Arial', 9, 'bold'))

# --- Parâmetros globais ---
ttk.Label(root, text="CONFIGURAÇÕES", font=('Arial', 10, 'bold')).pack(pady=(10, 4))

frame_params = ttk.Frame(root)
frame_params.pack(padx=14, fill='x')

def _campo(parent, label, default):
    ttk.Label(parent, text=label).pack(anchor='w')
    e = ttk.Entry(parent)
    e.insert(0, default)
    e.pack(fill='x', pady=2)
    return e

entry_tempo = _campo(frame_params, "Tempo (s):",        "60")
entry_freq  = _campo(frame_params, "Frequência (Hz):",  "0.3")
entry_fov   = _campo(frame_params, "FOV Giro (Graus):", "60")

# ==============================================================================
# SECÇÃO STATIC
# ==============================================================================
frame_static = ttk.LabelFrame(root, text=" STATIC ", style="Static.TLabelframe", padding=8)
frame_static.pack(padx=14, pady=(10, 4), fill='x')

combo_modo = ttk.Combobox(frame_static, values=["SINOSOIDAL", "ORIGINAL"], state="readonly")
combo_modo.set("SINOSOIDAL")
combo_modo.pack(fill='x', pady=(0, 4))

btn_scan = ttk.Button(
    frame_static, text="▶  SCAN",
    command=lambda: executar_script("scan.py")
)
btn_scan.pack(fill='x', pady=2)

btn_search = ttk.Button(
    frame_static, text="▶  SEARCH",
    command=lambda: executar_script("search.py")
)
btn_search.pack(fill='x', pady=2)

btn_search_dbscan = ttk.Button(
    frame_static, text="▶  SEARCH DBSCAN",
    command=lambda: executar_script("searchDBSCAN.py")
)
btn_search_dbscan.pack(fill='x', pady=2)

ttk.Button(
    frame_static, text="↺  Recarregar Scripts",
    command=lambda: [
        [_invalidar_cache(f) for f in ("scan.py", "search.py", "searchDBSCAN.py", "slam.py")],
        [_compilar(f)        for f in ("scan.py", "search.py", "searchDBSCAN.py", "slam.py")],
        print("[CTRL] Scripts recarregados.")
    ]
).pack(fill='x', pady=(6, 0))

# ==============================================================================
# SECÇÃO MOBIL
# ==============================================================================
frame_mobil = ttk.LabelFrame(root, text=" MOBIL ", style="Mobil.TLabelframe", padding=8)
frame_mobil.pack(padx=14, pady=(6, 4), fill='x')

btn_slam = ttk.Button(
    frame_mobil, text="▶  SLAM",
    command=lambda: executar_script("slam.py")
)
btn_slam.pack(fill='x', pady=2)

# --- Registar todos os botões de ação (desativados durante execução) ---
_action_btns.extend([btn_scan, btn_search, btn_search_dbscan, btn_slam])

# --- Status bar ---
ttk.Separator(root, orient='horizontal').pack(fill='x', pady=8, padx=10)
lbl_status = ttk.Label(root, text="Pronto.", foreground="gray")
lbl_status.pack()

# ==============================================================================
# LOOP PRINCIPAL DO WEBOTS
# ==============================================================================
_tk_counter  = 0
_TK_INTERVAL = 4

while robot.step(timestep) != -1:
    _tk_counter += 1
    if _tk_counter >= _TK_INTERVAL:
        root.update()
        _tk_counter = 0
