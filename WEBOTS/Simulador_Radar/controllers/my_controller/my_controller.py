import tkinter as tk
from tkinter import ttk
import os
import math
import types
from controller import Robot

# ==============================================================================
# INICIALIZAÇÃO DO ROBÔ
# ==============================================================================
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

# ==============================================================================
# PRÉ-COMPILAÇÃO DOS SCRIPTS
#
# Problema original: exec() lia e compilava o ficheiro do disco cada vez que
# carregavas no botão. Em Python, compilar código é caro.
#
# Solução: compilar uma vez no arranque, reutilizar o bytecode em cada execução.
# Se o ficheiro não existir ainda (primeira vez), compila quando for necessário.
# ==============================================================================
_bytecode_cache = {}   # { "scan.py": <code object>, "search.py": <code object> }

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
    """Remove o ficheiro da cache para forçar recompilação (útil após editar)."""
    _bytecode_cache.pop(nome_ficheiro, None)

# Pré-compilar ambos os scripts no arranque do Webots
for _f in ("scan.py", "search.py", "searchDBSCAN.py"):
    _compilar(_f)

# ==============================================================================
# ESTADO DE EXECUÇÃO
# Impede que dois scripts corram ao mesmo tempo (duplo clique acidental)
# ==============================================================================
_a_executar = False

def executar_script(nome_ficheiro):
    global _a_executar

    if _a_executar:
        print(f"[CTRL] Já existe um script em execução. Aguarda que termine.")
        return

    # Atualizar variáveis de ambiente com os valores da GUI
    os.environ["MODO_SCAN"]   = combo_modo.get()
    os.environ["PARAM_TEMPO"] = entry_tempo.get()
    os.environ["PARAM_FREQ"]  = entry_freq.get()
    try:
        fov_rad = str(float(entry_fov.get()) * (math.pi / 180))
        os.environ["PARAM_FOV"] = fov_rad
    except ValueError:
        os.environ["PARAM_FOV"] = "1.047"

    # Obter bytecode (da cache ou compilar agora)
    codigo = _compilar(nome_ficheiro)
    if codigo is None:
        return

    print(f"\n[CTRL] A executar: {nome_ficheiro}...")
    _a_executar = True
    btn_scan.config(state="disabled")
    btn_search.config(state="disabled")
    btn_search_dbscan.config(state="disabled")

    try:
        exec(codigo, globals())
    except SystemExit:
        pass   # scripts usam raise SystemExit para sair limpo
    except Exception as e:
        import traceback
        print(f"[CTRL] ERRO em '{nome_ficheiro}':")
        traceback.print_exc()
    finally:
        _a_executar = False
        btn_scan.config(state="normal")
        btn_search.config(state="normal")
        btn_search_dbscan.config(state="normal")
        print(f"[CTRL] '{nome_ficheiro}' terminou.\n")

# ==============================================================================
# INTERFACE GRÁFICA
# ==============================================================================
root = tk.Tk()
root.title("MASTER CONTROL")
root.geometry("320x480")
root.attributes('-topmost', True)
root.resizable(False, False)

# --- Cabeçalho ---
ttk.Label(root, text="CONFIGURAÇÕES", font=('Arial', 10, 'bold')).pack(pady=10)

# --- Campos de parâmetros ---
frame_params = ttk.Frame(root)
frame_params.pack(padx=20, fill='x')

def _campo(parent, label, default):
    ttk.Label(parent, text=label).pack(anchor='w')
    e = ttk.Entry(parent)
    e.insert(0, default)
    e.pack(fill='x', pady=2)
    return e

entry_tempo = _campo(frame_params, "Tempo (s):",         "60")
entry_freq  = _campo(frame_params, "Frequência (Hz):",   "0.3")
entry_fov   = _campo(frame_params, "FOV Giro (Graus):",  "60")

# --- Secção SCAN ---
ttk.Separator(root, orient='horizontal').pack(fill='x', pady=10, padx=10)

frame_scan = ttk.Frame(root)
frame_scan.pack(padx=20, fill='x')

combo_modo = ttk.Combobox(frame_scan, values=["SINOSOIDAL", "ORIGINAL"], state="readonly")
combo_modo.set("SINOSOIDAL")
combo_modo.pack(fill='x', pady=2)

btn_scan = ttk.Button(
    frame_scan, text="▶  INICIAR SCAN",
    command=lambda: executar_script("scan.py")
)
btn_scan.pack(fill='x', pady=4)

# Botão para invalidar cache (útil se editares scan.py ou search.py em runtime)
ttk.Button(
    frame_scan, text="↺  Recarregar Scripts",
    command=lambda: [_invalidar_cache("scan.py"), _invalidar_cache("search.py"),
                     [_compilar(f) for f in ("scan.py", "search.py")],
                     print("[CTRL] Scripts recarregados.")]
).pack(fill='x')

# --- Secção SEARCH ---
ttk.Separator(root, orient='horizontal').pack(fill='x', pady=10, padx=10)

frame_search = ttk.Frame(root)
frame_search.pack(padx=20, fill='x')

btn_search = ttk.Button(
    frame_search, text="▶  INICIAR SEARCH",
    command=lambda: executar_script("search.py")
)
btn_search.pack(fill='x', pady=4)

btn_search_dbscan = ttk.Button(
    frame_search, text="▶  SEARCH DBSCAN",
    command=lambda: executar_script("searchDBSCAN.py")
)
btn_search_dbscan.pack(fill='x', pady=4)

# --- Status bar ---
ttk.Separator(root, orient='horizontal').pack(fill='x', pady=10, padx=10)
lbl_status = ttk.Label(root, text="Pronto.", foreground="gray")
lbl_status.pack()

# ==============================================================================
# LOOP PRINCIPAL DO WEBOTS
# Tkinter.update() só é chamado a cada 4 timesteps para não desperdiçar CPU
# quando a simulação está a correr pesado (scan/search ativos).
# ==============================================================================
_tk_counter = 0
_TK_INTERVAL = 4   # chamar root.update() 1 em cada 4 steps

while robot.step(timestep) != -1:
    _tk_counter += 1
    if _tk_counter >= _TK_INTERVAL:
        root.update()
        _tk_counter = 0