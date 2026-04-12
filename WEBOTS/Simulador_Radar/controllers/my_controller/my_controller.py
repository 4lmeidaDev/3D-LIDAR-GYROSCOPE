import tkinter as tk
from tkinter import ttk
import os
import math
from controller import Robot

# 1. INICIALIZAÇÃO ÚNICA DO ROBÔ
robot = Robot()
timestep = int(robot.getBasicTimeStep())

def executar_script(nome_ficheiro):
    os.environ["MODO_SCAN"] = combo_modo.get()
    os.environ["PARAM_TEMPO"] = entry_tempo.get()
    os.environ["PARAM_FREQ"] = entry_freq.get()
    try:
        fov_rad = str(float(entry_fov.get()) * (math.pi / 180))
        os.environ["PARAM_FOV"] = fov_rad
    except:
        os.environ["PARAM_FOV"] = "1.047"

    print(f"\n>>> A EXECUTAR: {nome_ficheiro}...")
    try:
        # Passamos o robot e o timestep para os scripts
        with open(nome_ficheiro, encoding="utf-8") as f:
            exec(f.read(), globals())
    except Exception as e:
        print(f"ERRO ao correr {nome_ficheiro}: {e}")

# --- Interface Gráfica ---
root = tk.Tk()
root.title("MASTER CONTROL")
root.geometry("320x450")

# Forçar a janela a ficar por cima para não a perderes no Webots
root.attributes('-topmost', True)

ttk.Label(root, text="CONFIGURAÇÕES", font=('Arial', 10, 'bold')).pack(pady=10)
ttk.Label(root, text="Tempo (s):").pack()
entry_tempo = ttk.Entry(root); entry_tempo.insert(0, "40"); entry_tempo.pack()
ttk.Label(root, text="Frequência (Hz):").pack()
entry_freq = ttk.Entry(root); entry_freq.insert(0, "0.5"); entry_freq.pack()
ttk.Label(root, text="FOV Giro (Graus):").pack()
entry_fov = ttk.Entry(root); entry_fov.insert(0, "60"); entry_fov.pack()

ttk.Separator(root, orient='horizontal').pack(fill='x', pady=15)
combo_modo = ttk.Combobox(root, values=["ORIGINAL", "SINOSOIDAL"])
combo_modo.set("ORIGINAL"); combo_modo.pack()
ttk.Button(root, text="INICIAR SCAN", command=lambda: executar_script("scan.py")).pack(pady=5)

ttk.Separator(root, orient='horizontal').pack(fill='x', pady=15)
ttk.Button(root, text="INICIAR SEARCH", command=lambda: executar_script("search.py")).pack(pady=10)

# --- LOOP PRINCIPAL DO WEBOTS (Substitui o mainloop) ---
while robot.step(timestep) != -1:
    root.update() # Mantém a interface Tkinter viva