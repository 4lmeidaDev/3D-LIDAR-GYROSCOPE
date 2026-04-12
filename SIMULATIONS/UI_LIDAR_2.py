import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.ticker as ticker
from scipy.spatial import KDTree
import threading

# --- MOTOR MATEMÁTICO (Azimute e Elevação Puros para os Planos) ---
def simular_rastreio(x_max_deg, x_min_deg, y_max_deg, y_min_deg, sample_rate, scan_freq, gimbal_freq, r=6.0, duracao=0.05):
    num_pontos = int(max(1, sample_rate * duracao))
    t = np.linspace(0, duracao, num_pontos)
    
    # Organiza os limites
    az_max, az_min = max(x_max_deg, x_min_deg), min(x_max_deg, x_min_deg)
    el_max, el_min = max(y_max_deg, y_min_deg), min(y_max_deg, y_min_deg)
    
    # Conversão para radianos
    az_max_rad = np.radians(az_max); az_min_rad = np.radians(az_min)
    el_max_rad = np.radians(el_max); el_min_rad = np.radians(el_min)
    
    # Offsets e Amplitudes para os Planos
    offset_az = (az_max_rad + az_min_rad) / 2.0
    amp_az = (az_max_rad - az_min_rad) / 2.0
    
    offset_el = (el_max_rad + el_min_rad) / 2.0
    amp_el = (el_max_rad - el_min_rad) / 2.0
    
    # Padrão de oscilação
    azimuth = offset_az + amp_az * np.sin(2 * np.pi * scan_freq * t)
    elevation = offset_el + amp_el * np.sin(2 * np.pi * gimbal_freq * t + np.pi/2)
    
    # Mapeamento 3D: X é Profundidade (Frente), Y é Horizontal (Plano XoZ), Z é Vertical (Plano YoZ)
    x_f = r * np.cos(elevation) * np.cos(azimuth)
    y_f = r * np.cos(elevation) * np.sin(azimuth)
    z_f = r * np.sin(elevation)
    
    return x_f, y_f, z_f

def calcular_maior_buraco(x_max, x_min, y_max, y_min, sample_rate, scan_freq, gimbal_freq, dist, duracao):
    x_f, y_f, z_f = simular_rastreio(x_max, x_min, y_max, y_min, sample_rate, scan_freq, gimbal_freq, r=dist, duracao=duracao)
    
    mask = x_f > 0.01
    if not np.any(mask): return float('inf')

    # Projeção na parede plana para garantir matemática precisa em ângulos não centrados no 0
    y_proj = y_f[mask] * (dist / x_f[mask])
    z_proj = z_f[mask] * (dist / x_f[mask])
    
    pontos_frente = np.vstack((y_proj, z_proj)).T
    arvore_pontos = KDTree(pontos_frente)
    
    az_max, az_min = max(x_max, x_min), min(x_max, x_min)
    el_max, el_min = max(y_max, y_min), min(y_max, y_min)
    
    y_max_dist = dist * np.tan(np.radians(az_max))
    y_min_dist = dist * np.tan(np.radians(az_min))
    z_max_dist = dist * np.tan(np.radians(el_max))
    z_min_dist = dist * np.tan(np.radians(el_min))
    
    grid_y, grid_z = np.meshgrid(
        np.linspace(y_min_dist, y_max_dist, 40), 
        np.linspace(z_min_dist, z_max_dist, 40)
    )
    
    pontos_teste = np.vstack((grid_y.ravel(), grid_z.ravel())).T
    distancias, _ = arvore_pontos.query(pontos_teste)
    
    return np.max(distancias)

# --- INTERFACE GRÁFICA ---
class LidarSimulatorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LIDAR 3D Analyzer - Base Limpa + FOV X/Y Customizável")
        self.root.geometry("1400x950")
        
        self.paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        self.left_container = ttk.Frame(self.paned)
        self.paned.add(self.left_container, weight=1)
        
        self.canvas_scroll = tk.Canvas(self.left_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.left_container, orient="vertical", command=self.canvas_scroll.yview)
        
        self.control_frame = ttk.Frame(self.canvas_scroll, padding=15)
        
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas_window = self.canvas_scroll.create_window((0, 0), window=self.control_frame, anchor="nw")
        
        self.control_frame.bind("<Configure>", lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all")))
        self.canvas_scroll.bind("<Configure>", lambda e: self.canvas_scroll.itemconfig(self.canvas_window, width=e.width))
        
        self.canvas_scroll.bind('<Enter>', self._bind_mousewheel)
        self.canvas_scroll.bind('<Leave>', self._unbind_mousewheel)

        ttk.Label(self.control_frame, text="Configuração de Varrimento", font=("Arial", 12, "bold")).pack(pady=5)

        self.params_config = [
            ("x_max", "FOV Plano XoZ (Horizontal) Máx º", 22.5, -90, 90, 1.0),
            ("x_min", "FOV Plano XoZ (Horizontal) Mín º", -22.5, -90, 90, 1.0),
            ("y_max", "FOV Plano YoZ (Vertical) Máx º", 22.5, -90, 90, 1.0),
            ("y_min", "FOV Plano YoZ (Vertical) Mín º", -22.5, -90, 90, 1.0),
            ("rate", "Sample Rate (Hz)", 4500.0, 500, 12000, 100.0),
            ("scan", "Scan Freq (Hz)", 10.0, 1, 50, 1.0),
            ("gimbal", "Gimbal Freq (Hz)", 2.5, 0.1, 10, 0.5),
            ("dur", "Tempo Captura (s)", 0.05, 0.01, 1.0, 0.05),
            ("dist", "Distância Lidar (m)", 6.0, 1.0, 50.0, 1.0),
            ("grid", "Grelha (m/div)", 0.5, 0.1, 10.0, 0.1) 
        ]

        self.entries = {}
        self.botoes = {}
        
        for key, label, default, p_min, p_max, step in self.params_config:
            self.create_input_group(key, label, default, p_min, p_max, step)

        stats_group = ttk.LabelFrame(self.control_frame, text="Análise de Deteção", padding=10)
        stats_group.pack(fill=tk.X, pady=(10, 5))
        
        self.lbl_buraco = ttk.Label(stats_group, text="Maior Buraco: A calcular...", font=("Arial", 11, "bold"))
        self.lbl_buraco.pack(anchor=tk.W)

        tele_group = ttk.LabelFrame(self.control_frame, text="Telemetria da Câmara", padding=10)
        tele_group.pack(fill=tk.X, pady=(10, 5))
        
        self.label_elev = ttk.Label(tele_group, text="Elevação: 30.00º", font=("Courier", 10))
        self.label_elev.pack(anchor=tk.W)
        self.label_azim = ttk.Label(tele_group, text="Azimute: -45.00º", font=("Courier", 10))
        self.label_azim.pack(anchor=tk.W)

        rot_group = ttk.LabelFrame(self.control_frame, text="Navegação e Vistas", padding=10)
        rot_group.pack(fill=tk.X, pady=10)
        
        arrow_frame = ttk.Frame(rot_group); arrow_frame.pack(pady=5)
        ttk.Button(arrow_frame, text="▲", width=4, command=lambda: self.rotate_view(10, 0)).grid(row=0, column=1)
        ttk.Button(arrow_frame, text="◀", width=4, command=lambda: self.rotate_view(0, -10)).grid(row=1, column=0)
        ttk.Button(arrow_frame, text="▶", width=4, command=lambda: self.rotate_view(0, 10)).grid(row=1, column=2)
        ttk.Button(arrow_frame, text="▼", width=4, command=lambda: self.rotate_view(-10, 0)).grid(row=2, column=1)

        ttk.Button(rot_group, text="REPOR VISTA ORIGINAL", command=self.reset_view_to_default).pack(fill=tk.X, pady=(10, 2))
        
        view_f = ttk.Frame(rot_group); view_f.pack(fill=tk.X, pady=5)
        ttk.Button(view_f, text="Topo", width=8, command=lambda: self.set_view(90, -90)).pack(side=tk.LEFT, expand=True)
        ttk.Button(view_f, text="Frente", width=8, command=lambda: self.set_view(0, -90)).pack(side=tk.LEFT, expand=True)
        ttk.Button(view_f, text="Lado", width=8, command=lambda: self.set_view(0, 0)).pack(side=tk.LEFT, expand=True)

        self.var_cortar_metade = tk.BooleanVar(value=False)
        chk_corte = ttk.Checkbutton(rot_group, text="Ocultar Metade Traseira", variable=self.var_cortar_metade, command=self.atualizar_grafico)
        chk_corte.pack(fill=tk.X, pady=(10, 0))

        self.var_animar = tk.BooleanVar(value=False)
        chk_animar = ttk.Checkbutton(rot_group, text="VER SIMULAÇÃO DE PONTOS EM TEMPO REAL", variable=self.var_animar, command=self.on_toggle_animacao)
        chk_animar.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(self.control_frame, text="GUARDAR PNG", command=self.save_image).pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.plot_frame = ttk.Frame(self.paned, padding=5)
        self.paned.add(self.plot_frame, weight=4)
        
        self.fig = plt.figure(figsize=(8, 8)); self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_proj_type('ortho') 
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_camera_move)
        
        self.request_update()

    def _bind_mousewheel(self, event):
        self.canvas_scroll.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas_scroll.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas_scroll.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas_scroll.unbind_all("<MouseWheel>")
        self.canvas_scroll.unbind_all("<Button-4>")
        self.canvas_scroll.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.canvas_scroll.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas_scroll.yview_scroll(1, "units")

    def create_input_group(self, key, label, default, p_min, p_max, step):
        group = ttk.Frame(self.control_frame)
        group.pack(fill=tk.X, pady=5)
        
        is_angle = key in ["x_max", "x_min", "y_max", "y_min"]
        ttk.Label(group, text=label, font=("Arial", 9, "bold" if is_angle or key == "gimbal" else "normal")).pack(anchor=tk.W)
        
        f = ttk.Frame(group)
        f.pack(fill=tk.X)
        
        btn_minus = ttk.Button(f, text="-", width=3, command=lambda k=key, s=step, m=p_min, M=p_max: self.increment_val(k, -s, m, M))
        btn_minus.pack(side=tk.LEFT)
        
        ent = ttk.Entry(f, width=10, justify=tk.CENTER)
        ent.insert(0, str(default))
        ent.pack(side=tk.LEFT, padx=5)
        ent.bind("<Return>", lambda event: self.request_update())
        
        btn_plus = ttk.Button(f, text="+", width=3, command=lambda k=key, s=step, m=p_min, M=p_max: self.increment_val(k, s, m, M))
        btn_plus.pack(side=tk.LEFT)
        
        self.entries[key] = ent
        self.botoes[key] = (btn_minus, btn_plus)

        if key == "gimbal":
            self.var_otimizar_gimbal = tk.BooleanVar(value=False)
            chk_otimizar = ttk.Checkbutton(
                group, text="Usar valor otimizado", variable=self.var_otimizar_gimbal, command=self.on_toggle_otimizar
            )
            chk_otimizar.pack(anchor=tk.W, pady=(2, 0))

    def on_toggle_otimizar(self):
        if self.var_otimizar_gimbal.get():
            self.botoes["gimbal"][0].config(state=tk.DISABLED)
            self.botoes["gimbal"][1].config(state=tk.DISABLED)
            self.request_update()
        else:
            self.entries["gimbal"].config(state=tk.NORMAL)
            self.botoes["gimbal"][0].config(state=tk.NORMAL)
            self.botoes["gimbal"][1].config(state=tk.NORMAL)

    def increment_val(self, key, delta, p_min, p_max):
        try:
            current = float(self.entries[key].get())
            new_val = max(p_min, min(p_max, current + delta))
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, f"{new_val:.2f}")
            self.request_update()
        except ValueError:
            messagebox.showerror("Erro", "Valor inválido na caixa de texto.")

    def on_toggle_animacao(self):
        self.anim_index = 0
        self.desenhar_grafico()

    def request_update(self):
        if hasattr(self, 'var_otimizar_gimbal') and self.var_otimizar_gimbal.get():
            self.entries["gimbal"].config(state=tk.NORMAL)
            self.entries["gimbal"].delete(0, tk.END)
            self.entries["gimbal"].insert(0, "Calc...")
            self.entries["gimbal"].config(state=tk.DISABLED)
            threading.Thread(target=self.otimizar_gimbal_bg, daemon=True).start()
        else:
            self.atualizar_grafico()

    def otimizar_gimbal_bg(self):
        try:
            x_max = float(self.entries["x_max"].get()); x_min = float(self.entries["x_min"].get())
            y_max = float(self.entries["y_max"].get()); y_min = float(self.entries["y_min"].get())
            rate = float(self.entries["rate"].get())
            scan = float(self.entries["scan"].get())
            dist = float(self.entries["dist"].get())
            dur = float(self.entries["dur"].get())

            frequencias_teste = np.arange(0.5, 5.1, 0.1)
            melhor_freq = 0
            menor_buraco = float('inf')

            for freq in frequencias_teste:
                if scan % freq == 0: continue
                buraco = calcular_maior_buraco(x_max, x_min, y_max, y_min, rate, scan, freq, dist, dur)
                if buraco < menor_buraco:
                    menor_buraco = buraco
                    melhor_freq = freq

            self.root.after(0, self.aplicar_otimizacao_bg, melhor_freq)
        except Exception:
            pass 

    def aplicar_otimizacao_bg(self, melhor_freq):
        self.entries["gimbal"].config(state=tk.NORMAL)
        self.entries["gimbal"].delete(0, tk.END)
        self.entries["gimbal"].insert(0, f"{melhor_freq:.2f}")
        self.entries["gimbal"].config(state=tk.DISABLED)
        self.atualizar_grafico()

    def ocultar_eixos_invisiveis(self):
        e = int(round(self.ax.elev))
        a = int(round(self.ax.azim)) % 360 
        
        vis_x, vis_y, vis_z = True, True, True
        if e == 90 or e == -90: vis_z = False
        elif e == 0:
            if a == 90 or a == 270: vis_y = False
            elif a == 0 or a == 180: vis_x = False

        self.ax.tick_params(axis='x', labelcolor='black' if vis_x else 'none')
        self.ax.tick_params(axis='y', labelcolor='black' if vis_y else 'none')
        self.ax.tick_params(axis='z', labelcolor='black' if vis_z else 'none')
        
        self.ax.set_xlabel("X (m)" if vis_x else "")
        self.ax.set_ylabel("Y (m)" if vis_y else "")
        self.ax.set_zlabel("Z (m)" if vis_z else "")

    def on_camera_move(self, event):
        elev, azim = self.ax.elev, self.ax.azim
        self.label_elev.config(text=f"Elevação: {elev:.2f}º")
        self.label_azim.config(text=f"Azimute: {azim:.2f}º")
        self.ocultar_eixos_invisiveis()

    def rotate_view(self, elev_delta, azim_delta):
        elev_atual = self.ax.elev
        azim_atual = self.ax.azim
        self.ax.view_init(elev=elev_atual + elev_delta, azim=azim_atual + azim_delta)
        self.on_camera_move(None) 
        if self.var_cortar_metade.get(): self.atualizar_grafico()
        else: self.canvas.draw_idle()

    def set_view(self, elev, azim):
        self.ax.view_init(elev=elev, azim=azim)
        self.on_camera_move(None)
        self.atualizar_grafico()

    def reset_view_to_default(self):
        self.var_cortar_metade.set(False) 
        self.set_view(30, -45)

    def save_image(self):
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path: self.fig.savefig(path, dpi=300); messagebox.showinfo("Sucesso", "Imagem guardada!")

    def atualizar_grafico(self):
        try:
            params = {k: float(self.entries[k].get()) for k in self.entries}
            p = params
            
            raio_buraco_m = calcular_maior_buraco(
                p["x_max"], p["x_min"], p["y_max"], p["y_min"], p["rate"], p["scan"], p["gimbal"], p["dist"], p["dur"]
            )
            
            if raio_buraco_m == float('inf'):
                self.lbl_buraco.config(text="Maior Buraco: Infinito (Sem dados)", foreground="red")
            else:
                diam_cm = (raio_buraco_m * 2) * 100
                cor = "#008000" if diam_cm <= 10.0 else "#cc0000" 
                self.lbl_buraco.config(text=f"Maior Buraco: {diam_cm:.1f} cm", foreground=cor)

            x, y, z = simular_rastreio(
                p["x_max"], p["x_min"], p["y_max"], p["y_min"], p["rate"], p["scan"], p["gimbal"], r=p["dist"], duracao=p["dur"]
            )
            tempo_decorrido = np.arange(len(x))

            if self.var_cortar_metade.get():
                e = int(round(self.ax.elev))
                a = int(round(self.ax.azim)) % 360
                mask = np.ones_like(x, dtype=bool) 
                
                if e == 90: mask = z >= 0
                elif e == -90: mask = z <= 0
                elif e == 0:
                    if a == 270 or a == -90: mask = y <= 0 
                    elif a == 90: mask = y >= 0            
                    elif a == 0: mask = x >= 0             
                    elif a == 180: mask = x <= 0           
                
                x = x[mask]; y = y[mask]; z = z[mask]
                tempo_decorrido = tempo_decorrido[mask]

            self.last_x, self.last_y, self.last_z = x, y, z
            self.last_c = tempo_decorrido
            
            # Caixa limitadora (Bounding Box) dinâmica para os eixos não encolherem/esticarem
            dist = p["dist"]
            az_max, az_min = max(p["x_max"], p["x_min"]), min(p["x_max"], p["x_min"])
            el_max, el_min = max(p["y_max"], p["y_min"]), min(p["y_max"], p["y_min"])
            
            y_proj = max(abs(dist * np.tan(np.radians(az_max))), abs(dist * np.tan(np.radians(az_min))))
            z_proj = max(abs(dist * np.tan(np.radians(el_max))), abs(dist * np.tan(np.radians(el_min))))
            
            max_dev = max(dist, y_proj, z_proj)
            self.last_lim_box = max_dev * 1.1 
            self.last_grid = p["grid"]

            self.anim_index = 0 
            self.desenhar_grafico()
            
        except Exception as e:
            pass

    def desenhar_grafico(self):
        if not hasattr(self, 'last_x'): return

        x, y, z, c = self.last_x, self.last_y, self.last_z, self.last_c
        lim_box = self.last_lim_box
        grid = self.last_grid

        elev_real = self.ax.elev
        azim_real = self.ax.azim

        if hasattr(self, 'var_animar') and self.var_animar.get():
            if not hasattr(self, 'anim_index'): self.anim_index = 0
            
            self.anim_step = max(1, len(self.last_x) // 40)
            self.anim_index += self.anim_step

            x = self.last_x[:self.anim_index]
            y = self.last_y[:self.anim_index]
            z = self.last_z[:self.anim_index]
            c = self.last_c[:self.anim_index]

            if self.anim_index > len(self.last_x):
                self.anim_index = 0

            if hasattr(self, 'anim_job') and self.anim_job:
                self.root.after_cancel(self.anim_job)
            self.anim_job = self.root.after(40, self.desenhar_grafico)
        else:
            if hasattr(self, 'anim_job') and self.anim_job:
                self.root.after_cancel(self.anim_job)
                self.anim_job = None

        for artist in list(self.ax.collections) + list(self.ax.texts):
            artist.remove()
        
        self.ax.scatter([0], [0], [0], color='red', s=100, edgecolors='black', zorder=10)
        
        if len(x) > 0:
            self.ax.scatter(x, y, z, c=c, cmap='plasma', s=12, alpha=0.7)
        
        self.ax.set_xlim([-lim_box, lim_box]); self.ax.set_ylim([-lim_box, lim_box]); self.ax.set_zlim([-lim_box, lim_box])
        self.ax.set_box_aspect((1, 1, 1)) 
        
        locador = ticker.MultipleLocator(grid) 
        for eixo in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            eixo.set_major_locator(locador)
        
        self.ax.view_init(elev=elev_real, azim=azim_real)
        self.on_camera_move(None) 
        self.canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk(); app = LidarSimulatorUI(root); root.mainloop()