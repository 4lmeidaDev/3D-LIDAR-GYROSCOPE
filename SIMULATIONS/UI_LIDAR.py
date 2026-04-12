import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.ticker as ticker
from scipy.spatial import KDTree
import threading
import time

# --- MOTOR MATEMÁTICO ---
def simular_rastreio(fov_deg, sample_rate, scan_freq, gimbal_freq, r=6.0, duracao=0.05):
    num_pontos = int(max(1, sample_rate * duracao))
    t = np.linspace(0, duracao, num_pontos)
    theta = 2 * np.pi * scan_freq * t
    amp = np.radians(fov_deg / 2)
    alpha = amp * np.sin(2 * np.pi * gimbal_freq * t)          
    beta = amp * np.sin(2 * np.pi * gimbal_freq * t + np.pi/2) 
    
    x_l = r * np.cos(theta); y_l = r * np.sin(theta); z_l = np.zeros_like(t)
    y1 = y_l * np.cos(alpha) - z_l * np.sin(alpha)
    z1 = y_l * np.sin(alpha) + z_l * np.cos(alpha)
    x_f = x_l * np.cos(beta) + z1 * np.sin(beta)
    z_f = z1 * np.cos(beta) - x_l * np.sin(beta)
    return x_f, y1, z_f

def calcular_maior_buraco(fov_deg, sample_rate, scan_freq, gimbal_freq, dist, duracao):
    x_f, y1, z_f = simular_rastreio(fov_deg, sample_rate, scan_freq, gimbal_freq, r=dist, duracao=duracao)
    
    pontos_frente = np.vstack((y1[x_f > 0], z_f[x_f > 0])).T
    if len(pontos_frente) == 0: return float('inf')

    arvore_pontos = KDTree(pontos_frente)
    
    limite_fisico = dist * np.tan(np.radians(fov_deg / 2))
    
    grid_y, grid_z = np.meshgrid(
        np.linspace(-limite_fisico, limite_fisico, 40), 
        np.linspace(-limite_fisico, limite_fisico, 40)
    )
    
    pontos_teste = np.vstack((grid_y.ravel(), grid_z.ravel())).T
    
    distancias, _ = arvore_pontos.query(pontos_teste)
    return np.max(distancias)

# --- INTERFACE GRÁFICA ---
class LidarSimulatorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LIDAR 3D Analyzer - Controlo de Tempo (Slow Motion)")
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
        
        ttk.Label(self.control_frame, text="Configuração", font=("Arial", 12, "bold")).pack(pady=5)

        self.params_config = [
            ("fov", "FOV Total (º)", 0, 1, 180, 1.0),
            ("rate", "Sample Rate (Hz)", 5000.0, 500, 12000, 100.0),
            ("scan", "Scan Freq (Hz)", 10.0, 1, 50, 1.0),
            ("gimbal", "Gimbal Freq (Hz)", 2.5, 0, 10, 0.05),
            ("dur", "Tempo Captura (s)", 1, 0.01, 10.0, 0.05), 
            ("dist", "Distância Lidar (m)", 12.0, 1.0, 50.0, 1.0),
            ("grid", "Grelha (m/div)", 2, 0.1, 10.0, 0.5) 
        ]

        self.entries = {}
        self.botoes = {}
        
        for key, label, default, p_min, p_max, step in self.params_config:
            self.create_input_group(key, label, default, p_min, p_max, step)

        stats_group = ttk.LabelFrame(self.control_frame, text="Análise de Deteção", padding=10)
        stats_group.pack(fill=tk.X, pady=(10, 5))
        
        self.lbl_buraco = ttk.Label(stats_group, text="Maior Buraco: A calcular...", font=("Arial", 11, "bold"))
        self.lbl_buraco.pack(anchor=tk.W)
        
        self.lbl_tempo = ttk.Label(stats_group, text="Tempo Físico: 0.000 s", font=("Courier", 10))
        self.lbl_tempo.pack(anchor=tk.W, pady=(5,0))

        tele_group = ttk.LabelFrame(self.control_frame, text="Telemetria da Câmara", padding=10)
        tele_group.pack(fill=tk.X, pady=(10, 5))
        
        self.label_elev = ttk.Label(tele_group, text="Elevação: 30.00º", font=("Courier", 10))
        self.label_elev.pack(anchor=tk.W)
        self.label_azim = ttk.Label(tele_group, text="Azimute: -45.00º", font=("Courier", 10))
        self.label_azim.pack(anchor=tk.W)

        rot_group = ttk.LabelFrame(self.control_frame, text="Navegação 3D", padding=10)
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
        chk_corte = ttk.Checkbutton(
            rot_group, text="Ocultar Metade Traseira", variable=self.var_cortar_metade, command=self.atualizar_grafico 
        )
        chk_corte.pack(fill=tk.X, pady=(10, 0))

        # ---> NOVO PAINEL DE ANIMAÇÃO COM OPÇÕES DE VELOCIDADE <---
        anim_group = ttk.LabelFrame(self.control_frame, text="Controlo de Simulação Temporal", padding=10)
        anim_group.pack(fill=tk.X, pady=(10, 0))

        self.var_animar = tk.BooleanVar(value=False)
        chk_animar = ttk.Checkbutton(
            anim_group, text="VER SIMULAÇÃO ANIMADA", variable=self.var_animar, command=self.on_toggle_animacao 
        )
        chk_animar.pack(anchor=tk.W, pady=(0, 5))

        self.var_speed = tk.DoubleVar(value=1.0)
        rb_real = ttk.Radiobutton(
            anim_group, text="Em Tempo Real (1x)", variable=self.var_speed, value=1.0
        )
        rb_real.pack(anchor=tk.W, padx=15, pady=2)
        
        rb_slow = ttk.Radiobutton(
            anim_group, text="Slow Down (Câmara Lenta 0.1x)", variable=self.var_speed, value=0.1
        )
        rb_slow.pack(anchor=tk.W, padx=15, pady=2)

        ttk.Button(self.control_frame, text="GUARDAR PNG", command=self.save_image).pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # --- PAINEL GRÁFICO ---
        self.plot_frame = ttk.Frame(self.paned, padding=5)
        self.paned.add(self.plot_frame, weight=4)
        
        self.fig = plt.figure(figsize=(8, 8)); self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_proj_type('ortho') 
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_camera_move)
        
        self.scatter_pts = None
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
        ttk.Label(group, text=label, font=("Arial", 9, "bold" if key == "gimbal" else "normal")).pack(anchor=tk.W)
        
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
        self.last_frame_time = time.time()
        self.anim_time = 0.0
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
            fov = float(self.entries["fov"].get())
            rate = float(self.entries["rate"].get())
            scan = float(self.entries["scan"].get())
            dist = float(self.entries["dist"].get())
            dur = float(self.entries["dur"].get())

            frequencias_teste = np.arange(0.5, 5.1, 0.1)
            melhor_freq = 0
            menor_buraco = float('inf')

            for freq in frequencias_teste:
                if scan % freq == 0: continue
                buraco = calcular_maior_buraco(fov, rate, scan, freq, dist, dur)
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
            elev_real = self.ax.elev
            azim_real = self.ax.azim
            
            params = {k: float(self.entries[k].get()) for k in self.entries}
            
            raio_buraco_m = calcular_maior_buraco(
                params["fov"], params["rate"], params["scan"], params["gimbal"], params["dist"], params["dur"]
            )
            
            if raio_buraco_m == float('inf'):
                self.lbl_buraco.config(text="Maior Buraco: Infinito (Sem dados)", foreground="red")
            else:
                diam_cm = (raio_buraco_m * 2) * 100
                cor = "#008000" if diam_cm <= 10.0 else "#cc0000" 
                self.lbl_buraco.config(text=f"Maior Buraco: {diam_cm:.1f} cm", foreground=cor)

            x, y, z = simular_rastreio(
                params["fov"], params["rate"], params["scan"], params["gimbal"], r=params["dist"], duracao=params["dur"]
            )
            
            tempo_decorrido = np.arange(len(x))
            tempo_fisico = np.linspace(0, params["dur"], len(x))

            if self.var_cortar_metade.get():
                e = int(round(elev_real))
                a = int(round(azim_real)) % 360
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
                tempo_fisico = tempo_fisico[mask] 

            self.last_x, self.last_y, self.last_z = x, y, z
            self.last_c = tempo_decorrido
            self.last_t = tempo_fisico 
            self.last_dur = params["dur"]

            self.ax.clear()
            self.ax.scatter([0], [0], [0], color='red', s=100, edgecolors='black', zorder=10)
            
            lim = params["dist"] * 1.2
            self.ax.set_xlim([-lim, lim]); self.ax.set_ylim([-lim, lim]); self.ax.set_zlim([-lim, lim])
            self.ax.set_box_aspect((1, 1, 1)) 
            
            locador = ticker.MultipleLocator(params["grid"]) 
            for eixo in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
                eixo.set_major_locator(locador)
                
            self.ax.view_init(elev=elev_real, azim=azim_real)
            self.ocultar_eixos_invisiveis()
            
            self.scatter_pts = None

            self.anim_time = 0.0 
            self.last_frame_time = time.time()
            self.desenhar_grafico()
            
        except Exception as e:
            pass

    def desenhar_grafico(self):
        if not hasattr(self, 'last_x'): return

        if hasattr(self, 'var_animar') and self.var_animar.get():
            current_time = time.time()
            if not hasattr(self, 'last_frame_time'): 
                self.last_frame_time = current_time
            
            # ---> O SEGREDO DO CONTROLO DE VELOCIDADE <---
            # Calcula quanto tempo o CPU demorou a desenhar o frame anterior
            delta_t = current_time - self.last_frame_time
            
            # Multiplica esse tempo pelo valor do Radio Button (1.0 ou 0.1)
            self.anim_time += delta_t * self.var_speed.get()
            self.last_frame_time = current_time
            
            if self.anim_time >= self.last_dur:
                self.anim_time = 0.0
            
            mask_time = self.last_t <= self.anim_time
            
            x = self.last_x[mask_time]
            y = self.last_y[mask_time]
            z = self.last_z[mask_time]
            c = self.last_c[mask_time]

            self.lbl_tempo.config(text=f"Tempo Físico: {self.anim_time:.3f} s")

            if hasattr(self, 'anim_job') and self.anim_job:
                self.root.after_cancel(self.anim_job)
            
            self.anim_job = self.root.after(15, self.desenhar_grafico)
            
        else:
            if hasattr(self, 'anim_job') and self.anim_job:
                self.root.after_cancel(self.anim_job)
                self.anim_job = None
            
            x, y, z, c = self.last_x, self.last_y, self.last_z, self.last_c
            if hasattr(self, 'last_dur'):
                self.lbl_tempo.config(text=f"Tempo Físico: {self.last_dur:.3f} s")

        if self.scatter_pts is not None:
            self.scatter_pts.remove()
            self.scatter_pts = None
        
        if len(x) > 0:
            self.scatter_pts = self.ax.scatter(x, y, z, c=c, cmap='plasma', s=4, alpha=0.7)
        
        self.canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk(); app = LidarSimulatorUI(root); root.mainloop()