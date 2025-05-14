# gui.py
import os
import matplotlib
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from threading import Thread, Lock


def configure_matplotlib_backend(verbose=False):
    if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
        try:
            if verbose:
                print("TkAgg available, using TkAgg (GUI)")
            matplotlib.use("TkAgg")
        except ImportError:
            if verbose:
                print("⚠️ TkAgg not available, using Agg (no GUI)")
            matplotlib.use("Agg")
    else:
        if verbose:
            print("⚠️ No DISPLAY detected, using Agg (no GUI)")
        matplotlib.use("Agg")


class ForceControlGUI:
    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.get("verbose", False)
        configure_matplotlib_backend(verbose=self.verbose)

        self.data_lock = Lock()
        self.force = kwargs.get("init_force", np.array([0.0, 0.0, -5.0]))
        self._running = True
        self.target_mvc = kwargs.get("target_mvc", 0.0)
        self.mass = kwargs.get("mass", 1.0)
        self.damping = kwargs.get("damping", 50.0)
        self.stiffness = kwargs.get("stiffness", 50.0)

        slider_ranges = kwargs.get("slider_ranges", {})

        def get_range_and_default(key, fallback):
            values = slider_ranges.get(key, fallback)
            if len(values) == 3:
                return values[:2], values[2]
            return values, (values[0] + values[1]) / 2

        self.fig, (self.ax_slider, self.ax_force, self.ax_vel) = plt.subplots(
            3, 1, figsize=(6, 6)
        )
        plt.subplots_adjust(hspace=0.6)

        slider_specs = []
        for key, label, y_pos in [
            ("mvc", "%MVC (Target Effort)", 0.94),
            ("z_force", "Z Force (N)", 0.88),
            ("mass", "Mass M", 0.82),
            ("damping", "Damping B", 0.76),
            ("stiffness", "Stiffness K", 0.70),
        ]:
            range_vals, default_val = get_range_and_default(key, [0.0, 100.0, 0.0])
            slider_specs.append(
                {
                    "name": f"slider_{key.replace('_', '')}",
                    "label": label,
                    "range": range_vals,
                    "valinit": default_val,
                    "y_pos": y_pos,
                }
            )

        for spec in slider_specs:
            ax = self.fig.add_axes([0.2, spec["y_pos"], 0.6, 0.03])
            slider = Slider(
                ax,
                spec["label"],
                spec["range"][0],
                spec["range"][1],
                valinit=spec["valinit"],
            )
            slider.on_changed(self._on_slider_change)
            setattr(self, spec["name"], slider)

        self.force_vals, self.vel_vals, self.time_vals = [], [], []
        (self.force_line,) = self.ax_force.plot([], [], label="Z Force [N]")
        (self.vel_line,) = self.ax_vel.plot(
            [], [], label="Max Joint Vel [rad/s]", color="orange"
        )

        self.ax_force.set_ylabel("Z Force [N]")
        self.ax_force.legend()
        self.ax_force.grid(True)

        self.ax_vel.set_ylabel("Joint Vel [rad/s]")
        self.ax_vel.set_xlabel("Time [s]")
        self.ax_vel.legend()
        self.ax_vel.grid(True)

        self.thread = Thread(target=self._background_update, daemon=True)
        self.thread.start()

        plt.show(block=False)

    def _on_slider_change(self, val):
        self.force[2] = self.slider_Fz.val
        self.mass = self.slider_M.val
        self.damping = self.slider_B.val
        self.stiffness = self.slider_K.val
        self.target_mvc = self.slider_mvc.val

    def get_force(self):
        return self.force

    def get_target_mvc(self):
        return self.target_mvc

    def get_admittance_params(self):
        return self.mass, self.damping, self.stiffness

    def update_plot(self, t, fz, qvel):
        with self.data_lock:
            self.time_vals.append(t)
            self.force_vals.append(fz)
            self.vel_vals.append(max(abs(v) for v in qvel))

    def set_window(self, *args, **kwargs):
        title = kwargs.get("title", "Force Control GUI")
        size = kwargs.get("size", (800, 600))
        pos = kwargs.get("pos", (100, 100))
        color = kwargs.get("color", (0.1, 0.1, 0.1))

        self.fig.canvas.manager.set_window_title(title)
        self.fig.patch.set_facecolor(color)

        try:
            backend_window = self.fig.canvas.manager.window
            backend_window.wm_geometry(f"{size[0]}x{size[1]}+{pos[0]}+{pos[1]}")
        except Exception as e:
            print(f"[Warning] Unable to set window geometry: {e}")

    def _background_update(self):
        while self._running:
            with self.data_lock:
                if self.time_vals:
                    self.force_line.set_data(self.time_vals, self.force_vals)
                    self.vel_line.set_data(self.time_vals, self.vel_vals)

                    self.ax_force.relim()
                    self.ax_force.autoscale_view()
                    self.ax_vel.relim()
                    self.ax_vel.autoscale_view()

            self.fig.canvas.draw_idle()
            time.sleep(0.05)

    def stop(self):
        self._running = False
        plt.close(self.fig)
