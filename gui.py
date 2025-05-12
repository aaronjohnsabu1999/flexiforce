# gui.py
import os
import matplotlib
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from threading import Thread
from threading import Lock


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

        self.fig, (self.ax_slider, self.ax_force, self.ax_vel) = plt.subplots(
            3, 1, figsize=(6, 6)
        )
        plt.subplots_adjust(hspace=0.6)

        # Force slider
        self.slider_ax = self.fig.add_axes([0.2, 0.88, 0.6, 0.03])
        self.slider = Slider(
            self.slider_ax, "Z Force (N)", -20.0, 0.0, valinit=self.force[2]
        )
        self.slider.on_changed(self._on_slider_change)

        self.slider_ax_M = self.fig.add_axes([0.2, 0.82, 0.6, 0.03])
        self.slider_M = Slider(self.slider_ax_M, "Mass M", 0.1, 10.0, valinit=1.0)
        self.slider_M.on_changed(self._on_slider_change)

        self.slider_ax_B = self.fig.add_axes([0.2, 0.76, 0.6, 0.03])
        self.slider_B = Slider(self.slider_ax_B, "Damping B", 0.0, 100.0, valinit=50.0)
        self.slider_B.on_changed(self._on_slider_change)

        self.slider_ax_K = self.fig.add_axes([0.2, 0.70, 0.6, 0.03])
        self.slider_K = Slider(self.slider_ax_K, "Stiffness K", 0.0, 100.0, valinit=0.0)
        self.slider_K.on_changed(self._on_slider_change)

        # Force and velocity plots
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

        # Start the update thread
        self.thread = Thread(target=self._background_update, daemon=True)
        self.thread.start()

        plt.show(block=False)

    def _on_slider_change(self, val):
        self.force[2] = val

    def get_force(self):
        return self.force

    def get_admittance_params(self):
        return self.slider_M.val, self.slider_B.val, self.slider_K.val

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

        # Set window title
        self.fig.canvas.manager.set_window_title(title)

        # Set figure background color
        self.fig.patch.set_facecolor(color)

        # Set window position and size (only if using TkAgg backend)
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
