import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from threading import Thread

matplotlib.use("TkAgg")


class ForceControlGUI:
    def __init__(self, *args, **kwargs):
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

    def _on_slider_change(self, val):
        self.force[2] = val

    def get_force(self):
        return self.force

    def update_plot(self, t, fz, qvel):
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
