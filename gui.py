import matplotlib

matplotlib.use("TkAgg")  # Linux-friendly, minimal dependencies

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from threading import Thread
import time


class ForceControlGUI:
    def __init__(self, init_force=-5.0):
        self.force_z = init_force
        self._running = True

        self.fig, (self.ax_slider, self.ax_force, self.ax_vel) = plt.subplots(
            3, 1, figsize=(6, 6)
        )
        plt.subplots_adjust(hspace=0.6)

        # Force slider
        self.slider_ax = self.fig.add_axes([0.2, 0.88, 0.6, 0.03])
        self.slider = Slider(
            self.slider_ax, "Z Force (N)", -20.0, 0.0, valinit=self.force_z
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
        self.force_z = val

    def get_force(self):
        return self.force_z

    def update_plot(self, t, fz, qvel):
        self.time_vals.append(t)
        self.force_vals.append(fz)
        self.vel_vals.append(max(abs(v) for v in qvel))

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
