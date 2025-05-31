# gui.py
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from threading import Thread, Lock
import logging


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
        self.enable_viewer = kwargs.get("enable_viewer", True)
        self.verbose = kwargs.get("verbose", False)
        configure_matplotlib_backend(verbose=self.verbose)

        self.data_lock = Lock()
        self._running = True
        self.controller = kwargs.get("controller", None)

        self.fig, (self.ax_slider, self.ax_force, self.ax_vel) = plt.subplots(
            3, 1, figsize=(6, 6)
        )
        plt.subplots_adjust(hspace=0.6)

        self.sliders = {}
        self.slider_last_values = {}
        sliders_config = kwargs.get("sliders", {})
        slider_count = 0
        for key, cfg in sliders_config.items():
            label = cfg.get("label", key)
            min_val, max_val = cfg["range"]
            default = cfg.get("default", (min_val + max_val) / 2)
            allowed = cfg.get("controller", "all")
            if allowed == "all" or (
                self.controller and self.controller.type in allowed
            ):
                y_pos = 0.94 - slider_count * 0.06
                ax = self.fig.add_axes([0.2, y_pos, 0.6, 0.03])
                slider = Slider(ax, label, min_val, max_val, valinit=default)
                self.sliders[key] = slider
                setattr(
                    self, key, default
                )  # Initialize the variable with the default value
                self.slider_last_values[key] = default

                # Create a callback function to update the attribute and log the change
                def make_callback(attr_name):
                    def callback(val):
                        setattr(self, attr_name, val)
                        logging.info(f"Slider '{attr_name}' changed to {val:.2f}")

                    return callback

                slider.on_changed(make_callback(key))
                logging.info(f"Slider '{key}' initialized to {default:.2f}")
                slider_count += 1

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

        self._schedule_update()
        # if self.enable_viewer:
        plt.show()

    def is_admittance_controller(self):
        return self.controller and self.controller.type == "AC"

    def is_parallel_force_motion_controller(self):
        return self.controller and self.controller.type == "PFC"

    def get_force(self):
        return np.array(
            [
                getattr(self, "force_x", 0.0),
                getattr(self, "force_y", 0.0),
                getattr(self, "force_z", -5.0),
            ]
        )

    def get_target_mvc(self):
        return getattr(self, "target_mvc", 0.0)

    def get_max_force(self):
        return getattr(self, "max_force", 400.0)

    def get_admittance_params(self):
        assert (
            self.is_admittance_controller()
        ), "Controller is not an AdmittanceController"
        return (
            getattr(self, "mass", 1.0),
            getattr(self, "damping", 10.0),
            getattr(self, "stiffness", 10.0),
        )

    def update_plot(self, t, fz, qvel):
        with self.data_lock:
            self.time_vals.append(t)
            self.force_vals.append(fz)
            self.vel_vals.append(max(abs(v) for v in qvel))

    def log_slider_changes(self):
        for key, slider in self.sliders.items():
            current_val = getattr(self, key)
            last_val = self.slider_last_values.get(key)

            if not np.isclose(current_val, last_val):
                logging.info(f"  Slider '{key}' changed to {current_val:.3f}")
                self.slider_last_values[key] = current_val

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

    def _schedule_update(self):
        self._background_update()
        self.fig.canvas.manager.window.after(50, self._schedule_update)

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

            # if self.enable_viewer:
            self.fig.canvas.draw_idle()
            time.sleep(0.05)

    def stop(self):
        self._running = False
        plt.close(self.fig)
