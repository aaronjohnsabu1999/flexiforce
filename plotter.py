import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class RealtimePlotter:
    def __init__(self, max_len=500):
        self.force_log = deque(maxlen=max_len)
        self.qvel_log = deque(maxlen=max_len)
        self.time_log = deque(maxlen=max_len)

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.force_line, = self.ax1.plot([], [], label='Z Force')
        self.qvel_line, = self.ax2.plot([], [], label='Max Joint Velocity')

        self.ax1.set_ylabel("Force [N]")
        self.ax2.set_ylabel("Joint Velocity [rad/s]")
        self.ax2.set_xlabel("Time [s]")

    def update(self, t, fz, qvel):
        self.time_log.append(t)
        self.force_log.append(fz)
        self.qvel_log.append(np.max(np.abs(qvel)))

        self.force_line.set_data(self.time_log, self.force_log)
        self.qvel_line.set_data(self.time_log, self.qvel_log)

        self.ax1.relim(); self.ax1.autoscale_view()
        self.ax2.relim(); self.ax2.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

