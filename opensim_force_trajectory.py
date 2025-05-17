# opensim_force_trajectory.py
import pandas as pd
import numpy as np


class OpenSimForceTrajectory:
    def __init__(self, path):
        self.df = self._load_sto(path)
        self.time = self.df["time"].values
        self.force_biceps = (
            self.df["BIClong"].values
            + self.df["BICshort"].values
            + self.df["BRA"].values
        )

    def _load_sto(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        header_idx = next(
            i for i, line in enumerate(lines) if "endheader" in line.lower()
        )
        return pd.read_csv(path, sep="\t", skiprows=header_idx + 1)

    def get_force_at(self, t):
        idx = np.searchsorted(self.time, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.force_biceps) - 1)
        return self.force_biceps[idx]  # Apply negative to match direction
