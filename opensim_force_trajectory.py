# opensim_force_trajectory.py
import os
import pandas as pd
import numpy as np
import logging
import opensim.simulation as osim


class OpenSimForceTrajectory:
    def __init__(self, model_path, sto_path=None, scale=-1.0):
        """
        model_path: path to arm26.osim
        sto_path: optional path to precomputed .sto file; if None, will auto-generate
        """
        self.scale = scale

        if sto_path is None or not os.path.exists(sto_path):
            if osim is None:
                raise RuntimeError("OpenSim API not available and no .sto file provided.")
            logging.warning("🔧 No .sto file found — running OpenSim to generate force trajectory...")
            sto_path = self._generate_sto_from_opensim(model_path)
            logging.info(f"✅ STO file generated at {sto_path}")

        self.df = self._load_sto(sto_path)
        self.time = self.df["time"].values
        self.force_biceps = (
            self.df["BIClong"].values
            + self.df["BICshort"].values
            + self.df["BRA"].values
        ) * self.scale

    def _generate_sto_from_opensim(self, model_path):
        model = osim.Model(model_path)
        state = model.initSystem()

        # Add ForceReporter to collect muscle forces
        force_reporter = osim.ForceReporter()
        force_reporter.setName("ForceReporter")
        model.addAnalysis(force_reporter)

        # Setup a very simple forward simulation (static pose, 1s sim, small dt)
        integrator = osim.RungeKuttaMersonIntegrator(model.getSystem())
        manager = osim.Manager(model, integrator)
        state.setTime(0)
        manager.initialize(state)
        manager.integrate(1.0)

        # Export forces
        output_dir = "OpenSIM_utils/Generated"
        os.makedirs(output_dir, exist_ok=True)
        force_reporter.printResults("forces", output_dir)

        # Return path to generated .sto file
        sto_path = os.path.join(output_dir, "forces_ForceReporter_forces.sto")
        return sto_path

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
        return self.force_biceps[idx]

    def get_full_trajectory(self):
        return self.time, self.force_biceps
