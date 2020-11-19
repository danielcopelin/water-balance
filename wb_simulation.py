from typing import Union
import pandas as pd


class Simulation:
    def __init__(self, timestep: pd.Timedelta, rainfall: pd.Series, pet: pd.Series):
        self.timestep = timestep
        self.rainfall = rainfall
        self.pet = pet

        self.surfaces = {}
        self.storages = {}
        self.demands = {}
        self.mains = {}
        self.pumps = {}

        self.length = self.rainfall.size

    def run(self):
        """main loop
        """
        for time in self.rainfall.index:
            self.time = time
            # do all the things
            for surface in self.surfaces:
                surface.step()
            for demand in self.demands:
                demand.step()
            for pump in self.pumps:
                pump.step()
            for storage in self.storages:
                storage.step()
