# %%

from typing import Union
import pandas as pd
from wb_nodes import *
from tqdm import tqdm


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
        # skip first timestep
        for time in tqdm(self.rainfall.index[1:]):
            self.time = time
            self.time_previous = self.rainfall.index[
                self.rainfall.index.get_loc(time) - 1
            ]
            # do all the things
            for storage in self.storages.values():
                storage.step()
            for surface in self.surfaces.values():
                surface.step()
            for demand in self.demands.values():
                demand.step()
            for pump in self.pumps.values():
                pump.step()


if __name__ == "__main__":
    import numpy as np

    timestep = pd.Timedelta("1 day")
    datetime_index = pd.date_range("2010-07-24", "2020-12-31", freq=timestep)
    rainfall = pd.Series(
        index=datetime_index, data=np.random.random(size=len(datetime_index))
    )
    pet = pd.Series(
        index=datetime_index, data=np.random.random(size=len(datetime_index))
    )
    demand = (
        pd.Series(index=datetime_index, data=np.random.random(size=len(datetime_index)))
        * 10000
    )
    sim = Simulation(timestep=timestep, rainfall=rainfall, pet=pet)

    # surfaces
    # sink = Storage("sink", sim, 0, 100000, 0, 1.3, None)
    pond = Storage("pond", sim, 0, 100000, 500, 200, None)
    # field1 = Demand("field1", sim, demand, pond, None, None, 0)
    catchmentA = SurfaceAWBM("catchmentA", sim, 150001.0, 0.2, pond)

    print(sim.surfaces)
    print(sim.storages)
    print(sim.demands)

    sim.run()

    # %%
    pond.volume.plot()
    # %%
    pond.leakage.plot()
    # %%
    catchmentA.total_runoff.plot()

# %%
