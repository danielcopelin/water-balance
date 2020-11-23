# %%

from typing import Union
from numpy.core.numeric import outer
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
            for surface in self.surfaces.values():
                surface.step()
            for storage in self.storages.values():
                storage.step()
            for demand in self.demands.values():
                demand.step()
            for pump in self.pumps.values():
                pump.step()

        return self.mass_balance()

    def mass_balance(self):
        def soil_stores(surface):
            return (
                (
                    (surface.imp_store * surface.imp_frac)
                    + (
                        surface.a1 * surface.store1
                        + surface.a2 * surface.store2
                        + surface.a3 * surface.store3
                    )
                    * (1 - surface.imp_frac)
                )
                * surface.area
                / 1000.0
            )

        # inflows
        rainfall_volume = sum(
            [
                self.rainfall.sum() / 1000.0 * surface.area
                for surface in self.surfaces.values()
            ]
        )
        external_supply = sum([main.supply.sum() for main in self.mains.values()])
        initial_storage = sum(
            [storage.initial_volume for storage in self.storages.values()]
        )
        inflow_volume = rainfall_volume + external_supply + initial_storage

        # outflows
        surface_et = sum([surface.et.sum() for surface in self.surfaces.values()])
        storage_et = sum([storage.et.sum() for storage in self.storages.values()])
        storage_leakage = sum(
            [storage.leakage.sum() for storage in self.storages.values()]
        )
        water_usage = sum(
            [demand.source_supplied.sum() for demand in self.demands.values()]
        )
        end_storage_volumes = sum(
            [storage.volume[-1] for storage in self.storages.values()]
        )
        end_soil_stores = sum(
            [soil_stores(surface)[-1] for surface in self.surfaces.values()]
        )
        end_baseflow_stores = sum(
            [surface.baseflow_store[-1] for surface in self.surfaces.values()]
        )
        end_runoff_stores = sum(
            [surface.runoff_store[-1] for surface in self.surfaces.values()]
        )
        outflow_volume = (
            surface_et
            + storage_et
            + storage_leakage
            + water_usage
            + end_storage_volumes
            + end_soil_stores
            + end_baseflow_stores
            + end_runoff_stores
        )

        # report back
        mass_balance = inflow_volume - outflow_volume
        results = {
            "infow": f"{inflow_volume:.1f}m3",
            "outflow": f"{outflow_volume:.1f}m3",
            "mass balance": f"{mass_balance:.1f}m3",
            "mass error %": f"{mass_balance / inflow_volume * 100.0:.1f}%",
        }
        print(results)
        return results


# %%
if __name__ == "__main__":
    import numpy as np

    silo_data = pd.read_csv(
        "-27.45_153.00.csv", index_col="YYYY-MM-DD", parse_dates=True
    )
    # datetime_index = pd.date_range("2010-07-24", "2020-12-31", freq=timestep)
    rainfall = silo_data["daily_rain"]
    pet = silo_data["evap_pan"]
    timestep = rainfall.index.to_series().diff()[1]
    # demand = (
    #     pd.Series(index=datetime_index, data=np.random.random(size=len(datetime_index)))
    #     * 10000
    # )
    sim = Simulation(timestep=timestep, rainfall=rainfall, pet=pet)

    # surfaces
    # sink = Storage("sink", sim, 0, 100000, 0, 1.3, None)
    pond = Storage("pond", sim, 1000, 5000, 2500, 50, None)
    # field1 = Demand("field1", sim, demand, pond, None, None, 0)
    catchmentA = SurfaceAWBM("catchmentA", sim, 50000.0, 0.5, pond)

    sim.run()

    # %%
    start_date = "2010-10-01"
    end_date = "2011-04-01"

    pd.concat(
        [
            catchmentA.imp_excess,
            catchmentA.baseflow_store,
            catchmentA.runoff_store,
            catchmentA.total_runoff,
        ],
        axis=1,
    )[start_date:end_date].plot()

    # %%
    pd.concat(
        [pond.volume, pond.supplied, pond.overflow, pond.et, pond.leakage,], axis=1
    )[start_date:end_date].plot()

# %%
