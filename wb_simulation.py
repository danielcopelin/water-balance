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

        # TODO check for name clashes and reject duplicates by subclassing dict
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

        self.mass_balance()

    def mass_balance(self):
        # TODO automatically register inflows and outflows within node objects
        # so that mass balance accounting is auto updating
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
            [surface.rainfall_volume.sum(0) for surface in self.surfaces.values()]
        )
        storage_direct_rainfall = sum(
            [storage.direct_rainfall.sum() for storage in self.storages.values()]
        )
        external_supply = sum([main.supply.sum() for main in self.mains.values()])
        initial_storage = sum(
            [storage.initial_volume for storage in self.storages.values()]
        )
        wastewater_streams = sum(
            [
                demand.wastewater_stream.sum()
                for demand in self.demands.values()
                if demand.wastewater_sink is not None
            ]
        )
        inflow_volume = (
            rainfall_volume
            + storage_direct_rainfall
            + external_supply
            + initial_storage
            + wastewater_streams
        )

        # outflows
        surface_et = sum([surface.et.sum() for surface in self.surfaces.values()])
        storage_et = sum([storage.et.sum() for storage in self.storages.values()])
        storage_leakage = sum(
            [storage.leakage.sum() for storage in self.storages.values()]
        )
        water_usage = sum(
            [
                demand.source_supplied.sum() + demand.backup_supplied.sum()
                for demand in self.demands.values()
            ]
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
        nonrouted_total_runoff = sum(
            [
                surface.total_runoff.sum()
                for surface in self.surfaces.values()
                if surface.outlet is None
            ]
        )
        nonrouted_storage_overflow = sum(
            [
                storage.overflow.sum()
                for storage in self.storages.values()
                if storage.outlet is None
            ]
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
            + nonrouted_total_runoff
            + nonrouted_storage_overflow
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

    def collate_results(self):
        results = [self.rainfall, self.pet]
        for surface in self.surfaces.values():
            results = results + surface.results
        for storage in self.storages.values():
            results = results + storage.results
        for demand in self.demands.values():
            results = results + demand.results + [demand.demand]
        for main in self.mains.values():
            results = results + main.results
        for pump in self.pumps.values():
            pump = results + pump.results

        results_df = pd.concat(results, axis=1)

        return results_df


# %%
if __name__ == "__main__":
    import numpy as np

    silo_data = pd.read_csv(
        "-27.45_153.00.csv", index_col="YYYY-MM-DD", parse_dates=True
    )
    rainfall = silo_data["daily_rain"]
    pet = silo_data["evap_pan"]
    timestep = rainfall.index.to_series().diff()[1]
    demand = (
        pd.Series(
            name="demand",
            index=rainfall.index,
            data=np.random.random(size=len(rainfall.index)),
        )
        * 1000
    )
    pump_schedule = pd.Series(name="pump_schedule", index=rainfall.index, data=200.0,)

    # create simulation
    sim = Simulation(timestep=timestep, rainfall=rainfall, pet=pet)

    # nodes
    wastewater_pond = Storage("wastewater_pond", sim, 0, 9999999999, 0, 0, None)
    pondA = Storage("pondA", sim, 1000, 5000, 2500, 50, None)
    pondB = Storage("pondB", sim, 2500, 7500, 6250, 0, None)
    pump = Pump("pump", sim, pondA, pondB, pump_schedule)
    mains = Mains("mains", sim)
    fieldA = Demand("field1", sim, demand, pondB, mains, wastewater_pond, 0.85)
    catchmentA = SurfaceAWBM("catchmentA", sim, 20000.0, 0.5, pondA)
    catchmentB = SurfaceAWBM("catchmentB", sim, 50000.0, 0.0, pondB)
    catchmentC = SurfaceAWBM("catchmentC", sim, 500.0, 0.99, pondB)

    sim.run()

    # %%
    start_date = "2010-10-01"
    end_date = "2011-04-01"

    # start_date = "2005-01-01"
    # end_date = "2005-06-01"

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
        [
            catchmentC.imp_excess,
            catchmentC.baseflow_store,
            catchmentC.runoff_store,
            catchmentC.total_runoff,
        ],
        axis=1,
    )[start_date:end_date].plot()

    # %%
    pd.concat(
        [
            pondA.inflow,
            pondA.volume,
            pondA.supplied,
            pondA.overflow,
            pondA.et,
            pondA.leakage,
        ],
        axis=1,
    )[start_date:end_date].plot()

    # %%
    pd.concat(
        [
            pondB.inflow,
            pondB.volume,
            pondB.supplied,
            pondB.overflow,
            pondB.et,
            pondB.leakage,
        ],
        axis=1,
    )[start_date:end_date].plot()

    # %%
    pd.concat([fieldA.source_supplied, fieldA.demand], axis=1)[
        start_date:end_date
    ].plot()

    # %%
    sim.collate_results().to_excel("full_results.xlsx")
# %%
