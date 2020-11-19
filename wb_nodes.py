from typing import Union
import numpy as np
import pandas as pd
from wb_simulation import Simulation


class Factor(float):
    """Float type with value constrained between 0.0 and 1.0.
    """

    MIN = 0.0
    MAX = 1.0

    def __new__(cls, *value):
        if value:
            v0 = float(value[0])
            if not (cls.MIN <= v0 <= cls.MAX):
                raise ValueError(f"Value {v0} for factor is out of range (0.0, 1.0).")

        return float.__new__(cls, *value)


class Storage:
    def __init__(
        self,
        name: str,
        simulation: Simulation,
        area: float,
        max_volume: float,
        initial_volume: float,
        demands: list,
        leakage: float,
        outlet: "Storage",
    ):
        self.name = name
        self.simulation = simulation
        self.area = area
        self.max_volume = max_volume
        self.initial_volume = initial_volume
        self.demands = demands
        self.leakage = leakage

        # initialise series to contain results
        self.inflow = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_storage_inflow"
        )
        self.volume = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_storage_volume"
        )
        self.supplied = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_storage_supplied"
        )
        self.overflow = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_storage_overflow"
        )

        # register with simulation
        self.simulation.storages[self.name] = self

    def step(self, time):
        pass

    # this will have to be integrated with the stepping to properly handle multiple demands from one storage
    def take(self, requested_volume):
        current_volume = self.volume[-1]

        if requested_volume <= current_volume:
            supplied_volume = requested_volume
        else:
            supplied_volume = current_volume

        self.volume.append(current_volume - supplied_volume)

        return supplied_volume

    def put(self, incoming_volume):
        pass


class Surface:
    def __init__(
        self,
        name: str,
        simulation: Simulation,
        area: float,
        imp_frac: float,
        outlet: Storage,
        a1: Factor = 0.134,
        a2: Factor = 0.433,
        c1: float = 7.0,
        c2: float = 70.0,
        c3: float = 150.0,
        bfi: Factor = 0.35,
        kbase: Factor = 0.95,
        ksurf: Factor = 0.35,
    ):

        self.simulation = simulation

        self.area = area
        self.imp_frac = imp_frac

        self.a1 = a1
        self.a2 = a2 if (a1 + a2 <= 1) else 1 - a1
        self.a3 = 1.0 - a1 - a2
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.bfi = bfi
        self.kbase = kbase
        self.ksurf = ksurf

        self.outlet = outlet

        # initialise series to contain results
        self.store1 = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_surface_store1"
        )
        self.store2 = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_surface_store2"
        )
        self.store3 = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_surface_store3"
        )
        self.runoff = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_surface_runoff"
        )
        self.baseflow = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_surface_baseflow"
        )
        self.et = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_surface_et"
        )
        self.groundwater = pd.Series(
            index=simulation.rainfall.index, name=f"{self.name}_surface_groundwater"
        )

        # register with simulation
        self.simulation.surfaces[self.name] = self

    def step(self, time):

        self.runoff[time] = None  # update all series with value for current timestep


class Mains:
    def __init__(self, simulation: Simulation):

        # initialise series to contain results
        self.supply = pd.Series(
            index=simulation.rainfall.index, data=0.0, name=f"{self.name}_mains_supply"
        )

        # register with simulation
        self.simulation.mains[self.name] = self

    def take(self, volume):
        # log requested mains supply and return it
        time = self.simulation.time
        self.supply[time] = self.supply[time] + volume
        return volume


class Demand:
    def __init__(
        self,
        name: str,
        simulation: Simulation,
        potable_demand: Union[pd.Series, None],
        nonpotable_demand: Union[pd.Series, None],
        source: Storage,
        backup: Union[Mains, None],
        wastewater_converstion_factor: float = 0.0,
    ):
        self.name = name
        self.simulation = simulation
        self.potable_demand = potable_demand
        self.nonpotable_demand = nonpotable_demand
        self.source = source
        self.backup = backup
        self.wastewater_converstion_factor = Factor(wastewater_converstion_factor)

        # initialise series to contain results
        self.satisfied_demand = pd.Series(
            index=simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_demand_satisfied",
        )
        self.unsatisfied_demand = pd.Series(
            index=simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_demand_unsatisfied",
        )

        # register with simulation
        self.simulation.demands[self.name] = self

    def step(self, time):
        current_demand = self.demand[time]
        satisfied_demand = self.source.take(current_demand)
        unsatisfied_demand = current_demand - satisfied_demand
        mains_demand = self.backup.take(unsatisfied_demand)

        self.satisfied_demand[time] = satisfied_demand
        self.unsatisfied_demand[time] = unsatisfied_demand


class Pump:
    def __init__(
        self,
        name: str,
        simulation: Simulation,
        source: Storage,
        destination: Storage,
        scheduled_flows: pd.Series,
    ):
        self.name: name
        self.simulation: simulation
        self.source: source
        self.destination: destination
        self.scheduled_flows: scheduled_flows

        # initialise series to contain results
        self.pumped_flows = pd.Series(
            index=simulation.rainfall.index, data=0.0, name=f"{self.name}_pumped_flows"
        )

        # register with simulation
        self.simulation.demands[self.name] = self

    def step(self):
        time = self.simulation.time
        # get scheduled flow for current timestep
        scheduled_flow = self.scheduled_flows[time]
        # take available flow from source and record it
        available_flow = self.source.take(scheduled_flow)
        self.pumped_flows[time] = available_flow
        # transfer available flow to destination
        self.destination.put(available_flow)


if __name__ == "__main__":
    import numpy as np

    timestep = pd.Timedelta("1 hour")
    datetime_index = pd.date_range("2020-01-01", "2020-12-31", freq=timestep)
    rainfall = pd.Series(
        index=datetime_index, data=np.random.random(size=len(datetime_index))
    )
    pet = pd.Series(
        index=datetime_index, data=np.random.random(size=len(datetime_index))
    )
    sim = Simulation(timestep=timestep, rainfall=rainfall, pet=pet)

