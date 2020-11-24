from typing import Union
import numpy as np
import pandas as pd
from wb_simulation import Simulation


class Base:
    def __str__(cls):
        return f"{type(cls)}: {cls.name}"


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


class Storage(Base):
    def __init__(
        self,
        name: str,
        simulation: Simulation,
        area: float,  # m2
        max_volume: float,  # m3
        initial_volume: float,  # m3
        leakage_rate: float,  # m3/timestep
        outlet: Union["Storage", None],
    ):
        self.name = name
        self.simulation = simulation
        self.area = area
        self.max_volume = max_volume
        self.initial_volume = initial_volume
        self.leakage_rate = leakage_rate
        self.outlet = outlet

        self.direct_rainfall = self.simulation.rainfall * self.area / 1000.0

        self.completed_step = pd.Timestamp("1900-01-01")

        # initialise series to contain results
        self.inflow = pd.Series(
            index=simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_storage_inflow",
        )
        self.volume = pd.Series(
            index=simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_storage_volume",
        )
        self.supplied = pd.Series(
            index=simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_storage_supplied",
        )
        self.overflow = pd.Series(
            index=simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_storage_overflow",
        )
        self.et = pd.Series(
            index=simulation.rainfall.index, data=0.0, name=f"{self.name}_storage_et"
        )
        self.leakage = pd.Series(
            index=simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_storage_leakage",
        )
        self.results = [
            self.direct_rainfall,
            self.inflow,
            self.volume,
            self.supplied,
            self.overflow,
            self.et,
            self.leakage,
        ]

        # update initial volume
        self.volume[0] = self.initial_volume

        # register with simulation
        self.simulation.storages[self.name] = self

    def step(self):
        time = self.simulation.time
        time_previous = self.simulation.time_previous

        # make sure that previous volume is only carried forward once per step
        if self.completed_step < time:
            self.volume.loc[time] = self.volume.loc[time_previous]
            self.completed_step = time

        # handle rainfall on storage & record in results
        if self.area > 0:
            rainfall_volume = self.direct_rainfall.loc[time]
            self.volume.loc[time] = self.volume.loc[time] + rainfall_volume
            self.inflow.loc[time] = self.inflow.loc[time] + rainfall_volume

        # handle et & record in results
        if self.area > 0:
            et_volume = min(
                self.volume.loc[time],
                self.simulation.pet.loc[time] * self.area / 1000.0,
            )
            self.volume.loc[time] = self.volume.loc[time] - et_volume
            self.et.loc[time] = self.et.loc[time] + et_volume

        # handle leakage & record in results
        if self.leakage_rate > 0:
            leakage = min(self.volume.loc[time], self.leakage_rate)
            self.volume.loc[time] = self.volume.loc[time] - leakage
            self.leakage.loc[time] = leakage

        # calculate and route any overflow
        if self.volume.loc[time] > self.max_volume:
            overflow = self.volume.loc[time] - self.max_volume
            self.volume.loc[time] = self.max_volume
            self.overflow.loc[time] = overflow
        else:
            overflow = 0.0

        if self.outlet is not None:
            self.outlet.put(overflow)

        self.done = False

    def take(self, requested_volume):
        time = self.simulation.time
        time_previous = self.simulation.time_previous

        # make sure that previous volume is only carried forward once per step
        if self.completed_step < time:
            self.volume.loc[time] = self.volume.loc[time_previous]
            self.completed_step = time

        current_volume = self.volume.loc[time]

        if requested_volume <= current_volume:
            supplied_volume = requested_volume
        else:
            supplied_volume = current_volume

        # update storage volume and record supplied flows in results
        self.volume.loc[time] = self.volume.loc[time] - supplied_volume
        self.supplied.loc[time] = self.supplied.loc[time] + supplied_volume

        return supplied_volume

    def put(self, incoming_volume):
        time = self.simulation.time
        time_previous = self.simulation.time_previous

        # make sure that previous volume is only carried forward once per step
        if self.completed_step < time:
            self.volume.loc[time] = self.volume.loc[time_previous]
            self.completed_step = time

        # update storage volume, handle overflows immediately and record incoming flows in results
        # new_volume = self.volume.loc[time] + incoming_volume
        # if new_volume > self.max_volume:
        #     overflow_volume = new_volume - self.max_volume
        # else:
        #     overflow_volume = 0.0
        # stored_volume = incoming_volume - overflow_volume

        self.volume.loc[time] = self.volume.loc[time] + incoming_volume
        # self.overflow.loc[time] = self.overflow.loc[time] + overflow_volume
        self.inflow.loc[time] = self.inflow.loc[time] + incoming_volume


class SurfaceAWBM(Base):
    def __init__(
        self,
        name: str,
        simulation: Simulation,
        area: float,
        imp_frac: float,
        outlet: Union[Storage, None],
        a1: float = 0.134,
        a2: float = 0.433,
        c1: float = 7.0,
        c2: float = 70.0,
        c3: float = 150.0,
        bfi: float = 0.35,
        kbase: float = 0.95,
        ksurf: float = 0.35,
        i: float = 1.0,
    ):
        self.name = name
        self.simulation = simulation
        self.area = area
        self.imp_frac = Factor(imp_frac)
        self.outlet = outlet

        self.a1 = Factor(a1)
        self.a2 = Factor(a2) if (a1 + a2 <= 1) else 1 - a1
        self.a3 = 1.0 - a1 - a2
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.bfi = Factor(bfi)
        self.kbase = Factor(kbase) ** (self.simulation.timestep / pd.Timedelta("1 day"))
        self.ksurf = Factor(ksurf) ** (self.simulation.timestep / pd.Timedelta("1 day"))
        self.i = i

        self.rainfall_volume = pd.Series(
            self.simulation.rainfall * self.area / 1000.0,
            name=f"{self.name}_surface_rainfall_volume",
        )

        # initialise series to contain results
        self.store1 = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_store1",
        )
        self.store2 = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_store2",
        )
        self.store3 = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_store3",
        )
        self.imp_store = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_imp_store",
        )
        self.et = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_et",
        )
        self.pervious_excess = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_pervious_excess",
        )
        self.imp_excess = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_imp_excess",
        )
        self.runoff_store = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_runoff_store",
        )
        self.runoff = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_runoff",
        )
        self.baseflow_store = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_baseflow_store",
        )
        self.baseflow = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_baseflow",
        )
        self.total_runoff = pd.Series(
            index=self.simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_surface_total_runoff",
        )
        self.results = [
            self.rainfall_volume,
            self.store1,
            self.store2,
            self.store3,
            self.imp_store,
            self.et,
            self.pervious_excess,
            self.imp_excess,
            self.runoff_store,
            self.runoff,
            self.baseflow_store,
            self.baseflow,
            self.total_runoff,
        ]

        # register with simulation
        self.simulation.surfaces[self.name] = self

    def step(self):
        time = self.simulation.time
        time_previous = self.simulation.time_previous
        rainfall = self.simulation.rainfall.loc[time]
        pet = self.simulation.pet.loc[time]

        # add rainfall
        self.store1.loc[time] = self.store1.loc[time_previous] + rainfall
        self.store2.loc[time] = self.store2.loc[time_previous] + rainfall
        self.store3.loc[time] = self.store3.loc[time_previous] + rainfall
        self.imp_store.loc[time] = self.imp_store.loc[time_previous] + rainfall

        # remove et & record
        et1 = min(self.store1.loc[time], pet)
        et2 = min(self.store2.loc[time], pet)
        et3 = min(self.store3.loc[time], pet)
        eti = min(self.imp_store.loc[time], pet)

        self.store1.loc[time] = self.store1.loc[time] - et1
        self.store2.loc[time] = self.store2.loc[time] - et2
        self.store3.loc[time] = self.store3.loc[time] - et3
        self.imp_store.loc[time] = self.imp_store.loc[time] - eti

        et = (self.a1 * et1 + self.a2 * et2 + self.a3 * et3) * (1.0 - self.imp_frac) + (
            eti * self.imp_frac
        )
        self.et.loc[time] = et * self.area / 1000.0

        # calculate rainfall excess (mm)
        if self.store1.loc[time] > self.c1:
            e1 = self.store1.loc[time] - self.c1
            self.store1.loc[time] = self.store1.loc[time] - e1
        else:
            e1 = 0.0
        if self.store2.loc[time] > self.c2:
            e2 = self.store2.loc[time] - self.c2
            self.store2.loc[time] = self.store2.loc[time] - e2
        else:
            e2 = 0.0
        if self.store3.loc[time] > self.c3:
            e3 = self.store3.loc[time] - self.c3
            self.store3.loc[time] = self.store3.loc[time] - e3
        else:
            e3 = 0.0
        if self.imp_store.loc[time] > self.i:
            ei = self.imp_store.loc[time] - self.i
            self.imp_store.loc[time] = self.imp_store.loc[time] - ei
        else:
            ei = 0.0

        # calculate excess volume from pervious areas
        pervious_excess = (
            (self.a1 * e1 + self.a2 * e2 + self.a3 * e3)
            * (1.0 - self.imp_frac)
            * self.area
            / 1000.0
        )  # in m3
        self.pervious_excess.loc[time] = pervious_excess

        # calculate impervious excess volume and route directly
        imp_excess = ei * self.imp_frac * self.area / 1000.0  # in m3
        self.imp_excess.loc[time] = imp_excess
        self.runoff.loc[time] = imp_excess

        # route baseflow and runoff from pervious excess through stores
        self.baseflow_store.loc[time] = self.baseflow_store.loc[time_previous] + (
            pervious_excess * self.bfi
        )
        baseflow = (1.0 - self.kbase) * self.baseflow_store.loc[time]
        self.baseflow_store[time] = self.baseflow_store[time] - baseflow
        self.baseflow.loc[time] = baseflow

        self.runoff_store.loc[time] = self.runoff_store.loc[time_previous] + (
            (pervious_excess) * (1.0 - self.bfi)
        )
        runoff = (1.0 - self.ksurf) * self.runoff_store.loc[time]
        self.runoff_store[time] = self.runoff_store[time] - runoff
        self.runoff.loc[time] = self.runoff.loc[time] + runoff

        # calculate total flow and send to outlet
        total_runoff = self.baseflow.loc[time] + self.runoff.loc[time]
        self.total_runoff.loc[time] = total_runoff

        if self.outlet is not None:
            self.outlet.put(total_runoff)


class Mains(Base):
    def __init__(self, name: str, simulation: Simulation):
        self.name = name
        self.simulation = simulation

        # initialise series to contain results
        self.supply = pd.Series(
            index=simulation.rainfall.index, data=0.0, name=f"{self.name}_mains_supply"
        )
        self.results = [self.supply]

        # register with simulation
        self.simulation.mains[self.name] = self

    def take(self, volume):
        # log requested mains supply and return it
        time = self.simulation.time
        self.supply[time] = self.supply[time] + volume
        return volume


class Demand(Base):
    def __init__(
        self,
        name: str,
        simulation: Simulation,
        demand: pd.Series,
        source: Storage,
        backup: Union[Mains, None],
        wastewater_sink: Union[Storage, None],
        wastewater_conversion_factor: float = 0.0,
    ):
        self.name = name
        self.simulation = simulation
        self.demand = demand
        self.source = source
        self.backup = backup
        self.wastewater_sink = wastewater_sink
        self.wastewater_conversion_factor = Factor(wastewater_conversion_factor)

        # initialise series to contain results
        self.source_supplied = pd.Series(
            index=simulation.rainfall.index, data=0.0, name=f"{self.name}_demand_source"
        )
        self.backup_supplied = pd.Series(
            index=simulation.rainfall.index, data=0.0, name=f"{self.name}_demand_backup"
        )
        self.wastewater_stream = pd.Series(
            index=simulation.rainfall.index,
            data=0.0,
            name=f"{self.name}_demand_wastewater_stream",
        )
        self.results = [
            self.source_supplied,
            self.backup_supplied,
            self.wastewater_stream,
        ]

        # register with simulation
        self.simulation.demands[self.name] = self

    def step(self):
        time = self.simulation.time
        # satisfy demand first from the source then from backup if needed
        current_demand = self.demand[time]
        source_supplied = self.source.take(current_demand)
        if self.backup:
            backup_required = current_demand - source_supplied
            backup_supplied = self.backup.take(backup_required)
        else:
            backup_supplied = 0.0
        # divert specified portion of demand to wastewater
        wastewater_stream = current_demand * self.wastewater_conversion_factor
        if self.wastewater_sink:
            self.wastewater_sink.put(wastewater_stream)
        # record in results
        self.source_supplied[time] = source_supplied
        self.backup_supplied[time] = backup_supplied
        self.wastewater_stream[time] = wastewater_stream


class Pump(Base):
    def __init__(
        self,
        name: str,
        simulation: Simulation,
        source: Storage,
        destination: Storage,
        scheduled_flows: pd.Series,
    ):
        self.name = name
        self.simulation = simulation
        self.source = source
        self.destination = destination
        self.scheduled_flows = scheduled_flows

        # initialise series to contain results
        self.pumped_flows = pd.Series(
            index=simulation.rainfall.index, data=0.0, name=f"{self.name}_pumped_flows"
        )
        self.results = [self.pumped_flows]

        # register with simulation
        self.simulation.pumps[self.name] = self

    def step(self):
        time = self.simulation.time

        # get scheduled flow for current timestep
        scheduled_flow = self.scheduled_flows[time]

        # take available flow from source and record it
        available_flow = self.source.take(scheduled_flow)

        # transfer available flow to destination
        self.destination.put(available_flow)

        # record in results
        self.pumped_flows[time] = available_flow

