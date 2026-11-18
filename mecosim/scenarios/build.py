"""Turn a :class:`ScenarioSpec` into SUMO input files (netgenerate, randomTrips.py) and mecosim entities."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

from mecosim.adapters.sumo.connection import sumo_binary, sumo_tool
from mecosim.core.entities import Hub, VehicleSpec, Zone
from mecosim.paths import absolute
from mecosim.scenarios.spec import ScenarioSpec

__all__ = ["BuiltScenario", "build_scenario", "PLUG_LENGTH_M"]

#: Stopping length reserved per plug at a hub (vehicle length 5 m + minGap 2.5 m + margin).
PLUG_LENGTH_M = 8.0


@dataclass
class BuiltScenario:
    """Files and entities of a built scenario."""

    spec: ScenarioSpec
    out_dir: Path
    net_file: Path
    sumocfg: Path
    route_files: list[Path]
    additional_files: list[Path]
    hubs: dict[str, Hub]
    zones: dict[str, Zone]
    fleet: dict[str, VehicleSpec]
    zone_of_edge: dict[str, str] = field(default_factory=dict)

    def sumo_args(self, output_dir: Path | None = None, extra: list[str] | None = None) -> list[str]:
        """SUMO arguments: the config plus output files in ``output_dir``."""
        out = Path(output_dir or self.out_dir / "output")
        out.mkdir(parents=True, exist_ok=True)
        args = [
            "-c",
            str(self.sumocfg),
            "--tripinfo-output",
            str(out / "tripinfo.xml"),
            "--statistic-output",
            str(out / "statistics.xml"),
            "--stop-output",
            str(out / "stops.xml"),
            "--log",
            str(out / "sumo.log"),
        ]
        return args + list(extra or [])


def _run(cmd: list[str], cwd: Path) -> None:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")


def _generate_network(spec: ScenarioSpec, out: Path) -> Path:
    net = spec.network
    if net.kind == "file":
        if net.net_file is None:
            raise ValueError("network.kind=file needs network.net_file")
        return absolute(net.net_file)
    path = out / "net.net.xml"
    _run(
        [
            sumo_binary("netgenerate"),
            "--grid",
            "--grid.number",
            str(net.grid_number),
            "--grid.length",
            str(net.grid_length_m),
            "--default.lanenumber",
            str(net.lanes),
            "--default.speed",
            str(net.speed_mps),
            "--no-turnarounds.except-deadend",
            "true",
            "--seed",
            str(spec.seed),
            "-o",
            str(path),
        ],
        out,
    )
    return path


def _assign_zones(spec: ScenarioSpec, net_file: Path) -> tuple[dict[str, str], dict[str, tuple[float, float]], object]:
    import sumolib

    net = sumolib.net.readNet(str(net_file))
    xmin, ymin, xmax, ymax = net.getBoundary()
    width = max(1e-9, xmax - xmin)
    zone_of_edge: dict[str, str] = {}
    mids: dict[str, tuple[float, float]] = {}
    for edge in net.getEdges():
        if edge.getFunction() == "internal" or not edge.allows("taxi"):
            continue
        shape = edge.getShape()
        mx = sum(p[0] for p in shape) / len(shape)
        my = sum(p[1] for p in shape) / len(shape)
        mids[edge.getID()] = (mx, my)
        frac = (mx - xmin) / width
        for z in spec.zones:
            lo, hi = z.x_range
            if lo <= frac < hi or (hi >= 1.0 and frac >= lo):
                zone_of_edge[edge.getID()] = z.id
                break
    return zone_of_edge, mids, net


def build_scenario(spec: ScenarioSpec, out_dir: str | Path) -> BuiltScenario:
    """Generate the network, stations, fleet, ride requests and background traffic for ``spec`` in ``out_dir``."""
    out = absolute(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim = spec.simulation
    net_file = _generate_network(spec, out)
    zone_of_edge, mids, net = _assign_zones(spec, net_file)
    xmin, ymin, xmax, ymax = net.getBoundary()  # type: ignore[attr-defined]

    # ---------------------------------------------------------------------------------- hubs
    hubs: dict[str, Hub] = {}
    used: set[str] = set()
    stations: list[str] = []
    for hs in spec.hubs:
        need = hs.plugs * PLUG_LENGTH_M + 10.0
        zspec = next(z for z in spec.zones if z.id == hs.zone)
        if hs.edge is not None:
            edge_id = hs.edge
        else:
            cx = xmin + (xmax - xmin) * 0.5 * (zspec.x_range[0] + min(1.0, zspec.x_range[1]))
            cy = 0.5 * (ymin + ymax)
            cands = [
                e
                for e, z in zone_of_edge.items()
                if z == hs.zone and e not in used and net.getEdge(e).getLength() >= need  # type: ignore[attr-defined]
            ]
            if not cands:
                raise ValueError(f"no edge in zone {hs.zone} long enough for hub {hs.id} ({need:.0f} m)")
            edge_id = min(cands, key=lambda e: (mids[e][0] - cx) ** 2 + (mids[e][1] - cy) ** 2)
        used.add(edge_id)
        length = net.getEdge(edge_id).getLength()  # type: ignore[attr-defined]
        span = hs.plugs * PLUG_LENGTH_M
        start = max(0.0, 0.5 * (length - span))
        station_id = f"cs_{hs.id}"
        stations.append(
            f'    <chargingStation id={quoteattr(station_id)} lane="{edge_id}_0" startPos="{start:.2f}" '
            f'endPos="{start + span:.2f}" power="0" efficiency="1" chargeInTransit="false" chargeDelay="0"/>'
        )
        hubs[hs.id] = Hub(
            id=hs.id,
            zone=hs.zone,
            station_id=station_id,
            edge_id=edge_id,
            plugs=hs.plugs,
            power_kw=hs.power_kw,
            charge_efficiency=hs.charge_efficiency,
            discharge_efficiency=hs.discharge_efficiency,
            curbside=hs.curbside,
            charger_ids=tuple(f"{hs.id}_c{p}" for p in range(hs.plugs)),
            building_id=hs.building,
        )
    add_file = out / "stations.add.xml"
    add_file.write_text("<additional>\n" + "\n".join(stations) + "\n</additional>\n", encoding="utf-8")

    # ---------------------------------------------------------------------------------- fleet
    fleet: dict[str, VehicleSpec] = {}
    fs = spec.fleet
    params = {"device.battery.capacity": fs.capacity_kwh * 1000.0, **fs.energy_params}
    vtype_params = "\n".join(f'    <param key="{k}" value="{v}"/>' for k, v in params.items())
    vehicles = []
    hub_list = list(hubs.values())
    for i in range(fs.n_vehicles):
        vid = f"{fs.id_prefix}{i}"
        hub = hub_list[i % len(hub_list)]
        vehicles.append(
            f'  <vehicle id="{vid}" type="av_taxi" depart="{sim.begin_s:.1f}" line="taxi" departPos="base">\n'
            f'    <route edges="{hub.edge_id}"/>\n'
            f'    <param key="device.battery.chargeLevel" value="{fs.initial_soc * fs.capacity_kwh * 1000.0:.3f}"/>\n'
            f"  </vehicle>"
        )
        fleet[vid] = VehicleSpec(vid, fs.capacity_kwh, fs.max_power_kw, fs.initial_soc, ev_id=vid)
    fleet_file = out / "fleet.rou.xml"
    fleet_file.write_text(
        "<routes>\n"
        f'  <vType id="av_taxi" vClass="taxi" personCapacity="{fs.person_capacity}" emissionClass="Energy/unknown">\n'
        '    <param key="has.battery.device" value="true"/>\n'
        '    <param key="has.taxi.device" value="true"/>\n'
        f"{vtype_params}\n"
        "  </vType>\n" + "\n".join(vehicles) + "\n</routes>\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------------------- demand
    begin, end = sim.begin_s, sim.begin_s + sim.duration_s
    random_trips = str(sumo_tool("randomTrips.py"))
    person_file = out / "persons.rou.xml"
    _run(
        [
            sys.executable,
            random_trips,
            "-n",
            str(net_file),
            "--personrides",
            "taxi",
            "-p",
            str(spec.demand.ride_period_s),
            "-b",
            str(begin),
            "-e",
            str(end),
            "--min-distance",
            str(spec.demand.ride_min_distance_m),
            "--seed",
            str(spec.seed),
            "--prefix",
            "p",
            "--vclass",
            "taxi",
            "-o",
            str(person_file),
        ],
        out,
    )
    route_files = [fleet_file, person_file]
    if spec.demand.background_period_s:
        bg_file = out / "background.trips.xml"
        _run(
            [
                sys.executable,
                random_trips,
                "-n",
                str(net_file),
                "-p",
                str(spec.demand.background_period_s),
                "-b",
                str(begin),
                "-e",
                str(end),
                "--fringe-factor",
                str(spec.demand.background_fringe_factor),
                "--seed",
                str(spec.seed + 1),
                "--prefix",
                "bg",
                "--validate",
                "-o",
                str(bg_file),
            ],
            out,
        )
        route_files.append(bg_file)

    # ---------------------------------------------------------------------------------- config
    period = sim.decision_period_s or sim.master_step_s
    cfg = out / "scenario.sumocfg"
    cfg.write_text(
        f"""<configuration>
  <input>
    <net-file value="{net_file}"/>
    <route-files value="{",".join(str(p) for p in route_files)}"/>
    <additional-files value="{add_file}"/>
  </input>
  <time>
    <begin value="{begin}"/>
    <step-length value="{sim.mobility_step_s}"/>
  </time>
  <processing>
    <ignore-route-errors value="true"/>
    <time-to-teleport value="300"/>
  </processing>
  <random_number>
    <seed value="{spec.seed}"/>
  </random_number>
  <taxi_device>
    <device.taxi.dispatch-algorithm value="traci"/>
    <device.taxi.dispatch-period value="{period}"/>
    <device.taxi.idle-algorithm value="stop"/>
  </taxi_device>
  <report>
    <no-step-log value="true"/>
    <duration-log.disable value="true"/>
  </report>
</configuration>
""",
        encoding="utf-8",
    )
    zones = {z.id: Zone(z.id, z.role, z.pv_scale, tuple(z.buildings)) for z in spec.zones}
    return BuiltScenario(spec, out, net_file, cfg, route_files, [add_file], hubs, zones, fleet, zone_of_edge)
