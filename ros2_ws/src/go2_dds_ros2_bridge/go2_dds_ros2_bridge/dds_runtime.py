from __future__ import annotations

import argparse
from dataclasses import dataclass


RUN_MODE_CHOICES = ("simulation", "hardware")


@dataclass(frozen=True)
class DdsRuntimeProfile:
    domain_id: int
    interface: str


DEFAULT_SIMULATION_DDS_RUNTIME = DdsRuntimeProfile(domain_id=1, interface="wlo1")
DEFAULT_HARDWARE_DDS_RUNTIME = DdsRuntimeProfile(domain_id=0, interface="enp108s0")


def add_dds_runtime_arguments(parser: argparse.ArgumentParser, *, default_run_mode: str = "hardware") -> None:
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODE_CHOICES,
        default=default_run_mode,
        help="Select the default DDS domain/interface pair.",
    )
    parser.add_argument(
        "--dds-domain-id",
        type=int,
        default=None,
        help="Override the resolved raw DDS domain id used by this node.",
    )
    parser.add_argument(
        "--dds-interface",
        type=str,
        default=None,
        help="Override the resolved network interface used by this node.",
    )
    parser.add_argument(
        "--simulation-dds-domain-id",
        type=int,
        default=None,
        help="Override the shared default DDS domain id used when --run-mode simulation is selected.",
    )
    parser.add_argument(
        "--hardware-dds-domain-id",
        type=int,
        default=None,
        help="Override the shared default DDS domain id used when --run-mode hardware is selected.",
    )
    parser.add_argument(
        "--simulation-dds-interface",
        type=str,
        default=None,
        help="Override the shared default network interface used when --run-mode simulation is selected.",
    )
    parser.add_argument(
        "--hardware-dds-interface",
        type=str,
        default=None,
        help="Override the shared default network interface used when --run-mode hardware is selected.",
    )


def runtime_profile_for_run_mode(
    run_mode: str,
    *,
    simulation_dds_domain_id: int | None = None,
    hardware_dds_domain_id: int | None = None,
    simulation_dds_interface: str | None = None,
    hardware_dds_interface: str | None = None,
) -> DdsRuntimeProfile:
    if run_mode not in RUN_MODE_CHOICES:
        valid_modes = ", ".join(RUN_MODE_CHOICES)
        raise ValueError(f"Unsupported run mode '{run_mode}'. Expected one of: {valid_modes}")

    simulation_profile = DdsRuntimeProfile(
        domain_id=DEFAULT_SIMULATION_DDS_RUNTIME.domain_id
        if simulation_dds_domain_id is None
        else int(simulation_dds_domain_id),
        interface=DEFAULT_SIMULATION_DDS_RUNTIME.interface
        if simulation_dds_interface is None
        else str(simulation_dds_interface),
    )
    hardware_profile = DdsRuntimeProfile(
        domain_id=DEFAULT_HARDWARE_DDS_RUNTIME.domain_id
        if hardware_dds_domain_id is None
        else int(hardware_dds_domain_id),
        interface=DEFAULT_HARDWARE_DDS_RUNTIME.interface
        if hardware_dds_interface is None
        else str(hardware_dds_interface),
    )
    return simulation_profile if run_mode == "simulation" else hardware_profile


def resolve_runtime_arguments(args: argparse.Namespace) -> DdsRuntimeProfile:
    resolved_profile = runtime_profile_for_run_mode(
        args.run_mode,
        simulation_dds_domain_id=getattr(args, "simulation_dds_domain_id", None),
        hardware_dds_domain_id=getattr(args, "hardware_dds_domain_id", None),
        simulation_dds_interface=getattr(args, "simulation_dds_interface", None),
        hardware_dds_interface=getattr(args, "hardware_dds_interface", None),
    )
    return DdsRuntimeProfile(
        domain_id=resolved_profile.domain_id if args.dds_domain_id is None else int(args.dds_domain_id),
        interface=resolved_profile.interface if args.dds_interface is None else str(args.dds_interface),
    )