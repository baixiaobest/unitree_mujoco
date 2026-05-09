import argparse
import subprocess
import sys
import time
from pathlib import Path

DEVICE = "cuda"
SIMULATION_DOMAIN_ID = 1
HARDWARE_DOMAIN_ID = 0
SIMULATION_INTERFACE = "wlo1"
HARDWARE_INTERFACE = "enp108s0"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "logs"
    / "rsl_rl"
    / "EncoderActorCriticGO2"
    / "Locomotion"
    / "exported"
    / "policy_estimator.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GO2 position-control or velocity-control policy on hardware or simulation.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the TorchScript model to run.",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=("position_control", "velocity_control"),
        default="velocity_control",
        help="Select the observation/control mode for the loaded model.",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=("simulation", "hardware"),
        default="simulation",
        help="Choose whether to run against the simulation bridge or the real robot.",
    )
    parser.add_argument(
        "--debug-print",
        action="store_true",
        help="Print debug information such as the measured control update rate once per second.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=30.0,
        help="Proportional gain for joint position control.",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=0.5,
        help="Derivative gain for joint position control.",
    )
    parser.add_argument(
        "--no-status-monitor",
        action="store_true",
        help="Do not launch the status monitor GUI alongside the control loop.",
    )
    return parser.parse_args()


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    return str(path)


def create_simulation_env_class(base_env_cls):
    class SimulationGO2HardwareEnvironment(base_env_cls):
        def _init_unitree_services(self):
            """Skip service initialization in simulation mode."""
            print("Skipping Unitree service initialization (simulation mode)")

        def run(self):
            """Allow the simulation environment to drive the hardware step loop."""
            print("Hardware environment ready - control handed to simulation environment")

    return SimulationGO2HardwareEnvironment


def launch_status_monitor(run_mode: str) -> subprocess.Popen[bytes] | None:
    """Launch the status monitor as a companion process for the selected DDS domain."""
    monitor_script = Path(__file__).resolve().parent / "status_monitor.py"
    command = [sys.executable, str(monitor_script), "--run-mode", run_mode, "--device", DEVICE]
    try:
        print(f"Launching status monitor: {' '.join(command)}")
        return subprocess.Popen(command, cwd=str(monitor_script.parent))
    except Exception as error:  # noqa: BLE001 - control loop should still run if the monitor cannot launch.
        print(f"Warning: failed to launch status monitor: {error}")
        return None


def stop_status_monitor(monitor_process: subprocess.Popen[bytes] | None) -> None:
    """Terminate the status monitor process if it is still running."""
    if monitor_process is None or monitor_process.poll() is not None:
        return
    monitor_process.terminate()
    try:
        monitor_process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        monitor_process.kill()
        monitor_process.wait(timeout=5.0)


def main():
    args = parse_args()

    from env.go2_hardware_environment import GO2HardwareEnvironment
    from env.hardware_simulation_environment import HardwareSimulationEnvironment
    from robot_comm.robot_communication import RobotCommunication
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    model_path = resolve_model_path(args.model_path)
    is_simulation = args.run_mode == "simulation"
    status_monitor_process = None

    if is_simulation:
        print("Using simulated hardware environment")
        ChannelFactoryInitialize(SIMULATION_DOMAIN_ID, SIMULATION_INTERFACE)
    else:
        print("Connecting to real hardware")
        ChannelFactoryInitialize(HARDWARE_DOMAIN_ID, HARDWARE_INTERFACE)

    robot_comm = RobotCommunication(device=DEVICE)
    env_cls = create_simulation_env_class(GO2HardwareEnvironment) if is_simulation else GO2HardwareEnvironment
    env = env_cls(
        robot_comm=robot_comm,
        model_path=model_path,
        device=DEVICE,
        kp=args.kp,
        kd=args.kd,
        up_down_test=False,
        enable_logging=True,
        log_dir="../logs",
        policy_mode=args.policy_mode,
        runtime_mode=args.run_mode,
        debug_print=args.debug_print,
    )

    if not args.no_status_monitor:
        status_monitor_process = launch_status_monitor(args.run_mode)

    try:
        if is_simulation:
            sim_env = HardwareSimulationEnvironment(simulator_update_time=0.02)
            sim_env.hardware_env = env
            try:
                sim_env.start()
                while sim_env.running:
                    time.sleep(0.1)
            finally:
                sim_env.stop()
            return

        env.run()
    finally:
        stop_status_monitor(status_monitor_process)


if __name__ == "__main__":
    main()