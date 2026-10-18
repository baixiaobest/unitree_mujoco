"""Stand the real Go2 up using Unitree's built-in high-level sport controller.

This does NOT use any of this repo's custom locomotion policy -- it calls the
robot's own onboard balance controller via the standard SportClient RPC API,
the same command the official Unitree app's "stand" button triggers.
"""
import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", type=str, default="enp130s0")
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)

    client = SportClient(enableLease=False)
    client.Init()

    print("Sending StandUp...")
    code = client.StandUp()
    print(f"StandUp returned code={code}")
    time.sleep(1.0)

    print("Sending BalanceStand...")
    code = client.BalanceStand()
    print(f"BalanceStand returned code={code}")
