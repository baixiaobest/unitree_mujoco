#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from pathlib import Path
import argparse
import os
import sys

# Add the parent directory to the path to import math_utils
sys.path.append('/home/baixiao/Desktop/IsaacLab/source/unitree_mujoco/simulate_python')
from utils.math_utils import euler_xyz_from_quat

def load_log_data(log_path):
    """Load log data from CSV file"""
    return pd.read_csv(log_path)

def load_metadata(metadata_path):
    """Load metadata from JSON file"""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def create_base_position_plot(data, output_dir, figsize=(14, 8)):
    """Create time series plots for base position"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find base position columns
    pos_cols = [col for col in data.columns if 'base_position_' in col]
    
    if not pos_cols:
        print("Base position columns not found in data")
        return None
    
    fig = plt.figure(figsize=figsize)
    for col in sorted(pos_cols):
        plt.plot(data['time'], data[col], label=col)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Base Position over time')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save figure
    plt.savefig(output_dir / 'base_position_time_series.png')
    print("Created base position time series plot")
    
    return fig

def create_command_plot(data, output_dir, figsize=(14, 8)):
    """Create time series plot for commands"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find command columns
    cmd_cols = [col for col in data.columns if 'command_' in col]
    
    if not cmd_cols:
        print("Command columns not found in data")
        return None
    
    fig = plt.figure(figsize=figsize)
    for col in sorted(cmd_cols):
        plt.plot(data['time'], data[col], label=col)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Command Value')
    plt.title('Command over time')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save figure
    plt.savefig(output_dir / 'command_time_series.png')
    print("Created command time series plot")
    
    return fig

def create_estimator_vs_true_velocity_plot(data, output_dir, figsize=(14, 10)):
    """Create time series plots comparing estimated and true base linear velocity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    axis_labels = ["x", "y", "z"]
    estimated_cols = [f"estimated_base_lin_vel_{index}" for index in range(3)]
    true_cols = [f"obs_base_lin_vel_{index}" for index in range(3)]

    available_axes = [
        index for index, (estimated_col, true_col) in enumerate(zip(estimated_cols, true_cols, strict=False))
        if estimated_col in data.columns and true_col in data.columns
    ]

    if not available_axes:
        print("Estimated and true base linear velocity columns not found in data")
        return None

    fig, axes = plt.subplots(len(available_axes), 1, figsize=figsize, sharex=True)
    if len(available_axes) == 1:
        axes = [axes]

    for axis, velocity_index in zip(axes, available_axes, strict=False):
        estimated_col = estimated_cols[velocity_index]
        true_col = true_cols[velocity_index]
        axis_label = axis_labels[velocity_index]

        axis.plot(data["time"], data[true_col], label=f"true {axis_label}", linewidth=2)
        axis.plot(data["time"], data[estimated_col], label=f"estimated {axis_label}", linewidth=2, alpha=0.85)
        axis.set_ylabel(f"{axis_label} vel (m/s)")
        axis.set_title(f"Base Linear Velocity {axis_label.upper()}: estimator vs true")
        axis.grid(True)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.savefig(output_dir / "estimator_vs_true_base_lin_vel.png")
    print("Created estimator vs true base linear velocity plot")

    return fig

def create_joint_plots(data, output_dir, figsize=(14, 8)):
    """Create separate time series plots for joint positions and velocities"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    figures = []
    
    # Find joint position columns
    joint_pos_cols = [col for col in data.columns if 'joint_positions_' in col]
    
    if joint_pos_cols:
        fig = plt.figure(figsize=figsize)
        for col in sorted(joint_pos_cols):
            plt.plot(data['time'], data[col], label=col)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Position (rad)')
        plt.title('Joint Positions over time')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save figure
        plt.savefig(output_dir / 'joint_positions_time_series.png')
        print("Created joint positions time series plot")
        figures.append(fig)
    
    # Find joint velocity columns
    joint_vel_cols = [col for col in data.columns if 'joint_velocities_' in col]
    
    if joint_vel_cols:
        fig = plt.figure(figsize=figsize)
        for col in sorted(joint_vel_cols):
            plt.plot(data['time'], data[col], label=col)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Velocity (rad/s)')
        plt.title('Joint Velocities over time')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save figure
        plt.savefig(output_dir / 'joint_velocities_time_series.png')
        print("Created joint velocities time series plot")
        figures.append(fig)
    
    return figures

def create_time_series_plots(data, output_dir, figsize=(14, 8)):
    """Create time series plots for specified columns, excluding certain groups"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Skip these prefixes (handled separately)
    skip_prefixes = ['joint', 'base_position', 'obstacle', 'obst']
    
    # Group columns by their prefix (before underscore)
    column_groups = {}
    for col in data.columns:
        # Skip step and time columns
        if col in ['step', 'time']:
            continue
        
        # Skip columns with specific prefixes
        if any(col.startswith(prefix) for prefix in skip_prefixes):
            continue
            
        # Group by prefix
        if '_' in col:
            prefix = col.split('_')[0]
            if prefix not in column_groups:
                column_groups[prefix] = []
            column_groups[prefix].append(col)
        else:
            # For columns without underscore, use the column name as prefix
            if col not in column_groups:
                column_groups[col] = [col]
    
    figures = []
    
    # Plot each group
    for group_name, group_cols in column_groups.items():
        fig = plt.figure(figsize=figsize)
        for col in sorted(group_cols):
            plt.plot(data['time'], data[col], label=col)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title(f'{group_name} over time')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save figure
        plt.savefig(output_dir / f'{group_name}_time_series.png')
        print(f"Created time series plot for {group_name}")
        figures.append(fig)
    
    return figures

def plot_robot_trajectory(data, output_dir, figsize=(10, 8)):
    """Create trajectory plot for robot position and command targets"""
    # Check for position columns
    pos_x_col = [col for col in data.columns if 'base_position_0' in col]
    pos_y_col = [col for col in data.columns if 'base_position_1' in col]
    
    # Check for quaternion columns
    quat_w_col = [col for col in data.columns if 'base_quaternion_0' in col]
    quat_x_col = [col for col in data.columns if 'base_quaternion_1' in col]
    quat_y_col = [col for col in data.columns if 'base_quaternion_2' in col]
    quat_z_col = [col for col in data.columns if 'base_quaternion_3' in col]
    
    # Check for command columns
    cmd_x_col = [col for col in data.columns if 'command_0' in col]
    cmd_y_col = [col for col in data.columns if 'command_1' in col]
    
    if not pos_x_col or not pos_y_col:
        print("Position columns not found in data")
        return None
    
    pos_x_col = pos_x_col[0]
    pos_y_col = pos_y_col[0]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=figsize)
    
    # Plot robot trajectory
    plt.plot(data[pos_x_col], data[pos_y_col], 'b-', linewidth=2, label='Robot Path')
    
    # Mark start and end points
    plt.plot(data[pos_x_col].iloc[0], data[pos_y_col].iloc[0], 'go', markersize=10, label='Start')
    plt.plot(data[pos_x_col].iloc[-1], data[pos_y_col].iloc[-1], 'ro', markersize=10, label='End')
    
    # Add direction arrows
    step_size = max(1, len(data) // 20)  # Add ~20 arrows along the path
    for i in range(0, len(data) - step_size, step_size):
        plt.arrow(data[pos_x_col].iloc[i], data[pos_y_col].iloc[i],
                 data[pos_x_col].iloc[i+step_size] - data[pos_x_col].iloc[i],
                 data[pos_y_col].iloc[i+step_size] - data[pos_y_col].iloc[i],
                 head_width=0.05, head_length=0.1, fc='b', ec='b', alpha=0.5)
    
    # Add command positions if all required columns exist
    if (cmd_x_col and cmd_y_col and quat_w_col and quat_x_col and 
        quat_y_col and quat_z_col):
        
        cmd_x_col = cmd_x_col[0]
        cmd_y_col = cmd_y_col[0]
        quat_w_col = quat_w_col[0]
        quat_x_col = quat_x_col[0]
        quat_y_col = quat_y_col[0]
        quat_z_col = quat_z_col[0]
        
        # Convert command positions from local to global frame
        global_cmd_x = []
        global_cmd_y = []
        
        # Create a batch tensor for quaternions
        quaternions = torch.zeros((len(data), 4), dtype=torch.float32)
        
        for i in range(len(data)):
            # Extract quaternion components
            quaternions[i, 0] = data[quat_w_col].iloc[i]  # w
            quaternions[i, 1] = data[quat_x_col].iloc[i]  # x
            quaternions[i, 2] = data[quat_y_col].iloc[i]  # y
            quaternions[i, 3] = data[quat_z_col].iloc[i]  # z
        
        # Extract yaw from all quaternions at once using math_utils
        _, _, yaws = euler_xyz_from_quat(quaternions)
        
        for i in range(len(data)):
            # Extract robot position
            robot_x = data[pos_x_col].iloc[i]
            robot_y = data[pos_y_col].iloc[i]
            
            # Get the pre-computed yaw
            yaw = yaws[i].item()
            
            # Extract command in local frame
            local_x = data[cmd_x_col].iloc[i]
            local_y = data[cmd_y_col].iloc[i]
            
            # Transform using 2D rotation with yaw angle only
            global_x = robot_x + local_x * np.cos(yaw) - local_y * np.sin(yaw)
            global_y = robot_y + local_x * np.sin(yaw) + local_y * np.cos(yaw)
            
            global_cmd_x.append(global_x)
            global_cmd_y.append(global_y)
        
        # Plot command points in global frame (at selected intervals)
        for i in range(0, len(global_cmd_x), step_size):
            plt.plot(global_cmd_x[i], global_cmd_y[i], 'mx', markersize=8)
            
            # Draw lines connecting robot to command targets
            plt.plot([data[pos_x_col].iloc[i], global_cmd_x[i]], 
                    [data[pos_y_col].iloc[i], global_cmd_y[i]], 
                    'g--', alpha=0.3)
        
        # Add legend entry for command targets
        plt.scatter([], [], c='m', marker='x', s=64, label='Command Targets')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Robot Trajectory with Command Targets')
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis('equal')  # Equal scaling for x and y
    
    # Save figure
    plt.savefig(output_dir / 'robot_trajectory.png')
    print("Created robot trajectory plot with command targets")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Analyze robot log data')
    parser.add_argument('log_path', help='Path to the log file or directory containing logs')
    parser.add_argument('--output', '-o', default=None, help='Output directory for plots')
    parser.add_argument('--latest', '-l', action='store_true', help='Use only the latest log file')
    parser.add_argument('--no-display', '-nd', action='store_true', help='Do not display plots, only save them')
    
    args = parser.parse_args()
    show_plots = not args.no_display
    
    # Handle directory input
    log_path = Path(args.log_path)
    if log_path.is_dir():
        log_files = sorted(log_path.glob("robot_log_*.csv"))
        if not log_files:
            print(f"No log files found in {log_path}")
            return
            
        if args.latest:
            # Use only the latest log file
            log_path = log_files[-1]
            print(f"Using latest log file: {log_path}")
        else:
            # Process all log files
            for log_file in log_files:
                print(f"Processing log file: {log_file}")
                # Set output directory based on log filename
                if args.output:
                    output_dir = Path(args.output) / log_file.stem
                else:
                    output_dir = log_path / 'analysis' / log_file.stem
                
                process_log_file(log_file, output_dir, show_plots)
            return
    
    # Set default output directory if not specified
    if args.output is None:
        output_dir = log_path.parent / 'analysis' / log_path.stem
    else:
        output_dir = Path(args.output)
    
    process_log_file(log_path, output_dir, show_plots)

def process_log_file(log_path, output_dir, show_plots=True):
    """Process a single log file"""
    # Check if file exists
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return
        
    # Try to find metadata file
    metadata_path = log_path.parent / f"metadata_{log_path.stem.split('_')[-1]}.json"
    
    # Load data and metadata
    print(f"Loading log data from {log_path}...")
    data = load_log_data(log_path)
    
    if metadata_path.exists():
        metadata = load_metadata(metadata_path)
        print(f"Log contains {len(data)} entries from {metadata.get('total_steps', 'unknown')} steps")
    else:
        print(f"Metadata file not found: {metadata_path}")
        metadata = None
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create plots
    print(f"Creating plots in {output_dir}...")
    
    # Base position plot
    create_base_position_plot(data, output_dir)
    
    # Command plot
    create_command_plot(data, output_dir)

    # Estimated vs true base linear velocity plot
    create_estimator_vs_true_velocity_plot(data, output_dir)
    
    # Joint plots (separate for positions and velocities)
    # create_joint_plots(data, output_dir)
    
    # Other time series plots
    # create_time_series_plots(data, output_dir)
    
    # Trajectory plot
    plot_robot_trajectory(data, output_dir)

    plt.show() if show_plots else plt.close('all')
    
    print(f"Analysis complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()