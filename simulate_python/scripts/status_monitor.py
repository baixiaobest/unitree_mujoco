import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                             QProgressBar, QGroupBox, QPushButton, QScrollArea)
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# Import the RobotCommunication class
sys.path.append("../")
from robot_comm.robot_communication import RobotCommunication
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

class StatusMonitor(QMainWindow):
    # Define joint names as a class variable
    joint_names = [
        "0 (FR_0)", "1 (FR_1)", "2 (FR_2)",
        "3 (FL_0)", "4 (FL_1)", "5 (FL_2)",
        "6 (RR_0)", "7 (RR_1)", "8 (RR_2)",
        "9 (RL_0)", "10 (RL_1)", "11 (RL_2)"
    ]

    def __init__(self, robot_comm: RobotCommunication):
        super().__init__()
        
        self.robot_comm = robot_comm
        self.init_ui()
        
        # Timer for updating the GUI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(100)  # Update every 100ms (10Hz)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Unitree Robot Status Monitor')
        self.setGeometry(100, 100, 800, 600)  # Adjusted window size
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        main_layout.setSpacing(5)  # Reduce spacing
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs
        self.joint_tab = self.create_joint_tab()
        self.base_tab = self.create_base_tab()
        self.imu_tab = self.create_imu_tab()
        self.command_tab = self.create_command_tab()
        
        # Add tabs to widget
        self.tabs.addTab(self.wrap_in_scroll_area(self.joint_tab), "Joint Status")
        self.tabs.addTab(self.base_tab, "Base Status")
        self.tabs.addTab(self.imu_tab, "IMU Data")
        self.tabs.addTab(self.command_tab, "Commands")
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
        
        # Control panel (start/stop updates)
        control_layout = QHBoxLayout()
        self.refresh_rate_label = QLabel("Refresh Rate: 10 Hz")
        self.toggle_button = QPushButton("Pause Updates")
        self.toggle_button.clicked.connect(self.toggle_updates)
        
        control_layout.addWidget(self.refresh_rate_label)
        control_layout.addWidget(self.toggle_button)
        main_layout.addLayout(control_layout)
        
        self.show()
    
    def wrap_in_scroll_area(self, widget):
        """Wrap a widget in a QScrollArea for scrolling"""
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll_area
    
    def create_joint_tab(self):
        """Create the Joint Status tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        layout.setSpacing(5)  # Reduce spacing

        # Create group boxes for positions, velocities, and torques
        self.position_group = QGroupBox("Joint Positions (rad)")
        self.velocity_group = QGroupBox("Joint Velocities (rad/s)")
        self.torque_group = QGroupBox("Joint Torques (N·m)")

        # Create layouts for each group
        pos_layout = QGridLayout()
        vel_layout = QGridLayout()
        torque_layout = QGridLayout()

        # Initialize labels and progress bars for 12 joints
        self.joint_pos_labels = []
        self.joint_pos_bars = []
        self.joint_vel_labels = []
        self.joint_vel_bars = []
        self.joint_torque_labels = []
        self.joint_torque_bars = []

        for i, joint_name in enumerate(self.joint_names):  # Use class variable
            # Joint position
            label = QLabel(f"{joint_name}: 0.00")
            label.setFont(QFont("Arial", 12))  # Smaller font
            bar = QProgressBar()
            bar.setRange(-180, 180)  # Range in degrees
            bar.setValue(0)
            bar.setFixedHeight(15)  # Compact height
            pos_layout.addWidget(label, i, 0)
            pos_layout.addWidget(bar, i, 1)
            self.joint_pos_labels.append(label)
            self.joint_pos_bars.append(bar)

            # Joint velocity
            label = QLabel(f"{joint_name}: 0.00")
            label.setFont(QFont("Arial", 12))  # Smaller font
            bar = QProgressBar()
            bar.setRange(-10, 10)  # Range for velocity
            bar.setValue(0)
            bar.setFixedHeight(15)  # Compact height
            vel_layout.addWidget(label, i, 0)
            vel_layout.addWidget(bar, i, 1)
            self.joint_vel_labels.append(label)
            self.joint_vel_bars.append(bar)

            # Joint torque
            label = QLabel(f"{joint_name}: 0.00")
            label.setFont(QFont("Arial", 12))  # Smaller font
            bar = QProgressBar()
            bar.setRange(-20, 20)  # Range for torque
            bar.setValue(0)
            bar.setFixedHeight(15)  # Compact height
            torque_layout.addWidget(label, i, 0)
            torque_layout.addWidget(bar, i, 1)
            self.joint_torque_labels.append(label)
            self.joint_torque_bars.append(bar)

        # Set layouts to group boxes
        self.position_group.setLayout(pos_layout)
        self.velocity_group.setLayout(vel_layout)
        self.torque_group.setLayout(torque_layout)

        # Add group boxes to tab layout
        layout.addWidget(self.position_group)
        layout.addWidget(self.velocity_group)
        layout.addWidget(self.torque_group)

        tab.setLayout(layout)
        return tab
        
    def create_base_tab(self):
        """Create the Base Status tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Position and velocity group box
        base_status_group = QGroupBox("Base Status")
        base_layout = QGridLayout()
        
        # Position labels
        pos_x_label = QLabel("X Position:")
        pos_y_label = QLabel("Y Position:")
        pos_z_label = QLabel("Z Position:")
        
        self.pos_x_value = QLabel("0.00 m")
        self.pos_y_value = QLabel("0.00 m")
        self.pos_z_value = QLabel("0.00 m")
        
        # Velocity labels
        vel_x_label = QLabel("X Velocity:")
        vel_y_label = QLabel("Y Velocity:")
        vel_z_label = QLabel("Z Velocity:")
        
        self.vel_x_value = QLabel("0.00 m/s")
        self.vel_y_value = QLabel("0.00 m/s")
        self.vel_z_value = QLabel("0.00 m/s")
        
        # Orientation labels
        roll_label = QLabel("Roll:")
        pitch_label = QLabel("Pitch:")
        yaw_label = QLabel("Yaw:")
        
        self.roll_value = QLabel("0.00°")
        self.pitch_value = QLabel("0.00°")
        self.yaw_value = QLabel("0.00°")
        
        # Add to layout
        base_layout.addWidget(pos_x_label, 0, 0)
        base_layout.addWidget(self.pos_x_value, 0, 1)
        base_layout.addWidget(pos_y_label, 1, 0)
        base_layout.addWidget(self.pos_y_value, 1, 1)
        base_layout.addWidget(pos_z_label, 2, 0)
        base_layout.addWidget(self.pos_z_value, 2, 1)
        
        base_layout.addWidget(vel_x_label, 3, 0)
        base_layout.addWidget(self.vel_x_value, 3, 1)
        base_layout.addWidget(vel_y_label, 4, 0)
        base_layout.addWidget(self.vel_y_value, 4, 1)
        base_layout.addWidget(vel_z_label, 5, 0)
        base_layout.addWidget(self.vel_z_value, 5, 1)
        
        base_layout.addWidget(roll_label, 6, 0)
        base_layout.addWidget(self.roll_value, 6, 1)
        base_layout.addWidget(pitch_label, 7, 0)
        base_layout.addWidget(self.pitch_value, 7, 1)
        base_layout.addWidget(yaw_label, 8, 0)
        base_layout.addWidget(self.yaw_value, 8, 1)
        
        base_status_group.setLayout(base_layout)
        
        # Trajectory plot
        trajectory_group = QGroupBox("Position Trajectory")
        traj_layout = QVBoxLayout()
        
        self.position_plot = pg.PlotWidget()
        self.position_plot.setBackground('w')
        self.position_plot.setTitle("Robot Position (X-Y)")
        self.position_plot.setLabel('left', "Y Position (m)")
        self.position_plot.setLabel('bottom', "X Position (m)")
        self.position_plot.showGrid(x=True, y=True)
        
        self.position_curve = self.position_plot.plot(pen=pg.mkPen('b', width=2))
        self.position_data_x = np.array([])
        self.position_data_y = np.array([])
        
        traj_layout.addWidget(self.position_plot)
        trajectory_group.setLayout(traj_layout)
        
        # Add groups to tab
        layout.addWidget(base_status_group)
        layout.addWidget(trajectory_group)
        
        tab.setLayout(layout)
        return tab
        
    def create_imu_tab(self):
        """Create the IMU Data tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Gyroscope group
        gyro_group = QGroupBox("Gyroscope (rad/s)")
        gyro_layout = QGridLayout()
        
        gyro_x_label = QLabel("X (Roll Rate):")
        gyro_y_label = QLabel("Y (Pitch Rate):")
        gyro_z_label = QLabel("Z (Yaw Rate):")
        
        self.gyro_x_value = QLabel("0.00")
        self.gyro_y_value = QLabel("0.00")
        self.gyro_z_value = QLabel("0.00")
        
        gyro_layout.addWidget(gyro_x_label, 0, 0)
        gyro_layout.addWidget(self.gyro_x_value, 0, 1)
        gyro_layout.addWidget(gyro_y_label, 1, 0)
        gyro_layout.addWidget(self.gyro_y_value, 1, 1)
        gyro_layout.addWidget(gyro_z_label, 2, 0)
        gyro_layout.addWidget(self.gyro_z_value, 2, 1)
        
        gyro_group.setLayout(gyro_layout)
        
        # Accelerometer group
        accel_group = QGroupBox("Accelerometer (m/s²)")
        accel_layout = QGridLayout()
        
        accel_x_label = QLabel("X:")
        accel_y_label = QLabel("Y:")
        accel_z_label = QLabel("Z:")
        
        self.accel_x_value = QLabel("0.00")
        self.accel_y_value = QLabel("0.00")
        self.accel_z_value = QLabel("0.00")
        
        accel_layout.addWidget(accel_x_label, 0, 0)
        accel_layout.addWidget(self.accel_x_value, 0, 1)
        accel_layout.addWidget(accel_y_label, 1, 0)
        accel_layout.addWidget(self.accel_y_value, 1, 1)
        accel_layout.addWidget(accel_z_label, 2, 0)
        accel_layout.addWidget(self.accel_z_value, 2, 1)
        
        accel_group.setLayout(accel_layout)
        
        # IMU Plot
        imu_plot_group = QGroupBox("IMU Data Plot")
        imu_plot_layout = QVBoxLayout()
        
        self.imu_plot = pg.PlotWidget()
        self.imu_plot.setBackground('w')
        self.imu_plot.setTitle("IMU Data")
        self.imu_plot.addLegend()
        self.imu_plot.showGrid(x=True, y=True)
        
        self.gyro_curves = [
            self.imu_plot.plot(pen=pg.mkPen('r', width=2), name="Gyro X"),
            self.imu_plot.plot(pen=pg.mkPen('g', width=2), name="Gyro Y"),
            self.imu_plot.plot(pen=pg.mkPen('b', width=2), name="Gyro Z"),
        ]
        
        self.accel_curves = [
            self.imu_plot.plot(pen=pg.mkPen('r', width=2, style=Qt.DashLine), name="Accel X"),
            self.imu_plot.plot(pen=pg.mkPen('g', width=2, style=Qt.DashLine), name="Accel Y"),
            self.imu_plot.plot(pen=pg.mkPen('b', width=2, style=Qt.DashLine), name="Accel Z"),
        ]
        
        self.imu_data_time = np.array([])
        self.gyro_data = [np.array([]) for _ in range(3)]
        self.accel_data = [np.array([]) for _ in range(3)]
        self.time_counter = 0
        
        imu_plot_layout.addWidget(self.imu_plot)
        imu_plot_group.setLayout(imu_plot_layout)
        
        # Add all groups to tab
        layout.addWidget(gyro_group)
        layout.addWidget(accel_group)
        layout.addWidget(imu_plot_group)
        
        tab.setLayout(layout)
        return tab
    
    def create_command_tab(self):
        """Create the Commands tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Command history group
        cmd_group = QGroupBox("Previous Joint Position Commands")
        cmd_layout = QGridLayout()
        
        self.cmd_labels = []
        for i in range(12):
            cmd_layout.addWidget(QLabel(f"Joint {i+1}:"), i, 0)
            label = QLabel("N/A")
            cmd_layout.addWidget(label, i, 1)
            self.cmd_labels.append(label)
            
        cmd_group.setLayout(cmd_layout)
        
        # Command plot
        cmd_plot_group = QGroupBox("Joint Position Commands")
        cmd_plot_layout = QVBoxLayout()
        
        self.cmd_plot = pg.PlotWidget()
        self.cmd_plot.setBackground('w')
        self.cmd_plot.setTitle("Joint Position Commands")
        self.cmd_plot.setLabel('left', "Position (rad)")
        self.cmd_plot.setLabel('bottom', "Joint Number")
        self.cmd_plot.showGrid(x=True, y=True)
        
        self.cmd_bar = pg.BarGraphItem(x=range(1, 13), height=[0]*12, width=0.6, brush='b')
        self.cmd_plot.addItem(self.cmd_bar)
        self.cmd_plot.setXRange(0, 13)
        
        cmd_plot_layout.addWidget(self.cmd_plot)
        cmd_plot_group.setLayout(cmd_plot_layout)
        
        # Add groups to tab
        layout.addWidget(cmd_group)
        layout.addWidget(cmd_plot_group)
        
        tab.setLayout(layout)
        return tab
    
    @pyqtSlot()
    def update_status(self):
        """Update all status displays with current robot data"""
        # Get latest data from robot communication
        joint_state = self.robot_comm.get_joint_state()
        base_state = self.robot_comm.get_base_state()
        euler_angles = self.robot_comm.get_euler_angles()
        prev_commands = self.robot_comm.get_previous_position_commands()
        
        # Update joint status tab
        if joint_state["positions"].numel() > 0:
            positions = joint_state["positions"].cpu().numpy()
            velocities = joint_state["velocities"].cpu().numpy()
            torques = joint_state["torques"].cpu().numpy()
            
            for i in range(min(12, len(positions))):
                # Update position display
                pos_deg = float(positions[i]) * 180.0 / np.pi
                self.joint_pos_labels[i].setText(f"{self.joint_names[i]}: {positions[i]:.2f} rad")
                self.joint_pos_bars[i].setValue(int(pos_deg))
                
                # Update velocity display
                self.joint_vel_labels[i].setText(f"{self.joint_names[i]}: {velocities[i]:.2f} rad/s")
                vel_scaled = min(10, max(-10, float(velocities[i])))
                self.joint_vel_bars[i].setValue(int(vel_scaled * 10))
                
                # Update torque display
                self.joint_torque_labels[i].setText(f"{self.joint_names[i]}: {torques[i]:.2f} N·m")
                torque_scaled = min(20, max(-20, float(torques[i])))
                self.joint_torque_bars[i].setValue(int(torque_scaled * 10))
        
        # Update base status tab
        if base_state["position"].numel() > 0:
            position = base_state["position"].cpu().numpy()
            velocity = base_state["velocity"].cpu().numpy()
            
            self.pos_x_value.setText(f"{position[0]:.3f} m")
            self.pos_y_value.setText(f"{position[1]:.3f} m")
            self.pos_z_value.setText(f"{position[2]:.3f} m")
            
            self.vel_x_value.setText(f"{velocity[0]:.3f} m/s")
            self.vel_y_value.setText(f"{velocity[1]:.3f} m/s")
            self.vel_z_value.setText(f"{velocity[2]:.3f} m/s")
            
            # Update position plot
            self.position_data_x = np.append(self.position_data_x, position[0])
            self.position_data_y = np.append(self.position_data_y, position[1])
            
            # Limit plot data points
            max_points = 100
            if len(self.position_data_x) > max_points:
                self.position_data_x = self.position_data_x[-max_points:]
                self.position_data_y = self.position_data_y[-max_points:]
            
            self.position_curve.setData(self.position_data_x, self.position_data_y)
        
        # Update Euler angles
        if euler_angles.numel() > 0:
            euler = euler_angles.cpu().numpy()
            
            # Convert to degrees
            roll_deg = float(euler[0]) * 180.0 / np.pi
            pitch_deg = float(euler[1]) * 180.0 / np.pi
            yaw_deg = float(euler[2]) * 180.0 / np.pi
            
            self.roll_value.setText(f"{roll_deg:.2f}°")
            self.pitch_value.setText(f"{pitch_deg:.2f}°")
            self.yaw_value.setText(f"{yaw_deg:.2f}°")
        
        # Update IMU data tab
        if base_state["gyroscope"].numel() > 0:
            gyro = base_state["gyroscope"].cpu().numpy()
            accel = base_state["accelerometer"].cpu().numpy()
            
            self.gyro_x_value.setText(f"{gyro[0]:.3f}")
            self.gyro_y_value.setText(f"{gyro[1]:.3f}")
            self.gyro_z_value.setText(f"{gyro[2]:.3f}")
            
            self.accel_x_value.setText(f"{accel[0]:.3f}")
            self.accel_y_value.setText(f"{accel[1]:.3f}")
            self.accel_z_value.setText(f"{accel[2]:.3f}")
            
            # Update IMU plot
            self.time_counter += 1
            self.imu_data_time = np.append(self.imu_data_time, self.time_counter/10.0)  # in seconds
            
            for i in range(3):
                self.gyro_data[i] = np.append(self.gyro_data[i], gyro[i])
                self.accel_data[i] = np.append(self.accel_data[i], accel[i])
            
            # Limit plot data points
            max_points = 100
            if len(self.imu_data_time) > max_points:
                self.imu_data_time = self.imu_data_time[-max_points:]
                for i in range(3):
                    self.gyro_data[i] = self.gyro_data[i][-max_points:]
                    self.accel_data[i] = self.accel_data[i][-max_points:]
            
            # Update the curves
            for i in range(3):
                self.gyro_curves[i].setData(self.imu_data_time, self.gyro_data[i])
                self.accel_curves[i].setData(self.imu_data_time, self.accel_data[i])
        
        # Update command tab
        if prev_commands is not None:
            commands = prev_commands.cpu().numpy()
            
            # Update command labels
            for i in range(min(12, len(commands))):
                self.cmd_labels[i].setText(f"{commands[i]:.3f} rad")
            
            # Update command bar graph
            self.cmd_bar.setOpts(height=commands[:12])
    
    def toggle_updates(self):
        """Toggle status updates on/off"""
        if self.timer.isActive():
            self.timer.stop()
            self.toggle_button.setText("Resume Updates")
        else:
            self.timer.start(100)
            self.toggle_button.setText("Pause Updates")


def main():
    # Initialize DDS communication with real hardware interface
    ChannelFactoryInitialize(0, "enp108s0")
    
    # Create robot communication that will connect to real hardware
    robot_comm = RobotCommunication(device="cuda")

    # Create application
    app = QApplication(sys.argv)
    
    # Create robot communication
    robot_comm = RobotCommunication(device="cuda")
    
    # Create and show the status monitor
    monitor = StatusMonitor(robot_comm)
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()