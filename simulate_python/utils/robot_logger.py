import os
import csv
import time
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class RobotLogger:
    def __init__(self, log_dir="logs", log_frequency=1, flush_frequency=50):
        """
        Initialize a data logger for robot data
        
        Args:
            log_dir: Directory to save log files
            log_frequency: How often to log data (every N steps)
            flush_frequency: How often to flush data to disk (every N logs)
        """
        self.log_dir = Path(log_dir)
        self.log_frequency = log_frequency
        self.flush_frequency = flush_frequency
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate a unique filename with timestamp
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = self.log_dir / f"robot_log_{self.timestamp}.csv"
        self.metadata_path = self.log_dir / f"metadata_{self.timestamp}.json"
        
        # Data buffers
        self.data_buffer = []
        self.header = None
        
        # Counters
        self.step_counter = 0
        self.log_counter = 0
        
        # Start time
        self.start_time = time.time()
        
        # Metadata for later analysis
        self.metadata = {
            "start_time": self.timestamp,
            "columns": [],
            "log_frequency": log_frequency,
        }
        
        print(f"Logger initialized. Data will be saved to {self.log_path}")
    
    def _process_value(self, value):
        """Process values to prepare them for logging"""
        # Handle torch tensors
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        # Handle numpy arrays
        elif isinstance(value, np.ndarray):
            return value
        # Return other types as is
        return value
    
    def log(self, **kwargs):
        """
        Log named values
        
        Args:
            **kwargs: Named values to log (name=value)
        """
        # Only log at specified frequency
        self.step_counter += 1
        if self.step_counter % self.log_frequency != 0:
            return
        
        # Track log count
        self.log_counter += 1
        
        # Get current time
        current_time = time.time() - self.start_time
        
        # Flatten tensors/arrays if needed
        flattened_data = {"time": current_time, "step": self.step_counter}
        
        for key, value in kwargs.items():
            processed_value = self._process_value(value)
            
            # Handle scalar values directly
            if np.isscalar(processed_value) or isinstance(processed_value, (int, float)):
                flattened_data[key] = processed_value
            # Handle arrays/tensors
            elif hasattr(processed_value, "shape"):
                if processed_value.size == 1:  # Scalar array/tensor
                    flattened_data[key] = float(processed_value)
                else:  # Multi-dimensional array/tensor
                    flat_val = processed_value.flatten()
                    for i, v in enumerate(flat_val):
                        flattened_data[f"{key}_{i}"] = float(v)
        
        # Initialize header if not set
        if self.header is None:
            self.header = list(flattened_data.keys())
            self.metadata["columns"] = self.header
            
            # Write header to file
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.header)
                writer.writeheader()
            
            # Write metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        
        # Add data to buffer
        self.data_buffer.append(flattened_data)
        
        # Flush to disk if needed
        if self.log_counter % self.flush_frequency == 0:
            self.flush()
    
    def flush(self):
        """Force write all buffered data to disk"""
        if not self.data_buffer:
            return
            
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerows(self.data_buffer)
        
        # Clear buffer after writing
        self.data_buffer = []
    
    def close(self):
        """Close the logger and ensure all data is saved"""
        self.flush()
        # Update metadata with end time
        self.metadata["end_time"] = time.strftime("%Y%m%d-%H%M%S")
        self.metadata["total_steps"] = self.step_counter
        self.metadata["total_logs"] = self.log_counter
        
        # Write updated metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Logger closed. Logged {self.log_counter} entries from {self.step_counter} steps.")