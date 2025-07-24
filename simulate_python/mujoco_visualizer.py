import numpy as np
import mujoco

class MujocoVisualizer:
    """Class for managing visualizations in MuJoCo scenes"""

    GREEN = [0, 1, 0, 1]
    RED = [1, 0, 0, 1]
    BLUE = [0, 0, 1, 1]
    DEFAULT_ARROW_SIZE = [0.05, 0.05, 1.0]  # shaft_length, shaft_width, head_width
    
    def __init__(self, scene):
        """Initialize the visualizer with an empty arrow buffer"""
        self.arrow_buffer = []  # Store arrow definitions to be rendered
        self.geom_start_idx = 0  # Store starting index for our geoms
        self.scene = scene

    def add_arrow(self, start, end, size=DEFAULT_ARROW_SIZE, color=GREEN):
        """Add an arrow to the buffer to be rendered on next frame
        
        Args:
            start: 3D position where the arrow starts
            end: 3D position where the arrow ends
            size: Arrow size parameters [shaft_length, shaft_width, head_width]
            color: RGBA color values [r, g, b, a]
        """
        self.arrow_buffer.append({
            'start': np.array(start, dtype=np.float32),
            'end': np.array(end, dtype=np.float32),
            'size': np.array(size, dtype=np.float32),
            'color': np.array(color, dtype=np.float32)
        })
    
    def clear_buffer(self):
        """Clear all arrows from the buffer"""
        self.arrow_buffer = []
        # Reset the scene's geom count to remove rendered arrows
        if self.scene is not None:
            self.scene.ngeom = self.geom_start_idx
    
    def render(self):
        """Render all arrows in the buffer to the MuJoCo scene
        
        Args:
            scene: MuJoCo scene object to render arrows into
        """
        if self.scene is None:
            return
            
        # Store the current scene geom count as our starting point
        # This allows other code to add geoms without being affected
        self.geom_start_idx = self.scene.ngeom
        
        # Render all arrows in the buffer
        for arrow in self.arrow_buffer:
            self._add_arrow_to_scene(
                arrow['start'], 
                arrow['end'], 
                arrow['size'], 
                arrow['color']
            )
    
    def _add_arrow_to_scene(self, start, end, size, color):
        """Add an arrow to the MuJoCo scene"""
        if self.scene.ngeom >= self.scene.maxgeom:
            return  # No space left in scene geoms
            
        # Get a reference to the next free geom
        arrow_geom = self.scene.geoms[self.scene.ngeom]
        self.scene.ngeom += 1
        
        # Configure the geom
        arrow_geom.type = mujoco.mjtGeom.mjGEOM_ARROW
        arrow_geom.pos = start
        arrow_geom.size = size
        arrow_geom.rgba = color
        
        # Calculate direction
        direction = end - start
        length = np.linalg.norm(direction)
        if length > 0.001:
            direction = direction / length
            
            # Create rotation matrix (align arrow with direction)
            z_axis = direction
            y_axis = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.95 else np.array([0, 1, 0])
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Set rotation matrix
            mat = np.column_stack([x_axis, y_axis, z_axis])
            for i in range(3):
                for j in range(3):
                    arrow_geom.mat[i, j] = mat[i, j]