import numpy as np

class ArrayGeometry:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n_sensors = len(self.x)
        if len(self.y) != self.n_sensors:
            raise ValueError("x and y arrays must have the same length")

        # Center the array
        self.x -= np.mean(self.x)
        self.y -= np.mean(self.y)
        self.z = np.zeros(self.n_sensors) # z = 0 (Planar array)
        self.positions = np.column_stack((self.x, self.y, self.z))

    def get_spatial_weights(self, window_type="none"):
        """
        Returns spatial weights for the array elements.
        If window_type == "blackman", computes radial distance from centroid
        and applies a Blackman window.
        """
        if window_type == "none" or window_type is False or window_type is None:
            return np.ones(self.n_sensors)

        if window_type == "blackman" or window_type is True:
            # Radial distance from centroid
            radii = np.sqrt(self.x**2 + self.y**2)
            max_r = np.max(radii)
            if max_r == 0:
                return np.ones(self.n_sensors)

            # Normalize radii to [-0.5, 0.5] to fit a typical window function width
            # Standard blackman window: w(n) = 0.42 - 0.5*cos(2*pi*n/N) + 0.08*cos(4*pi*n/N)
            # For a radial window, we want the peak (1.0) at radius 0, and tapering to 0 at max_r.
            # So we map r=0 to the center of the Blackman window, and r=max_r to the edge.

            # Map r to an equivalent angle theta from 0 to pi
            # At r=0, theta = 0; at r=max_r, theta = pi
            theta = np.pi * (radii / max_r)

            # A 1D blackman window mapped to half-space (center to edge)
            # Center of standard blackman is at n=N/2, which corresponds to the peak.
            # We use the formula shifted such that peak is at 0.
            # w(theta) = 0.42 + 0.5*cos(theta) + 0.08*cos(2*theta)
            weights = 0.42 + 0.5 * np.cos(theta) + 0.08 * np.cos(2 * theta)
            return weights

        raise ValueError(f"Unknown window type: {window_type}")

def load_array(filepath) -> ArrayGeometry:
    """
    Loads array geometry from a text file.
    Expects lines like:
    x=[...]
    y=[...]
    """
    x_vals = []
    y_vals = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('x='):
                # parse the list
                val_str = line.split('=', 1)[1]
                x_vals = eval(val_str)
            elif line.startswith('y='):
                val_str = line.split('=', 1)[1]
                y_vals = eval(val_str)

    if not x_vals or not y_vals:
        raise ValueError(f"Could not parse x and y coordinates from {filepath}")

    return ArrayGeometry(x_vals, y_vals)

def aliasing_frequency(array: ArrayGeometry, c=343.0) -> float:
    """
    Computes the theoretical spatial-aliasing frequency directly from the actual (x,y) layout.
    Finds the minimum distance between any two sensors.
    f_alias = c / (2 * d_min)
    """
    pos = array.positions
    # compute pairwise distances
    diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=-1)

    # ignore self-distance (0 on diagonal)
    np.fill_diagonal(distances, np.inf)

    d_min = np.min(distances)
    if d_min == 0:
        return np.inf # Sensors on top of each other

    return c / (2.0 * d_min)
