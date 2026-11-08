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
        window_type = str(window_type).lower()
        if window_type == "none" or window_type == "false":
            return np.ones(self.n_sensors)

        radii = np.sqrt(self.x**2 + self.y**2)
        max_r = np.max(radii)
        if max_r == 0:
            return np.ones(self.n_sensors)

        # Map r to an equivalent angle theta from 0 to pi
        # At r=0, theta = 0; at r=max_r, theta = pi
        theta = np.pi * (radii / max_r)

        if window_type == "blackman" or window_type == "true":
            # w(theta) = 0.42 + 0.5*cos(theta) + 0.08*cos(2*theta)
            return 0.42 + 0.5 * np.cos(theta) + 0.08 * np.cos(2 * theta)

        elif window_type == "hanning":
            # standard Hanning mapped to center (0.5 + 0.5*cos(theta))
            return 0.5 + 0.5 * np.cos(theta)

        elif window_type == "hamming":
            # standard Hamming mapped to center (0.54 + 0.46*cos(theta))
            return 0.54 + 0.46 * np.cos(theta)

        raise ValueError(f"Unknown window type: {window_type}")

import ast

def load_array(filepath) -> ArrayGeometry:
    """
    Loads array geometry from a file.
    Supports:
    1. .csv format: rows of coordinates. We extract the first two columns as x and y.
    2. .txt format: expects format like x=[...] and y=[...]
    """
    import os
    import re
    import ast
    import pandas as pd

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        # Load CSV using pandas
        # Need to support shape variations:
        # (N_sensors rows x 3 columns) OR (3 rows x N_sensors columns)
        try:
            df = pd.read_csv(filepath, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='all') # drop fully empty rows

            if df.shape[0] == 3 and df.shape[1] > 3:
                # Format is 3 rows (X, Y, Z) and N columns (sensors)
                x_vals = df.iloc[0, :].tolist()
                y_vals = df.iloc[1, :].tolist()
            else:
                # Format is N rows (sensors) and >=2 columns (X, Y, ...)
                df = df.dropna(subset=[0, 1])
                x_vals = df.iloc[:, 0].tolist()
                y_vals = df.iloc[:, 1].tolist()

            return ArrayGeometry(x_vals, y_vals)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV array geometry: {e}")

    # Fallback to standard .txt parsing
    with open(filepath, 'r') as f:
        content = f.read()

    x_match = re.search(r'x\s*=\s*(\[.*?\])', content, flags=re.DOTALL)
    y_match = re.search(r'y\s*=\s*(\[.*?\])', content, flags=re.DOTALL)

    if x_match and y_match:
        x_str = re.sub(r'\s+', '', x_match.group(1))
        y_str = re.sub(r'\s+', '', y_match.group(1))
        x_vals = ast.literal_eval(x_str)
        y_vals = ast.literal_eval(y_str)
    else:
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
