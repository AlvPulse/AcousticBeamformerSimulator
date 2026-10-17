import numpy as np
import ast

def load_array_geometry(filepath):
    """
    Loads array geometry from a text file defining x and y (and optionally z) coordinates.
    Example format:
    x=[-0.15, -0.05, 0.05, 0.15]
    y=[-0.15, -0.15, -0.15, -0.15]
    """
    coords = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip().lower()
            try:
                # Use literal_eval to safely parse lists
                coords[key] = np.array(ast.literal_eval(val.strip()))
            except (ValueError, SyntaxError):
                pass

    if 'x' not in coords or 'y' not in coords:
        raise ValueError("Array geometry file must contain 'x' and 'y' arrays.")

    n_sensors = len(coords['x'])
    positions = np.zeros((n_sensors, 3))
    positions[:, 0] = coords['x']
    positions[:, 1] = coords['y']
    if 'z' in coords:
        positions[:, 2] = coords['z']

    # Center the array
    positions -= np.mean(positions, axis=0)
    return positions

def generate_log_spiral_array(n_sensors=50, aperture_m=2.5, num_arms=5):
    """
    Generates a logarithmic spiral array to minimize spatial aliasing
    over a wide broadband range (100 Hz - 8000 Hz).
    """
    theta = np.linspace(0, 4 * np.pi, n_sensors)
    a = 0.05
    b = 0.2

    # Generate radii based on log-spiral equation
    r = a * np.exp(b * theta)

    # Scale to desired aperture diameter
    max_r = np.max(r)
    r = r * (aperture_m / 2) / max_r

    # Ensure uniform or interleaved distribution by creating arms
    positions = np.zeros((n_sensors, 3))

    for i in range(n_sensors):
        arm = i % num_arms
        angle_offset = arm * (2 * np.pi / num_arms)

        x = r[i] * np.cos(theta[i] + angle_offset)
        y = r[i] * np.sin(theta[i] + angle_offset)

        positions[i, 0] = x
        positions[i, 1] = y
        positions[i, 2] = 0.0 # Planar array at z=0

    # Ensure centered at origin
    positions -= np.mean(positions, axis=0)

    return positions
