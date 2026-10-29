import numpy as np

def generate_log_spiral_array(n_mics, diameter):
    """
    Generates a logarithmic spiral array.
    """
    a = 0.1
    b = 0.2
    theta = np.linspace(0, 4 * np.pi, n_mics)
    r = a * np.exp(b * theta)

    # Scale to desired diameter
    max_r = np.max(r)
    r_scaled = r * (diameter / 2.0) / max_r

    x = r_scaled * np.cos(theta)
    y = r_scaled * np.sin(theta)
    z = np.zeros_like(x)

    return np.stack([x, y, z], axis=1) # Shape: (n_mics, 3)

def generate_random_planar_array(n_mics, diameter):
    """
    Generates a dual-concentric random planar array.
    """
    r = np.random.uniform(0.1, diameter / 2.0, n_mics)
    theta = np.random.uniform(0, 2 * np.pi, n_mics)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)

def get_array_geometry(n_mics=50, diameter=2.5, array_type="log_spiral"):
    if array_type == "log_spiral":
        return generate_log_spiral_array(n_mics, diameter)
    else:
        return generate_random_planar_array(n_mics, diameter)
