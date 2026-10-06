import numpy as np

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
