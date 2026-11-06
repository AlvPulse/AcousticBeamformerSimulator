import re

with open("acoustic_sim/dsp_array.py", "r") as f:
    content = f.read()

new_load_array = r"""def load_array(filepath) -> ArrayGeometry:
    import re
    import ast
    with open(filepath, 'r') as f:
        content = f.read()

    # Try parsing old format (x=[...], y=[...])
    x_match = re.search(r'x\s*=\s*(\[.*?\])', content, flags=re.DOTALL)
    y_match = re.search(r'y\s*=\s*(\[.*?\])', content, flags=re.DOTALL)
    z_match = re.search(r'z\s*=\s*(\[.*?\])', content, flags=re.DOTALL)

    if x_match and y_match:
        x_str = re.sub(r'\s+', '', x_match.group(1))
        y_str = re.sub(r'\s+', '', y_match.group(1))
        x_vals = ast.literal_eval(x_str)
        y_vals = ast.literal_eval(y_str)

        geo = ArrayGeometry(x_vals, y_vals)
        if z_match:
            z_str = re.sub(r'\s+', '', z_match.group(1))
            z_vals = ast.literal_eval(z_str)
            geo.z = np.array(z_vals)
            geo.positions = np.column_stack((geo.x, geo.y, geo.z))

        return geo

    # Try parsing CSV tab-separated format
    lines = content.strip().split('\n')
    if len(lines) >= 2:
        try:
            x_vals = [float(v) for v in lines[0].strip().split()]
            y_vals = [float(v) for v in lines[1].strip().split()]

            geo = ArrayGeometry(x_vals, y_vals)

            if len(lines) >= 3 and lines[2].strip():
                z_vals = [float(v) for v in lines[2].strip().split()]
                if len(z_vals) == len(x_vals):
                    geo.z = np.array(z_vals)
                    geo.positions = np.column_stack((geo.x, geo.y, geo.z))

            return geo
        except Exception as e:
            raise ValueError(f"Could not parse x and y coordinates from {filepath}: {e}")

    raise ValueError(f"Could not parse x and y coordinates from {filepath}")
"""

content = re.sub(r'def load_array\(filepath\) -> ArrayGeometry:.*?return ArrayGeometry\(x_vals, y_vals\)', new_load_array.replace('\\', '\\\\'), content, flags=re.DOTALL)

with open("acoustic_sim/dsp_array.py", "w") as f:
    f.write(content)
