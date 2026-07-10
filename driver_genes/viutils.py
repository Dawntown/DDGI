import matplotlib.colors as mcolors
import colorsys
import seaborn as sns
import pandas as pd
import re

def adjust_color(color, h_factor=1.0, s_factor=1.0, l_factor=1.0, return_type='rgb'):
    """
    Adjust the color by changing the hue, saturation, and lightness, and ensure the output is a valid RGB color.
    Args:
        color: The color, can be a string or an RGB tuple.
        h_factor: Hue adjustment factor. 1.0 means no change, 0.5 halves the hue, 2.0 doubles it.
        s_factor: Saturation adjustment factor. 1.0 means no change, 0.5 halves the saturation, 2.0 doubles it.
        l_factor: Lightness adjustment factor. 1.0 means no change, 0.5 halves the lightness, 2.0 doubles it.
    Returns:
        new_rgb: The adjusted color as an RGB tuple, with each value in [0, 1].
    """
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # Adjust hue and wrap it into [0,1)
    h_new = (h * h_factor) % 1.0
    # Adjust saturation and lightness, then clip to [0,1]
    s_new = min(max(s * s_factor, 0.0), 1.0)
    l_new = min(max(l * l_factor, 0.0), 1.0)
    new_rgb = colorsys.hls_to_rgb(h_new, l_new, s_new)
    # Ensure each RGB component is within [0,1]
    new_rgb = tuple(min(max(c, 0.0), 1.0) for c in new_rgb)
    if return_type == 'rgb':
        return new_rgb
    elif return_type == 'hex':
        return mcolors.to_hex(new_rgb)
    else:
        raise ValueError(f"Invalid return type: {return_type}")




benchmark_palette = pd.DataFrame([
    {'hex': '#A67EB7', 'name': 'Ours'},
    {'hex': '#6F94CD', 'name': 'CellNavi'},
    {'hex': '#5DA39D', 'name': 'GEARS'},
    {'hex': '#EF845D', 'name': 'CPA'},
    {'hex': '#9AA7B1', 'name': 'DEG'},
])


def get_palette(palette_name='benchmark', return_type='cmap'):
    if palette_name == 'benchmark':
        df = benchmark_palette
    else:
        raise ValueError(f"Invalid palette name: {palette_name}")
    if return_type == 'cmap':
        return mcolors.ListedColormap(df['hex'].tolist(), name=palette_name)
    elif return_type == 'df':
        return df
    else:
        raise ValueError(f"Invalid return type: {return_type}")
    

def valid_filename(name):
    return re.sub(r'[^a-zA-Z0-9]', '_', name).lower()