import os, sys, io, subprocess
import numpy as np
import matplotlib.pyplot as plt

from .logging_utils import log


def _ensure_pysheds():
    try:
        from pysheds.grid import Grid; return Grid
    except ImportError:
        log('pysheds not found — installing...', 'warn')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pysheds', 'rasterio'], check=False)
        try:
            from pysheds.grid import Grid; log('pysheds installed OK', 'ok'); return Grid
        except ImportError:
            log('pysheds install failed', 'error'); return None


def _ensure_rasterio():
    try:
        import rasterio; return rasterio
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rasterio'], check=False)
        import rasterio; return rasterio


def path_to_rasterio(src):
    src = src.strip()
    return f'/vsicurl/{src}' if src.startswith('http://') or src.startswith('https://') else src


def render_dem_overview(rio_path, max_pixels=600):
    rasterio = _ensure_rasterio()
    with rasterio.open(rio_path) as src:
        full_w, full_h = src.width, src.height
        scale = max(full_w, full_h) / max_pixels
        out_w = max(1, int(full_w/scale)) if scale > 1 else full_w
        out_h = max(1, int(full_h/scale)) if scale > 1 else full_h
        data = src.read(1, out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.average).astype(float)
        nodata = src.nodata
        if nodata is not None: data[data == nodata] = np.nan
        profile = {'crs': src.crs, 'bounds': src.bounds, 'transform': src.transform,
                   'width': src.width, 'height': src.height, 'nodata': nodata}
    return data, profile


def hillshade(arr, azimuth=315, altitude=45):
    az = np.radians(360 - azimuth); alt = np.radians(altitude)
    dy, dx = np.gradient(np.where(np.isnan(arr), 0, arr))
    slope = np.arctan(np.sqrt(dx**2 + dy**2)); aspect = np.arctan2(-dy, dx)
    hs = np.sin(alt)*np.cos(slope) + np.cos(alt)*np.sin(slope)*np.cos(az - aspect)
    hs = np.clip(hs, 0, 1); hs[np.isnan(arr)] = np.nan
    return hs


def pixel_to_lonlat(col_frac, row_frac, bounds, crs):
    from pyproj import Transformer
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    x = left + col_frac * (right - left)
    y = top  - row_frac * (top - bottom)
    epsg = crs.to_epsg()
    if epsg == 4326 or (abs(x) <= 180 and abs(y) <= 90): return x, y
    tr = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    return tr.transform(x, y)


def render_to_png(data, profile, src_name, outlet_lat=None, outlet_lon=None):
    hs = hillshade(data)
    valid = data[~np.isnan(data)]
    vmin, vmax = (float(valid.min()), float(valid.max())) if valid.size else (0, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
    norm_data = (data - vmin) / max(vmax - vmin, 1)
    rgba = plt.cm.terrain(norm_data); rgba[np.isnan(data)] = [0.05, 0.05, 0.05, 1]
    ax.imshow(rgba, origin='upper', aspect='equal')
    hs_display = np.where(np.isnan(hs), 0, hs)
    ax.imshow(hs_display, cmap='gray', alpha=0.4, origin='upper', aspect='equal', vmin=0, vmax=1)
    ax.set_title(f'{os.path.basename(src_name)} | Elev {vmin:.0f}–{vmax:.0f} m',
                 color='#cdd9e5', fontsize=9, pad=6)
    ax.tick_params(colors='#484f58', labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
    if outlet_lat is not None and outlet_lon is not None:
        b = profile['bounds']; crs = profile['crs']
        try:
            from pyproj import Transformer
            if crs.to_epsg() != 4326:
                tr = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
                x, y = tr.transform(outlet_lon, outlet_lat)
            else:
                x, y = outlet_lon, outlet_lat
            h, w = data.shape
            col = (x - b.left) / (b.right - b.left) * w
            row = (b.top - y) / (b.top - b.bottom) * h
            ax.plot(col, row, 'r+', markersize=14, markeredgewidth=2.5)
            ax.plot(col, row, 'ro', markersize=6, fillstyle='none', markeredgewidth=1.5)
            ax.annotate(' outlet', (col, row), color='red', fontsize=8, xytext=(6, -6), textcoords='offset points')
        except Exception: pass
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, facecolor='#0d1117')
    buf.seek(0); plt.close(fig); return buf.read(), vmin, vmax
