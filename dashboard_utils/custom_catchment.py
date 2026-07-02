"""
Custom Catchment Delineation tab — dark theme.
"""

import os, sys, tempfile, zipfile, shutil, threading, traceback, math

import numpy as np
import geopandas as gpd
import ipywidgets as widgets
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree
from ipyleaflet import Map, GeoJSON, Marker, DrawControl, Rectangle, basemaps, basemap_to_tiles

from .logging_utils import log
from .dem_utils import _ensure_pysheds, path_to_rasterio

if not hasattr(np, 'in1d'):
    np.in1d = np.isin

# ── Dark palette (mirrors main dashboard) ───────────────────────────────────
_BG0  = '#020617'
_BG1  = '#0f172a'
_BG2  = '#1e293b'
_BDR  = '#334155'
_BDR2 = '#1e293b'
_TEXT = '#e2e8f0'
_MUTED = '#475569'
_DIM   = '#334155'


def _status_warn(msg):
    return (f'<div style="background:#422006;border-left:4px solid #d97706;'
            f'padding:8px 12px;font-size:12px;color:#fbbf24;border-radius:0 6px 6px 0">'
            f'{msg}</div>')

def _status_ok(msg):
    return (f'<div style="background:#052e16;border-left:4px solid #16a34a;'
            f'padding:8px 12px;font-size:12px;color:#4ade80;border-radius:0 6px 6px 0">'
            f'{msg}</div>')

def _card_layout():
    return widgets.Layout(
        border=f'1px solid {_BDR}', border_radius='8px',
        padding='12px', margin='6px 0',
        background_color=_BG2,
    )

def _h(txt):
    return widgets.HTML(txt)


def _read_elevation(rio_path, lat, lon):
    import rasterio
    from rasterio.warp import transform as warp_transform
    with rasterio.open(rio_path) as src:
        if src.crs.to_epsg() != 4326:
            xs, ys = warp_transform('EPSG:4326', src.crs, [lon], [lat])
        else:
            xs, ys = [lon], [lat]
        left, bottom, right, top = src.bounds
        if not (left <= xs[0] <= right and bottom <= ys[0] <= top):
            return None
        row, col = src.index(xs[0], ys[0])
        row = max(0, min(row, src.height - 1))
        col = max(0, min(col, src.width  - 1))
        val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
        nodata = src.nodata
    if nodata is not None and val == nodata:
        return None
    return float(val)


def _clip_dem_to_buffer(rio_path, lat, lon, buffer_km, out_path):
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    buf_deg_lat = buffer_km / 110.54
    buf_deg_lon = buffer_km / (111.32 * math.cos(math.radians(lat)))

    with rasterio.open(rio_path) as src:
        if src.crs.to_epsg() == 4326:
            minx = lon - buf_deg_lon; maxx = lon + buf_deg_lon
            miny = lat - buf_deg_lat; maxy = lat + buf_deg_lat
        else:
            corners_lon = [lon - buf_deg_lon, lon + buf_deg_lon,
                           lon - buf_deg_lon, lon + buf_deg_lon]
            corners_lat = [lat - buf_deg_lat, lat - buf_deg_lat,
                           lat + buf_deg_lat, lat + buf_deg_lat]
            xs, ys = warp_transform('EPSG:4326', src.crs, corners_lon, corners_lat)
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)

        minx = max(minx, src.bounds.left);   maxx = min(maxx, src.bounds.right)
        miny = max(miny, src.bounds.bottom); maxy = min(maxy, src.bounds.top)

        win = from_bounds(minx, miny, maxx, maxy, src.transform)
        win = win.round_lengths().round_offsets()
        data = src.read(1, window=win)
        win_transform = src.window_transform(win)

        profile = src.profile.copy()
        profile.update(width=data.shape[1], height=data.shape[0],
                       transform=win_transform, compress='lzw')

        res_x = abs(win_transform.a); res_y = abs(win_transform.e)
        m_per_deg_lon = 111320 * math.cos(math.radians(lat))
        cell_m = (res_x * m_per_deg_lon + res_y * 110540) / 2
        total_cells = data.shape[0] * data.shape[1]

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data, 1)

    return out_path, cell_m, total_cells


def _dem_wgs84_extent(rio_path):
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(rio_path) as src:
        b = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
    return b


def _polygon_to_geojson(polygon):
    return {
        'type': 'FeatureCollection',
        'features': [{'type': 'Feature',
                      'geometry': mapping(polygon),
                      'properties': {}}],
    }


def build_tab(state, refs):
    _cc = {
        'dem_path': None, 'outlet_lat': None, 'outlet_lon': None,
        'catchment_geom': None, 'user_polygon': None,
        'catch_layer': None, 'dem_rect': None, 'outlet_marker': None,
        'click_armed': False, 'wgs84_extent': None,
        'cell_size_m': 90, '_fdir_path': None, '_facc_path': None,
    }

    # ── status bar ────────────────────────────────────────────────────────────
    cc_status = widgets.HTML(
        value=_status_warn('<b>Step 1:</b> Load a DEM file to enable elevation queries.')
    )

    # ── height system (mirrors main dashboard) ───────────────────────────────
    # _CHROME=148  JupyterLab menu+toolbar+tab-bar+status-bar
    # _ROOT_PAD=28 outer VBox padding
    # _LTAB=36     top-level tab bar
    # _HDR=44      custom-catchment header + status strip
    _CHROME   = 148
    _ROOT_PAD = 28
    _LTAB     = 36
    _HDR      = 44
    _PANEL_H  = f'calc(100vh - {_CHROME + _ROOT_PAD + _LTAB}px)'   # full side height
    _MAP_H    = _PANEL_H                                             # map fills same height

    # ── map — CartoDB Dark Matter ─────────────────────────────────────────────
    m = Map(
        center=[64.5, 26.0], zoom=5,
        scroll_wheel_zoom=True,
        basemap=basemaps.CartoDB.DarkMatter,
        prefer_canvas=True,
        layout=widgets.Layout(width='100%', height='100%', min_height='300px'),
    )

    draw_ctrl = DrawControl(
        polygon={'shapeOptions': {'color': '#fb923c', 'fillOpacity': 0.12,
                                  'weight': 2, 'dashArray': '6 4'}},
        polyline={}, circlemarker={}, rectangle={}, circle={},
    )

    def _on_draw(change):
        geo = draw_ctrl.last_draw
        if geo and geo.get('geometry'):
            _cc['user_polygon'] = shape(geo['geometry'])
            cc_run_lbl.value = (
                f'<small style="color:#fb923c">✏️ Custom boundary drawn — '
                f'"Use this catchment" will apply your shape.</small>'
            )

    draw_ctrl.observe(_on_draw, names='last_draw')
    m.add(draw_ctrl)

    elev_lbl = widgets.HTML(
        value=f'<span style="font-family:monospace;font-size:13px;color:{_DIM}">'
              f'Load a DEM then click the map to read elevation.</span>'
    )

    # ── A: DEM source ─────────────────────────────────────────────────────────
    cc_src_input = widgets.Text(
        placeholder='/path/to/dem.tif  or  https://copernicus-dem-90m.s3.amazonaws.com/…',
        description='DEM file:',
        style={'description_width': '70px'},
        layout=widgets.Layout(width='100%'),
    )
    cc_load_btn = widgets.Button(description='Load DEM', button_style='info',
                                  layout=widgets.Layout(width='110px'))
    cc_load_lbl = widgets.HTML(value='')

    cog_hint = widgets.HTML(
        f'<details style="font-size:11px;color:{_MUTED};margin:6px 0">'
        f'<summary style="cursor:pointer;color:{_MUTED}">Free Copernicus 90 m COG URLs (each tile = 1°×1°)</summary>'
        f'<div style="margin-top:6px;line-height:1.8">'
        f'<b style="color:{_TEXT}">Oulu (N65 E025):</b><br>'
        f'<code style="color:#93c5fd;font-size:10px;word-break:break-all">'
        f'https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N65_00_E025_00_DEM/Copernicus_DSM_COG_30_N65_00_E025_00_DEM.tif'
        f'</code><br>'
        f'<b style="color:{_TEXT}">Tampere (N61 E023):</b><br>'
        f'<code style="color:#93c5fd;font-size:10px;word-break:break-all">'
        f'https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N61_00_E023_00_DEM/Copernicus_DSM_COG_30_N61_00_E023_00_DEM.tif'
        f'</code><br>'
        f'<b style="color:{_TEXT}">Helsinki (N60 E024):</b><br>'
        f'<code style="color:#93c5fd;font-size:10px;word-break:break-all">'
        f'https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N60_00_E024_00_DEM/Copernicus_DSM_COG_30_N60_00_E024_00_DEM.tif'
        f'</code>'
        f'</div></details>'
    )

    def _cc_load(b):
        src = cc_src_input.value.strip()
        if not src:
            cc_load_lbl.value = f'<small style="color:#ef4444">Enter a path or URL</small>'
            return
        cc_load_btn.disabled = True
        cc_load_lbl.value = f'<small style="color:{_MUTED}">Reading DEM metadata…</small>'

        def _run():
            try:
                rio_path = path_to_rasterio(src)
                ext = _dem_wgs84_extent(rio_path)
                minlon, minlat, maxlon, maxlat = ext
                _cc['dem_path'] = rio_path
                _cc['wgs84_extent'] = ext

                clat = (minlat + maxlat) / 2
                clon = (minlon + maxlon) / 2
                span = max(maxlat - minlat, maxlon - minlon)
                m.center = [clat, clon]
                m.zoom   = max(7, min(13, int(round(8 - math.log2(max(span, 0.001))))))

                # remove old DEM bounds rect if present
                if _cc['dem_rect'] is not None:
                    try: m.remove(_cc['dem_rect'])
                    except Exception: pass

                rect = Rectangle(
                    bounds=((minlat, minlon), (maxlat, maxlon)),
                    color='#60a5fa',        # blue-400 outline
                    fill_color='#1e3a5f',   # very dark blue fill
                    fill_opacity=0.08,
                    weight=2,
                    dash_array='8 5',       # dashed border
                )
                m.add(rect)
                _cc['dem_rect'] = rect

                cc_load_lbl.value = (
                    f'<small style="color:#4ade80">✔ DEM ready — covers '
                    f'<b>{minlat:.2f}°–{maxlat:.2f}°N, {minlon:.2f}°–{maxlon:.2f}°E</b>. '
                    f'Click anywhere on the map to read elevation.</small>'
                )
                cc_status.value = _status_ok(
                    f'<b>DEM loaded.</b> Covers {minlat:.2f}°–{maxlat:.2f}°N, '
                    f'{minlon:.2f}°–{maxlon:.2f}°E. '
                    f'Click <b>Place outlet on map</b>, click inside that area, then Delineate.'
                )
                elev_lbl.value = (
                    f'<span style="font-family:monospace;font-size:13px;color:#79c0ff">'
                    f'Click the map to read elevation ⛰</span>'
                )
                log(f'DEM loaded: {os.path.basename(src)} | '
                    f'{minlat:.2f}–{maxlat:.2f}°N, {minlon:.2f}–{maxlon:.2f}°E', 'ok')
            except Exception as exc:
                cc_load_lbl.value = f'<small style="color:#ef4444">Error: {exc}</small>'
                log(f'DEM load error: {exc}', 'error')
            finally:
                cc_load_btn.disabled = False

        threading.Thread(target=_run, daemon=True).start()

    cc_load_btn.on_click(_cc_load)

    # ── B: Place outlet ───────────────────────────────────────────────────────
    outlet_lbl = widgets.HTML(
        value=f'<small style="color:{_DIM}">No outlet set yet.</small>'
    )
    arm_btn = widgets.Button(description='📍 Place outlet on map',
                              button_style='warning',
                              layout=widgets.Layout(width='200px'))
    arm_lbl = widgets.HTML(value='')

    def _arm(b):
        if not _cc.get('dem_path'):
            arm_lbl.value = '<small style="color:#ef4444">Load a DEM first.</small>'
            return
        _cc['click_armed'] = True
        arm_btn.button_style = 'danger'
        arm_btn.description  = '🎯 Click the map now…'
        arm_lbl.value = f'<small style="color:#fbbf24">Click your outlet on the map.</small>'

    arm_btn.on_click(_arm)

    def _on_map_click(**kwargs):
        if kwargs.get('type') != 'click':
            return
        coords = kwargs.get('coordinates')
        if not coords:
            return
        lat, lon = coords[0], coords[1]

        rio_path = _cc.get('dem_path')
        if rio_path:
            def _query():
                try:
                    val = _read_elevation(rio_path, lat, lon)
                    if val is not None:
                        elev_lbl.value = (
                            f'<span style="font-family:monospace;font-size:13px;color:#79c0ff">'
                            f'⛰ <b>{val:.1f} m</b> @ {lat:.5f}°N {lon:.5f}°E</span>'
                        )
                    else:
                        elev_lbl.value = (
                            f'<span style="font-family:monospace;font-size:13px;color:{_DIM}">'
                            f'outside DEM — {lat:.5f}°N {lon:.5f}°E</span>'
                        )
                except Exception:
                    pass
            threading.Thread(target=_query, daemon=True).start()

        if not _cc['click_armed']:
            return

        ext = _cc.get('wgs84_extent')
        if ext:
            minlon, minlat, maxlon, maxlat = ext
            if not (minlon <= lon <= maxlon and minlat <= lat <= maxlat):
                outlet_lbl.value = (
                    f'<small style="color:#ef4444">⚠ Outside DEM '
                    f'({minlat:.2f}–{maxlat:.2f}°N, {minlon:.2f}–{maxlon:.2f}°E). '
                    f'Click inside that area.</small>'
                )
                return

        _cc['click_armed'] = False
        arm_btn.button_style = 'warning'
        arm_btn.description  = '📍 Place outlet on map'
        arm_lbl.value = ''

        _cc['outlet_lat']   = lat
        _cc['outlet_lon']   = lon
        _cc['user_polygon'] = None

        if _cc['outlet_marker'] is not None:
            try: m.remove(_cc['outlet_marker'])
            except Exception: pass

        marker = Marker(location=[lat, lon], draggable=True, title='Outlet — drag to fine-tune')
        marker.on_move(_on_marker_move)
        m.add(marker)
        _cc['outlet_marker'] = marker

        outlet_lbl.value = (
            f'<small style="color:#4ade80">📍 {lat:.5f}°N, {lon:.5f}°E — '
            f'drag pin to fine-tune, then Delineate.</small>'
        )

    def _on_marker_move(**kwargs):
        loc = kwargs.get('location') or []
        if len(loc) == 2:
            _cc['outlet_lat'], _cc['outlet_lon'] = loc[0], loc[1]
            outlet_lbl.value = (
                f'<small style="color:#4ade80">📍 {loc[0]:.5f}°N, {loc[1]:.5f}°E</small>'
            )

    m.on_interaction(_on_map_click)

    # ── C: Delineate ──────────────────────────────────────────────────────────
    buffer_input = widgets.BoundedFloatText(
        value=15.0, min=1.0, max=200.0, step=1.0,
        description='Buffer radius (km)',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='310px'),
    )
    buffer_info = widgets.HTML(
        value=f'<small style="color:{_MUTED}">Only this area around the outlet is loaded — '
              f'keeps RAM low for high-res DEMs. Re-run Prepare if you move the outlet far away.</small>'
    )

    snap_area_input = widgets.BoundedFloatText(
        value=2.0, min=0.1, max=500.0, step=0.5,
        description='Min upstream area (km²)',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='310px'),
    )
    snap_dist_input = widgets.BoundedIntText(
        value=2000, min=100, max=20000, step=100,
        description='Max snap distance (m)',
        style={'description_width': '180px'},
        layout=widgets.Layout(width='310px'),
    )
    param_info = widgets.HTML(value='')

    def _update_param_info(change=None):
        area   = snap_area_input.value
        dist   = snap_dist_input.value
        cell_m = _cc.get('cell_size_m', 90)
        min_cells = max(1, int(area * 1e6 / (cell_m ** 2)))
        param_info.value = (
            f'<small style="color:{_MUTED}">DEM cell ≈ {cell_m:.0f} m | '
            f'{area} km² = {min_cells:,} cells | '
            f'snap within {dist} m of click</small>'
        )

    snap_area_input.observe(_update_param_info, names='value')
    snap_dist_input.observe(_update_param_info, names='value')

    prep_btn  = widgets.Button(description='⚙ Prepare DEM (run once)', button_style='info',
                                layout=widgets.Layout(width='220px', height='36px'))
    prep_prog = widgets.IntProgress(value=0, min=0, max=100,
                                     layout=widgets.Layout(width='200px', visibility='hidden'))
    prep_lbl  = widgets.HTML(value='')

    def _cc_prepare(b):
        if not _cc.get('dem_path'):
            prep_lbl.value = '<small style="color:#ef4444">Load a DEM first</small>'
            return
        prep_btn.disabled = True
        cc_run_btn.disabled = True
        prep_prog.layout.visibility = 'visible'
        prep_prog.value = 5
        prep_lbl.value = f'<small style="color:{_MUTED}">Conditioning DEM…</small>'

        lat_now = _cc.get('outlet_lat')
        lon_now = _cc.get('outlet_lon')
        buf_km  = buffer_input.value

        def _run():
            try:
                Grid = _ensure_pysheds()
                if Grid is None:
                    raise ImportError('pysheds not available')

                if lat_now is not None and lon_now is not None:
                    prep_lbl.value = (
                        f'<small style="color:{_MUTED}">Clipping DEM to '
                        f'{buf_km:.0f} km buffer around outlet…</small>'
                    )
                    tmp_clip_dir = tempfile.mkdtemp(prefix='hbv_clip_')
                    clip_path = os.path.join(tmp_clip_dir, 'dem_clip.tif')
                    clip_path, cell_m, total_cells = _clip_dem_to_buffer(
                        _cc['dem_path'], lat_now, lon_now, buf_km, clip_path)
                    work_path = clip_path
                    log(f'Clipped to {buf_km:.0f} km buffer — '
                        f'{total_cells/1e6:.1f}M cells, cell ≈ {cell_m:.0f} m', 'info')
                else:
                    import rasterio
                    with rasterio.open(_cc['dem_path']) as src:
                        total_cells = src.width * src.height
                        res_x = abs(src.transform.a); res_y = abs(src.transform.e)
                        mid_lat = (src.bounds.bottom + src.bounds.top) / 2
                        cell_m = (res_x * 111320 * math.cos(math.radians(mid_lat))
                                  + res_y * 110540) / 2
                    if total_cells > 20_000_000:
                        log(f'⚠ No outlet set — loading full DEM '
                            f'({total_cells/1e6:.0f}M cells). '
                            'Place the outlet first to use windowed loading.', 'warn')
                    work_path = _cc['dem_path']

                _cc['cell_size_m'] = cell_m
                _update_param_info()

                for pct, msg in [(20,'Loading into pysheds…'),(35,'Filling pits…'),
                                  (50,'Filling depressions…'),(63,'Resolving flats…'),
                                  (74,'Flow direction…'),(86,'Flow accumulation…'),
                                  (93,'Saving to temp files…')]:
                    prep_prog.value = pct
                    prep_lbl.value  = f'<small style="color:{_MUTED}">{msg}</small>'
                    if pct == 20:
                        grid = Grid.from_raster(work_path)
                        dem  = grid.read_raster(work_path)
                    elif pct == 35: pit_filled = grid.fill_pits(dem)
                    elif pct == 50: dep_filled = grid.fill_depressions(pit_filled)
                    elif pct == 63: inflated   = grid.resolve_flats(dep_filled)
                    elif pct == 74: fdir       = grid.flowdir(inflated)
                    elif pct == 86: facc       = grid.accumulation(fdir)
                    elif pct == 93:
                        tmp_dir   = tempfile.mkdtemp(prefix='hbv_dem_prep_')
                        fdir_path = os.path.join(tmp_dir, 'fdir.tif')
                        facc_path = os.path.join(tmp_dir, 'facc.tif')
                        grid.to_raster(fdir, fdir_path)
                        grid.to_raster(facc, facc_path)
                        _cc['_fdir_path'] = fdir_path
                        _cc['_facc_path'] = facc_path

                prep_prog.value = 100
                size_note = (f'{total_cells/1e6:.1f}M cells from {buf_km:.0f} km buffer'
                             if lat_now else f'{total_cells/1e6:.1f}M cells (full file)')
                prep_lbl.value = (
                    f'<small style="color:#4ade80">✔ DEM prepared — '
                    f'cell ≈ {cell_m:.0f} m | {size_note}. '
                    f'Set parameters and click <b>Delineate</b>.</small>'
                )
                cc_run_btn.disabled = False
                log(f'DEM prepared: cell ≈ {cell_m:.0f} m', 'ok')
            except Exception as exc:
                prep_lbl.value = f'<small style="color:#ef4444">Error: {exc}</small>'
                log(f'Prepare error: {exc}', 'error')
                log(traceback.format_exc(), 'error')
                cc_run_btn.disabled = True
            finally:
                prep_btn.disabled = False
                prep_prog.layout.visibility = 'hidden'

        threading.Thread(target=_run, daemon=True).start()

    prep_btn.on_click(_cc_prepare)

    # ── Delineate ─────────────────────────────────────────────────────────────
    cc_run_btn = widgets.Button(description='▶ Delineate', button_style='success',
                                 disabled=True,
                                 layout=widgets.Layout(width='130px', height='36px'))
    cc_prog    = widgets.IntProgress(value=0, min=0, max=100,
                                      layout=widgets.Layout(width='180px', visibility='hidden'))
    cc_run_lbl = widgets.HTML(value='')

    def _cc_delineate(b):
        if not _cc.get('_fdir_path'):
            cc_run_lbl.value = '<small style="color:#ef4444">Click "Prepare DEM" first</small>'
            return
        if _cc.get('outlet_lat') is None:
            cc_run_lbl.value = '<small style="color:#ef4444">Place the outlet on the map first</small>'
            return
        cc_run_btn.disabled = True
        cc_prog.layout.visibility = 'visible'
        cc_prog.value = 10
        cc_run_lbl.value = f'<small style="color:{_MUTED}">Snapping + delineating…</small>'

        lat       = _cc['outlet_lat']
        lon       = _cc['outlet_lon']
        min_area_km2 = snap_area_input.value
        max_snap_m   = snap_dist_input.value
        cell_m       = _cc.get('cell_size_m', 90)
        fdir_path    = _cc['_fdir_path']
        facc_path    = _cc['_facc_path']

        def _run():
            try:
                Grid  = _ensure_pysheds()
                grid  = Grid.from_raster(fdir_path)
                fdir  = grid.read_raster(fdir_path)
                facc  = grid.read_raster(facc_path)
                cc_prog.value = 25

                min_cells   = max(1, int(min_area_km2 * 1e6 / (cell_m ** 2)))
                stream_mask = facc >= min_cells
                if not stream_mask.any():
                    raise ValueError(
                        f'No stream with ≥{min_area_km2} km² upstream. '
                        'Lower "Min upstream area".')

                cc_prog.value = 40
                x_snap, y_snap = grid.snap_to_mask(stream_mask, (lon, lat), xytype='coordinate')
                snap_dist_m = math.hypot(
                    (x_snap - lon) * 111320 * math.cos(math.radians(lat)),
                    (y_snap - lat) * 110540,
                )
                log(f'Snapped → ({y_snap:.5f}°N, {x_snap:.5f}°E) — {snap_dist_m:.0f} m from click', 'info')

                if snap_dist_m > max_snap_m:
                    raise ValueError(
                        f'Nearest stream is {snap_dist_m:.0f} m away '
                        f'(limit = {max_snap_m} m). '
                        'Lower "Min upstream area" or raise "Max snap distance".')

                cc_prog.value = 60
                catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')
                grid.clip_to(catch)
                geoms = [shape(s) for s, v in grid.polygonize() if v]
                if not geoms:
                    raise ValueError('No polygon produced — try a different outlet.')
                polygon = unary_union(geoms)
                cc_prog.value = 85

                _cc['catchment_geom'] = polygon
                _cc['user_polygon']   = None
                _cc['outlet_lat']     = y_snap
                _cc['outlet_lon']     = x_snap
                if _cc['outlet_marker']:
                    _cc['outlet_marker'].location = [y_snap, x_snap]

                area_km2 = (gpd.GeoSeries([polygon], crs='EPSG:4326')
                            .to_crs('EPSG:3067').area.iloc[0] / 1e6)
                bnd = polygon.bounds

                if _cc['catch_layer'] is not None:
                    try: m.remove(_cc['catch_layer'])
                    except Exception: pass

                layer = GeoJSON(
                    data=_polygon_to_geojson(polygon),
                    style={'color':'#fb923c','fillColor':'#7c2d12',
                           'fillOpacity':0.30,'weight':3},
                    hover_style={'fillOpacity': 0.50},
                )
                m.add(layer)
                _cc['catch_layer'] = layer

                clat = (bnd[1]+bnd[3])/2; clon = (bnd[0]+bnd[2])/2
                span = max(bnd[3]-bnd[1], bnd[2]-bnd[0])
                m.center = [clat, clon]
                m.zoom   = max(8, min(14, int(round(8 - math.log2(max(span, 0.001))))))

                cc_prog.value = 100
                cc_run_lbl.value = (
                    f'<small style="color:#4ade80">✔ {area_km2:.2f} km² | '
                    f'snap {snap_dist_m:.0f} m — adjust params and Delineate again, '
                    f'or use polygon tool to edit.</small>'
                )
                cc_status.value = _status_ok(
                    f'<b>Catchment {area_km2:.2f} km²</b> | snap {snap_dist_m:.0f} m. '
                    f'Tune parameters and Delineate again, or <b>Use this catchment</b>.'
                )
                log(f'Catchment: {area_km2:.2f} km² (snap {snap_dist_m:.0f} m)', 'ok')
            except Exception as exc:
                cc_run_lbl.value = f'<small style="color:#ef4444">Error: {exc}</small>'
                log(f'Delineation error: {exc}', 'error')
            finally:
                cc_run_btn.disabled = False
                cc_prog.layout.visibility = 'hidden'

        threading.Thread(target=_run, daemon=True).start()

    cc_run_btn.on_click(_cc_delineate)

    # ── D: Use / Export ───────────────────────────────────────────────────────
    cc_use_btn    = widgets.Button(description='✔ Use this catchment',
                                    button_style='primary',
                                    layout=widgets.Layout(width='210px', height='36px'))
    cc_export_btn = widgets.Button(description='⬇ Export shapefile (.zip)',
                                    layout=widgets.Layout(width='210px', height='36px'))
    cc_use_lbl    = widgets.HTML(value='')
    cc_export_lbl = widgets.HTML(value='')

    def _active_polygon():
        return _cc.get('user_polygon') or _cc.get('catchment_geom')

    def _cc_use(b):
        polygon = _active_polygon()
        if polygon is None:
            cc_use_lbl.value = '<small style="color:#ef4444">Delineate first</small>'
            return
        try:
            tmp = tempfile.mkdtemp(prefix='hbv_custom_shp_')
            shp_path = os.path.join(tmp, 'custom_catchment.shp')
            gdf = gpd.GeoDataFrame(
                {'custom_id': ['custom_1'], 'geometry': [polygon]}, crs='EPSG:4326')
            gdf.to_file(shp_path)
            state['shapefile_path'] = shp_path
            state['gdf'] = gdf;        state['gdf_wgs'] = gdf.copy()
            state['id_col']       = 'custom_id'
            state['selected_ids'] = {'custom_1'}
            state['selected_id']  = 'custom_1'
            state['all_ids']      = ['custom_1']
            b_geom = polygon.bounds
            state['catchment_bounds'] = (b_geom[0], b_geom[1], b_geom[2], b_geom[3])
            state['id_to_idx'] = {'custom_1': 0}
            state['sindex']    = STRtree(gdf.geometry.values)
            area_km2 = (gpd.GeoSeries([polygon], crs='EPSG:4326')
                        .to_crs('EPSG:3067').area.iloc[0] / 1e6)
            source = 'adjusted' if _cc.get('user_polygon') else 'delineated'
            cc_use_lbl.value = (
                f'<small style="color:#4ade80">✔ {source} catchment loaded '
                f'({area_km2:.2f} km²) — go to Input tab → Step 3</small>'
            )
            log(f'Custom catchment loaded — {area_km2:.2f} km² ({source})', 'ok')
            go = refs.get('go_to_input')
            if go: go()
        except Exception as exc:
            cc_use_lbl.value = f'<small style="color:#ef4444">Error: {exc}</small>'
            log(f'Use catchment error: {exc}', 'error')

    cc_use_btn.on_click(_cc_use)

    def _cc_export(b):
        polygon = _active_polygon()
        if polygon is None:
            cc_export_lbl.value = '<small style="color:#ef4444">Delineate first</small>'
            return
        try:
            tmp = tempfile.mkdtemp(prefix='hbv_export_')
            shp_dir  = os.path.join(tmp, 'custom_catchment')
            os.makedirs(shp_dir)
            shp_path = os.path.join(shp_dir, 'custom_catchment.shp')
            area_km2 = round(gpd.GeoSeries([polygon], crs='EPSG:4326')
                             .to_crs('EPSG:3067').area.iloc[0] / 1e6, 4)
            gpd.GeoDataFrame(
                {'id': ['custom_1'], 'area_km2': [area_km2], 'geometry': [polygon]},
                crs='EPSG:4326').to_file(shp_path)
            zip_path = os.path.join(tmp, 'custom_catchment.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for fn in os.listdir(shp_dir):
                    zf.write(os.path.join(shp_dir, fn), arcname=fn)
            cc_export_lbl.value = f'<small style="color:#4ade80">✔ Saved: {zip_path}</small>'
            log(f'Catchment exported: {zip_path}', 'ok')
            dl_dir = (state.get('download_dir') or '').strip()
            if dl_dir:
                try:
                    dst = os.path.join(dl_dir, 'custom_catchment.zip')
                    shutil.copy(zip_path, dst)
                    cc_export_lbl.value += f'<br><small style="color:#4ade80">Copied → {dst}</small>'
                except Exception: pass
        except Exception as exc:
            cc_export_lbl.value = f'<small style="color:#ef4444">Export error: {exc}</small>'
            log(f'Export error: {exc}', 'error')

    cc_export_btn.on_click(_cc_export)

    # ── layout ────────────────────────────────────────────────────────────────
    def _step_lbl(num, title):
        return _h(f'<div style="font-size:13px;font-weight:700;color:{_TEXT};'
                  f'padding-bottom:6px;border-bottom:1px solid {_BDR};margin-bottom:8px">'
                  f'{num} {title}</div>')

    def _hint(txt):
        return _h(f'<div style="font-size:11px;color:{_MUTED};margin-bottom:6px;line-height:1.6">{txt}</div>')

    step_a = widgets.VBox([
        _step_lbl('①', 'Load DEM'),
        _hint('Local GeoTIFF or COG URL. Enables elevation readout on click.'),
        cog_hint,
        widgets.HBox([cc_src_input, cc_load_btn],
                      layout=widgets.Layout(gap='8px', align_items='center')),
        cc_load_lbl,
    ], layout=_card_layout())

    step_b = widgets.VBox([
        _step_lbl('②', 'Place outlet &amp; inspect elevation'),
        _hint('Click anywhere on the map to read elevation. '
              'Press <b style="color:#e2e8f0">Place outlet on map</b>, click your outlet, drag pin to fine-tune. '
              'Use the polygon draw tool on the map to draw a custom boundary.'),
        widgets.HBox([arm_btn, arm_lbl],
                      layout=widgets.Layout(align_items='center', gap='8px')),
        outlet_lbl,
        elev_lbl,
    ], layout=_card_layout())

    step_c = widgets.VBox([
        _step_lbl('③', 'Delineate'),
        _hint(
            '<b style="color:#e2e8f0">Step 3a</b>: prepare the DEM once (fill pits, flow direction, accumulation — slow). '
            '<b style="color:#e2e8f0">Step 3b</b>: tune parameters and Delineate as many times as needed (fast).<br>'
            '<b style="color:#e2e8f0">Min upstream area</b>: snaps to nearest stream draining ≥ this. '
            'Too large a result → lower it.<br>'
            '<b style="color:#e2e8f0">Max snap distance</b>: reject if snap moves outlet further than this.'
        ),
        widgets.HBox([buffer_input]),
        buffer_info,
        widgets.HBox([prep_btn, prep_prog],
                      layout=widgets.Layout(align_items='center', gap='8px', margin='4px 0')),
        prep_lbl,
        widgets.HBox([snap_area_input, snap_dist_input],
                      layout=widgets.Layout(gap='12px', flex_wrap='wrap')),
        param_info,
        widgets.HBox([cc_run_btn, cc_prog],
                      layout=widgets.Layout(align_items='center', gap='8px', margin='4px 0')),
        cc_run_lbl,
    ], layout=_card_layout())

    step_d = widgets.VBox([
        _step_lbl('④', 'Use / Export'),
        _hint('Drawn polygon (from the map tool) overrides the auto-delineated one.'),
        widgets.HBox([cc_use_btn, cc_export_btn],
                      layout=widgets.Layout(gap='8px', flex_wrap='wrap')),
        cc_use_lbl, cc_export_lbl,
    ], layout=_card_layout())

    # ── header strip (spans full width above both panels) ────────────────────
    header = widgets.VBox([
        _h(f'<div style="font-size:15px;font-weight:700;color:{_TEXT};margin-bottom:2px">'
           f'Custom Catchment Delineation</div>'
           f'<div style="font-size:11px;color:{_MUTED}">'
           f'Load a DEM · click map · place outlet · delineate · adjust · load into HBV</div>'),
        cc_status,
    ], layout=widgets.Layout(
        padding='10px 14px 8px 14px',
        background_color=_BG0,
        border_bottom=f'1px solid {_BDR}',
    ))

    # ── left panel: scrollable steps ─────────────────────────────────────────
    left_panel = widgets.VBox(
        [step_a, step_b, step_c, step_d],
        layout=widgets.Layout(
            width='400px', min_width='340px',
            height=_PANEL_H,
            overflow_y='auto',
            padding='10px',
            background_color=_BG1,
            border_right=f'1px solid {_BDR}',
        ),
    )

    # ── right panel: full-height map ─────────────────────────────────────────
    map_panel = widgets.VBox(
        [m],
        layout=widgets.Layout(
            flex='1',
            height=_MAP_H,
            min_height='300px',
            background_color=_BG0,
        ),
    )

    body = widgets.HBox(
        [left_panel, map_panel],
        layout=widgets.Layout(
            width='100%',
            height=_PANEL_H,
            background_color=_BG0,
        ),
    )

    return widgets.VBox(
        [header, body],
        layout=widgets.Layout(
            width='100%',
            background_color=_BG0,
            overflow='hidden',
        ),
    )
