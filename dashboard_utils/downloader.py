import os, sys, time, zipfile, threading, subprocess
import requests

from .logging_utils import log
from .netcdf_utils import clip_nc_to_bbox, _ensure_xarray

# ── Module-level state injected by notebook via init() ──────────────────────
_state = {}
_get_dl_dir = lambda: '/tmp'


def init(state, get_dl_dir):
    global _state, _get_dl_dir
    _state = state
    _get_dl_dir = get_dl_dir


# ── FMI S3 constants ────────────────────────────────────────────────────────
FMI_S3_BASE = 'https://fmi-gridded-obs-daily-1km.s3.amazonaws.com/Netcdf'

FMI_VARIABLES = {
    'precipitation': ('RRday',   'rrday',   'Daily precipitation sum (mm)', 1961, 2100,
                      lambda y: f'rrday_{y}.nc'),
    'temperature':   ('Tday',    'tday',    'Daily mean temperature (C)',    1961, 2100,
                      lambda y: f'tday_{y}.nc'),
    'et':            ('ET0_FAO', 'ET0_FAO', 'Daily potential ET — FAO (mm)', 1981, 2020,
                      lambda y: f'ET0_FAO_{y}_months_4_to_9.nc'),
}

ERA5_VARIABLES = {
    'precipitation': ('total_precipitation',     'reanalysis', 'Daily precip sum'),
    'temperature':   ('2m_temperature',           'reanalysis', 'Daily mean temp'),
    'et':            ('surface_latent_heat_flux', 'reanalysis', 'Surface latent heat flux'),
}

_CDS_API_KEY = '739687c0-0417-4450-a100-e75462c78a79'
_CDS_API_URL = 'https://cds.climate.copernicus.eu/api'

TASO_URLS = {k: 'https://wwwd3.ymparisto.fi/d3/gis_data/spesific/valumaalueet.zip'
             for k in ('taso1', 'taso2', 'taso3', 'taso4', 'taso5')}
SYKE_URBAN_URL = 'https://wwwd3.ymparisto.fi/d3/gis_data/spesific/taajama.zip'
SYKE_AGRI_URL  = 'https://wwwd3.ymparisto.fi/d3/gis_data/spesific/maatalousmaa.zip'


# ── FMI helpers ─────────────────────────────────────────────────────────────
def fmi_s3_urls(var_key, start_date, end_date):
    folder, _, _, yr_min, yr_max, fname_fn = FMI_VARIABLES[var_key]
    years = [y for y in range(start_date.year, end_date.year + 1) if yr_min <= y <= yr_max]
    if not years:
        raise ValueError(f'FMI {var_key} data only available {yr_min}–{yr_max}.')
    skipped = [y for y in range(start_date.year, end_date.year + 1) if not (yr_min <= y <= yr_max)]
    if skipped:
        log(f'FMI {var_key}: skipping years outside {yr_min}–{yr_max}: {skipped}', 'warn')
    return [f'{FMI_S3_BASE}/{folder}/{fname_fn(y)}' for y in years]


def download_and_merge_nc(var_key, state_key, start_date, end_date, prog_w, status_w, btn_w):
    btn_w.disabled = True; prog_w.layout.visibility = 'visible'; prog_w.value = 0
    urls = fmi_s3_urls(var_key, start_date, end_date); n = len(urls)
    bounds = _state.get('catchment_bounds')
    if bounds:
        bbox_str = f'{bounds[0]:.4f},{bounds[1]:.4f},{bounds[2]:.4f},{bounds[3]:.4f}'
        log(f'FMI S3: {n} file(s) for {var_key} — clipping to bbox {bbox_str}', 'dl')
        status_w.value = f'Downloading {n} file(s) + clipping...'
    else:
        bbox_str = None
        log(f'FMI S3: {n} file(s) for {var_key} — full Finland grid', 'dl')
        status_w.value = f'Downloading {n} file(s)...'

    def _run():
        try:
            paths = []
            for i, url in enumerate(urls):
                year = start_date.year + i; fname = f'{var_key}_{year}.nc'
                dest = os.path.join(_get_dl_dir(), fname)
                log(f'FMI S3: fetching {fname} ({i+1}/{n})...', 'dl')
                status_w.value = f'{fname} ({i+1}/{n})...'
                r = requests.get(url, stream=True, timeout=600); r.raise_for_status()
                total = int(r.headers.get('content-length', 0)); done = 0; _last_prog = 0.0
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=4*1024*1024):
                        if chunk:
                            f.write(chunk); done += len(chunk)
                            now = time.monotonic()
                            if now - _last_prog >= 0.5:
                                prog_w.value = min(int(((i+done/max(total, 1))/n)*100), 99)
                                _last_prog = now
                if bbox_str:
                    try:
                        clipped = dest.replace('.nc', '_clipped.nc')
                        clip_nc_to_bbox(dest, bbox_str, clipped)
                        os.remove(dest); dest = clipped
                        log(f'Clipped {fname} -> {os.path.getsize(dest)/1024/1024:.1f} MB', 'ok')
                    except Exception as clip_err:
                        log(f'Clip failed ({clip_err}) — keeping full file', 'warn')
                paths.append(dest)
            if len(paths) == 1:
                _state[state_key] = paths[0]; mb = os.path.getsize(paths[0])/1024/1024
                status_w.value = f'OK {os.path.basename(paths[0])} ({mb:.0f} MB)'
                log(f'Ready: {os.path.basename(paths[0])} ({mb:.0f} MB)', 'ok')
            else:
                try:
                    import xarray as xr
                    ds = xr.open_mfdataset(paths, combine='by_coords')
                    merged = os.path.join(_get_dl_dir(), f'{var_key}_{start_date.year}_{end_date.year}.nc')
                    ds.to_netcdf(merged); ds.close()
                    _state[state_key] = merged; mb = os.path.getsize(merged)/1024/1024
                    status_w.value = f'OK Merged {len(paths)} years -> {mb:.0f} MB'
                except ImportError:
                    _state[state_key] = paths[0]; status_w.value = f'OK {len(paths)} files'
            prog_w.value = 100
        except Exception as e:
            status_w.value = f'Error: {e}'; log(f'FMI error ({var_key}): {e}', 'error')
        finally:
            btn_w.disabled = False

    threading.Thread(target=_run).start()


# ── ERA5 helpers ─────────────────────────────────────────────────────────────
def _ensure_cdsapi():
    try:
        import cdsapi; return cdsapi
    except ImportError:
        log('cdsapi not found — installing...', 'warn')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cdsapi'], check=False)
        import importlib
        try:
            mod = importlib.import_module('cdsapi'); log('cdsapi installed OK', 'ok'); return mod
        except ImportError:
            return None


def ensure_cdsapirc():
    rc = os.path.expanduser('~/.cdsapirc')
    desired = f'url: {_CDS_API_URL}\nkey: {_CDS_API_KEY}\n'
    try:
        if not os.path.exists(rc) or open(rc).read().strip() != desired.strip():
            with open(rc, 'w') as f: f.write(desired)
            log('Wrote ~/.cdsapirc', 'ok')
    except Exception as e:
        log(f'Could not write ~/.cdsapirc: {e}', 'warn')


def download_era5(var_key, state_key, start_date, end_date, prog_w, status_w, btn_w):
    btn_w.disabled = True; prog_w.layout.visibility = 'visible'; prog_w.value = 0
    bounds = _state.get('catchment_bounds')
    if not bounds:
        status_w.value = 'Select a catchment first'; log('ERA5: no catchment', 'warn')
        btn_w.disabled = False; return
    ensure_cdsapirc()
    cds_var, product_type, _ = ERA5_VARIABLES[var_key]
    pad = 0.1
    area = [round(bounds[3]+pad, 2), round(bounds[0]-pad, 2), round(bounds[1]-pad, 2), round(bounds[2]+pad, 2)]
    years  = [str(y) for y in range(start_date.year, end_date.year+1)]
    months = [f'{m:02d}' for m in range(1, 13)]
    days   = [f'{d:02d}' for d in range(1, 32)]
    hours  = [f'{h:02d}:00' for h in range(24)]
    raw_path   = os.path.join(_get_dl_dir(), f'era5_{var_key}_{start_date.year}_{end_date.year}_hourly.nc')
    daily_path = os.path.join(_get_dl_dir(), f'era5_{var_key}_{start_date.year}_{end_date.year}_daily.nc')
    log(f'ERA5: {cds_var} | {start_date.year}–{end_date.year}', 'dl')
    status_w.value = 'Submitting ERA5 request...'

    def _run():
        try:
            cdsapi = _ensure_cdsapi()
            if cdsapi is None: status_w.value = 'cdsapi install failed'; return
            prog_w.value = 5; c = cdsapi.Client(quiet=True, progress=False); prog_w.value = 10
            status_w.value = 'ERA5 queued — waiting...'
            c.retrieve('reanalysis-era5-single-levels', {
                'product_type': [product_type], 'variable': [cds_var],
                'year': years, 'month': months, 'day': days, 'time': hours,
                'area': area, 'data_format': 'netcdf', 'download_format': 'unarchived',
            }, raw_path)
            prog_w.value = 80
            xr = _ensure_xarray()
            if xr is None:
                _state[state_key] = raw_path; status_w.value = 'OK (hourly, no xarray)'
            else:
                ds = xr.open_dataset(raw_path)
                vname = list(ds.data_vars)[0]
                time_dim = next((t for t in ('time', 'valid_time') if t in ds.dims or t in ds.coords), None)
                if time_dim and time_dim != 'time': ds = ds.rename({time_dim: 'time'})
                if var_key == 'precipitation':
                    daily = ds[vname].resample(time='1D').sum() * 1000.0
                elif var_key == 'temperature':
                    daily = ds[vname].resample(time='1D').mean()
                elif var_key == 'et':
                    daily = ds[vname].resample(time='1D').sum().abs() / 2.45e6
                else:
                    daily = ds[vname].resample(time='1D').mean()
                daily = daily.sel(time=slice(str(start_date), str(end_date)))
                daily.to_dataset(name=vname).to_netcdf(daily_path)
                ds.close(); os.remove(raw_path)
                _state[state_key] = daily_path; mb = os.path.getsize(daily_path)/1024/1024
                status_w.value = f'OK {mb:.1f} MB daily'; log(f'ERA5 daily ready ({mb:.1f} MB)', 'ok')
            prog_w.value = 100
        except Exception as e:
            import traceback
            status_w.value = f'Error: {e}'; log(f'ERA5 error: {e}', 'error')
            log(traceback.format_exc(), 'error')
        finally:
            btn_w.disabled = False

    threading.Thread(target=_run).start()


# ── Generic NC / Shapefile downloads ────────────────────────────────────────
def download_nc_url(url, var_key, state_key, prog_w, status_w, btn_w):
    btn_w.disabled = True; prog_w.layout.visibility = 'visible'; prog_w.value = 0
    status_w.value = 'Connecting...'

    def _run():
        try:
            fname = url.split('?')[0].split('/')[-1] or f'{var_key}.nc'
            dest = os.path.join(_state['temp_dir'], fname)
            r = requests.get(url, stream=True, timeout=600); r.raise_for_status()
            total = int(r.headers.get('content-length', 0)); done = 0; _last_prog = 0.0
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=4*1024*1024):
                    if chunk:
                        f.write(chunk); done += len(chunk)
                        now = time.monotonic()
                        if now - _last_prog >= 0.5:
                            prog_w.value = int(done/total*100) if total else 50; _last_prog = now
            _state[state_key] = dest; prog_w.value = 100
            status_w.value = f'OK {fname} ({done/1024/1024:.1f} MB)'
        except Exception as e:
            status_w.value = f'Error: {e}'
        finally:
            btn_w.disabled = False

    threading.Thread(target=_run).start()


def download_shp(url, state_key, prog_w, lbl_w, btn_w=None):
    if btn_w: btn_w.disabled = True
    prog_w.layout.visibility = 'visible'; lbl_w.value = 'Downloading...'

    def _run():
        try:
            fname = url.split('?')[0].split('/')[-1] or f'{state_key}.zip'
            dest = os.path.join(_state['temp_dir'], fname)
            r = requests.get(url, stream=True, timeout=300); r.raise_for_status()
            total = int(r.headers.get('content-length', 0)); done = 0; _last_prog = 0.0
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk); done += len(chunk)
                        now = time.monotonic()
                        if now - _last_prog >= 0.5:
                            prog_w.value = int(done/total*100) if total else 50; _last_prog = now
            prog_w.value = 100
            if fname.endswith('.zip'):
                exdir = os.path.join(_state['temp_dir'], state_key)
                os.makedirs(exdir, exist_ok=True)
                with zipfile.ZipFile(dest) as z: z.extractall(exdir)
                shps = [os.path.join(dp, fn) for dp, _, fns in os.walk(exdir)
                        for fn in fns if fn.endswith('.shp')]
                _state[state_key] = shps[0] if shps else dest
            else:
                _state[state_key] = dest
            lbl_w.value = f'OK {os.path.basename(_state[state_key])}'
        except Exception as e:
            lbl_w.value = f'Error: {e}'
        finally:
            if btn_w: btn_w.disabled = False

    threading.Thread(target=_run).start()
