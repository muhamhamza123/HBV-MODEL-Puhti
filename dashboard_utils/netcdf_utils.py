import sys, subprocess
import numpy as np

from .logging_utils import log


def _ensure_xarray():
    try:
        import xarray as xr
        return xr
    except ImportError:
        log('xarray not found — installing...', 'warn')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xarray', 'netCDF4', 'scipy'], check=False)
        import importlib
        try:
            xr = importlib.import_module('xarray')
            log('xarray installed OK', 'ok')
            return xr
        except ImportError:
            return None


def _reproject_bbox(minlon, minlat, maxlon, maxlat, src_path):
    from pyproj import Transformer
    import netCDF4 as nc
    with nc.Dataset(src_path, 'r') as ds:
        xname = next((v for v in ds.variables if v.lower() in ('x', 'lon', 'longitude', 'easting')), None)
        yname = next((v for v in ds.variables if v.lower() in ('y', 'lat', 'latitude', 'northing')), None)
        if xname is None or yname is None:
            return minlon, minlat, maxlon, maxlat
        xvals = ds.variables[xname][:]
        yvals = ds.variables[yname][:]
    x_range = float(xvals.max()) - float(xvals.min())
    y_range = float(yvals.max()) - float(yvals.min())
    if x_range < 360 and y_range < 180:
        return minlon, minlat, maxlon, maxlat
    x_mid = float(xvals.mean()); y_mid = float(yvals.mean())
    if 100_000 < x_mid < 900_000 and 6_000_000 < y_mid < 8_000_000:
        epsg = 3067
    else:
        return minlon, minlat, maxlon, maxlat
    transformer = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    corners = [transformer.transform(minlon, minlat), transformer.transform(maxlon, minlat),
               transformer.transform(minlon, maxlat), transformer.transform(maxlon, maxlat)]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def clip_nc_to_bbox(src_path, bbox_str, out_path):
    minlon, minlat, maxlon, maxlat = [float(x) for x in bbox_str.split(',')]
    try:
        minx, miny, maxx, maxy = _reproject_bbox(minlon, minlat, maxlon, maxlat, src_path)
        log(f'Clip bbox in file CRS: {minx:.1f},{miny:.1f},{maxx:.1f},{maxy:.1f}', 'info')
    except Exception as e:
        log(f'Bbox reprojection failed ({e}) — using raw lon/lat', 'warn')
        minx, miny, maxx, maxy = minlon, minlat, maxlon, maxlat
    xr = _ensure_xarray()
    if xr is not None:
        ds = xr.open_dataset(src_path)
        xname = next((c for c in ds.coords if c.lower() in ('x', 'lon', 'longitude', 'easting')), None)
        yname = next((c for c in ds.coords if c.lower() in ('y', 'lat', 'latitude', 'northing')), None)
        if xname and yname:
            yvals = ds[yname].values
            if yvals[0] > yvals[-1]:
                ds = ds.sel({xname: slice(minx, maxx), yname: slice(maxy, miny)})
            else:
                ds = ds.sel({xname: slice(minx, maxx), yname: slice(miny, maxy)})
        ds.to_netcdf(out_path); ds.close(); return
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError('Neither xarray nor netCDF4 available')
    with nc.Dataset(src_path, 'r') as src:
        xname = next((d for d in src.dimensions if d.lower() in ('x', 'lon', 'longitude', 'easting')), None)
        yname = next((d for d in src.dimensions if d.lower() in ('y', 'lat', 'latitude', 'northing')), None)
        if xname is None or yname is None: raise ValueError('Cannot find x/y dims')
        xcoord = src.variables[xname][:]; ycoord = src.variables[yname][:]
        x_idx = np.where((xcoord >= minx) & (xcoord <= maxx))[0]
        y_idx = np.where((ycoord >= miny) & (ycoord <= maxy))[0]
        if x_idx.size == 0 or y_idx.size == 0: raise ValueError('Bbox does not intersect grid')
        i0_x, i1_x = x_idx[0], x_idx[-1]+1; i0_y, i1_y = y_idx[0], y_idx[-1]+1
        with nc.Dataset(out_path, 'w') as dst:
            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
            for dname, dim in src.dimensions.items():
                if dname == xname: dst.createDimension(dname, i1_x-i0_x)
                elif dname == yname: dst.createDimension(dname, i1_y-i0_y)
                else: dst.createDimension(dname, None if dim.isunlimited() else len(dim))
            for vname, var in src.variables.items():
                dims = var.dimensions
                out_v = dst.createVariable(vname, var.datatype, dims,
                    fill_value=var._FillValue if hasattr(var, '_FillValue') else False)
                out_v.setncatts({a: var.getncattr(a) for a in var.ncattrs() if a != '_FillValue'})
                idx = [slice(None)] * len(dims)
                if xname in dims: idx[dims.index(xname)] = slice(i0_x, i1_x)
                if yname in dims: idx[dims.index(yname)] = slice(i0_y, i1_y)
                out_v[:] = var[tuple(idx)]
