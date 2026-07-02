"""
hbv_worker.py — MPI worker launched by the notebook via mpirun.

Launch pattern (from notebook):
    mpirun --np <N> --oversubscribe python hbv_worker.py --config /tmp/run_config.json

Rank 0 reads the config, scatters one catchment ID per rank.
Each rank independently runs hbv_prepare + hbv_S2S for its assigned catchment(s).
Results (output file paths) are gathered back to rank 0 and written to
a JSON result file that the notebook reads after mpirun exits.

If only 1 rank is available (MPI disabled/unavailable) the script falls back
to running all catchments sequentially in the same process — so the notebook
always works even without MPI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
import traceback

# ── MPI bootstrap ─────────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    _COMM  = MPI.COMM_WORLD
    _RANK  = _COMM.Get_rank()
    _SIZE  = _COMM.Get_size()
    _HAS_MPI = True
except ImportError:
    _COMM    = None
    _RANK    = 0
    _SIZE    = 1
    _HAS_MPI = False


def _log(msg: str) -> None:
    """Prefix every log line with the MPI rank so the notebook can show it."""
    print(f"[rank {_RANK:03d}] {msg}", flush=True)


# ── Argument parsing (done on every rank) ─────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # ── New HPC/Slurm mode (called by slurm/hbv_run.sh via the FastAPI) ──
    p.add_argument('--job-id',      help='HBV job UUID from the API')
    p.add_argument('--task',        type=int, default=0,
                   help='Slurm array task index (0-based)')
    p.add_argument('--total',       type=int, default=1,
                   help='Total Slurm array tasks')
    p.add_argument('--shapefile',   help='Path to catchment shapefile')
    p.add_argument('--precip-nc',   help='Precipitation NetCDF path')
    p.add_argument('--et-nc',       help='Evapotranspiration NetCDF path')
    p.add_argument('--temp-nc',     help='Temperature NetCDF path')
    p.add_argument('--urban-path',  help='Urban land-use raster/shapefile path')
    p.add_argument('--agri-path',   help='Agricultural land-use raster/shapefile path')
    p.add_argument('--para-csv',    help='Path to hbv_para.csv')
    p.add_argument('--catchments',  help='Comma-separated catchment IDs')
    p.add_argument('--id-col',      default='TASO_ID',
                   help='Shapefile column name for catchment IDs')
    p.add_argument('--output',      help='Output directory on shared NFS')
    # ── Legacy notebook mode (JSON config file) ──────────────────────────
    p.add_argument('--config',  help='Path to JSON config (legacy notebook mode)')
    p.add_argument('--results', help='Path for JSON results (legacy notebook mode)')
    return p.parse_args()


# ── Work unit: one catchment ──────────────────────────────────────────────
def _run_catchment(cfg: dict, sid_str: str) -> dict:
    """
    Run hbv_prepare + hbv_S2S for a single catchment.
    Returns a dict: {sid_str: {key: abs_output_path, ...}}
    Raises on error so the caller can catch and store the traceback.
    """
    import geopandas as gpd
    from hbv_prepare import prepare_meteorological_and_landuse_data_direct
    from hbv_S2S import run_hbv_model

    temp_dir       = cfg["temp_dir"]
    shapefile_path = cfg["shapefile_path"]
    id_col         = cfg["id_col"]
    out_root       = cfg["out_root"]          # project output dir

    # Resolve native dtype (int / float / str) for the catchment ID column
    gdf = gpd.read_file(shapefile_path, rows=1)
    col_dtype = gdf[id_col].dtype
    try:
        if col_dtype.kind in ("i", "u"):
            raw_sid = int(sid_str)
        elif col_dtype.kind == "f":
            raw_sid = float(sid_str)
        else:
            raw_sid = str(sid_str)
    except (ValueError, TypeError):
        raw_sid = str(sid_str)

    # Per-catchment output CSV
    out_csv = os.path.join(temp_dir, f"met_input_{sid_str}.csv")

    # Each catchment gets its own copy of hbv_para.csv so parallel workers
    # don't race when reading and writing the shared file.
    catchment_dir = os.path.join(temp_dir, f"catchment_{sid_str}")
    os.makedirs(catchment_dir, exist_ok=True)
    catchment_para = os.path.join(catchment_dir, "hbv_para.csv")
    import shutil as _shutil
    _shutil.copy2(cfg["hbvpara_path"], catchment_para)

    _log(f"hbv_prepare → catchment {raw_sid}")
    prepare_meteorological_and_landuse_data_direct(
        shapefile_path         = shapefile_path,
        catchment_id_name      = id_col,
        taso_id_of_interest    = raw_sid,
        precipitation_nc       = cfg["precipitation_nc"],
        evapotranspiration_nc  = cfg["evapotranspiration_nc"],
        temperature_nc         = cfg["temperature_nc"],
        output_csv_path        = out_csv,
        urban_land_path        = cfg["urban_land_path"],
        agricultural_land_path = cfg["agricultural_land_path"],
        csv_parameters_path    = catchment_para,
    )

    # hbv_S2S expects hbv_para.csv in cwd — use the per-catchment copy
    # (which now has the correct land use row written by hbv_prepare)
    rank_dir = os.path.join(temp_dir, f"rank_{_RANK}_catchment_{sid_str}")
    os.makedirs(rank_dir, exist_ok=True)
    shutil.copy(catchment_para,
                os.path.join(rank_dir, "hbv_para.csv"))
    orig_cwd = os.getcwd()
    os.chdir(rank_dir)

    try:
        _log(f"hbv_S2S → catchment {raw_sid}")
        try:
            output_files = run_hbv_model(out_csv, catchment_id=sid_str)
        except TypeError:
            output_files = run_hbv_model(out_csv)
    finally:
        os.chdir(orig_cwd)

    # Copy outputs to a permanent per-catchment folder under out_root
    cdir = os.path.join(out_root, f"catchment_{sid_str}")
    os.makedirs(cdir, exist_ok=True)
    final_files = {}
    for k, src in output_files.items():
        dst = os.path.join(cdir, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)
        final_files[k] = dst
        _log(f"saved {os.path.basename(dst)}")

    # Also save met_input
    met_dst = os.path.join(cdir, f"met_input_{sid_str}.csv")
    if os.path.exists(out_csv):
        shutil.copy(out_csv, met_dst)

    return {sid_str: final_files}


# ── Distribute catchments across ranks ────────────────────────────────────
def _scatter_catchments(catchment_ids: list[str]) -> list[str]:
    """
    Simple round-robin assignment: rank r gets every id where index % size == r.
    Works for any number of catchments vs ranks (handles N < size gracefully).
    """
    return [cid for i, cid in enumerate(catchment_ids) if i % _SIZE == _RANK]


# ── Pool worker (must be at module level so spawn can pickle it) ──────────
def _run_one(args):
    sid, cfg = args
    try:
        return sid, _run_catchment(cfg, sid), None
    except Exception:
        return sid, None, traceback.format_exc()


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    if args.config:
        # Legacy path: notebook wrote a JSON config file
        with open(args.config) as f:
            cfg = json.load(f)
        catchment_ids: list[str] = cfg["catchment_ids"]
    else:
        # HPC path: all params come from CLI args set by the Slurm script
        if not args.catchments:
            raise SystemExit('Provide --catchments or --config')
        catchment_ids = [c.strip() for c in args.catchments.split(',') if c.strip()]

        # Build a cfg dict identical in shape to the JSON config so
        # _run_catchment() works unchanged in both modes.
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix=f'hbv_{args.job_id or "run"}_')
        cfg = {
            'temp_dir':              temp_dir,
            'shapefile_path':        args.shapefile,
            'id_col':                args.id_col,
            'out_root':              args.output or '.',
            'precipitation_nc':      args.precip_nc,
            'evapotranspiration_nc': args.et_nc,
            'temperature_nc':        args.temp_nc,
            'urban_land_path':       args.urban_path,
            'agricultural_land_path': args.agri_path,
            'hbvpara_path':          args.para_csv,
        }

        # Slice catchments by Slurm task index (same logic as hbv_S2S.py CLI)
        task  = args.task
        total = args.total
        catchment_ids = catchment_ids[task::total]

    if _RANK == 0:
        _log(f"MPI size={_SIZE}, catchments={len(catchment_ids)}, "
             f"mpi4py={'yes' if _HAS_MPI else 'no (sequential fallback)'}")

    my_ids = _scatter_catchments(catchment_ids)
    _log(f"assigned {len(my_ids)} catchment(s): {my_ids}")

    # ── Run assigned catchments in parallel using multiprocessing ─────────
    # HBV_CPUS_PER_TASK is set by the Slurm script from the API's cpus_per_task.
    # Each subprocess handles one catchment, so we saturate all allocated CPUs.
    import multiprocessing as _mp
    _n_workers = int(os.environ.get('HBV_CPUS_PER_TASK', '') or _mp.cpu_count() or 1)
    _n_workers = min(_n_workers, max(len(my_ids), 1))
    _phys_cpus = _mp.cpu_count()
    _log(f"node={os.uname().nodename} physical_cpus={_phys_cpus} "
         f"pool_size={_n_workers} catchments={len(my_ids)}")

    my_results: dict[str, dict] = {}
    my_errors:  dict[str, str]  = {}

    if not my_ids:
        pass  # nothing to do for this task
    elif _n_workers == 1 or len(my_ids) == 1:
        # single-process path — avoids fork/spawn overhead and container issues
        for sid in my_ids:
            try:
                result = _run_catchment(cfg, sid)
                my_results.update(result)
                _log(f"catchment {sid} DONE")
            except Exception:
                tb = traceback.format_exc()
                _log(f"catchment {sid} ERROR: {tb}")
                my_errors[sid] = tb
    else:
        # Use spawn (not fork) so it works inside Apptainer/Singularity containers
        ctx = _mp.get_context('spawn')

        with ctx.Pool(processes=_n_workers) as pool:
            for sid, result, err in pool.map(_run_one, [(sid, cfg) for sid in my_ids]):
                if err:
                    _log(f"catchment {sid} ERROR: {err}")
                    my_errors[sid] = err
                else:
                    my_results.update(result)
                    _log(f"catchment {sid} DONE")

    # ── Gather results to rank 0 ──────────────────────────────────────────
    if _HAS_MPI and _SIZE > 1:
        all_results_list = _COMM.gather(my_results, root=0)
        all_errors_list  = _COMM.gather(my_errors,  root=0)
    else:
        all_results_list = [my_results]
        all_errors_list  = [my_errors]

    if _RANK == 0:
        merged_results: dict[str, dict] = {}
        merged_errors:  dict[str, str]  = {}
        for d in all_results_list:
            merged_results.update(d)
        for d in all_errors_list:
            merged_errors.update(d)

        payload = {
            "results": merged_results,   # {sid: {key: path}}
            "errors":  merged_errors,    # {sid: traceback_str}
        }

        n_ok  = len(merged_results)
        n_err = len(merged_errors)

        if args.results:
            # Legacy notebook mode — write results JSON for the notebook to read
            with open(args.results, "w") as f:
                json.dump(payload, f, indent=2)
            _log(f"All done — {n_ok} succeeded, {n_err} failed. "
                 f"Results written to {args.results}")
        else:
            # HPC mode — results are already in output_dir; just log summary
            _log(f"All done — {n_ok} succeeded, {n_err} failed.")
            if n_err:
                for sid, tb in merged_errors.items():
                    _log(f"FAILED {sid}:\n{tb}")
            # Write a simple status file the API can poll as a lightweight done-check
            status_path = os.path.join(cfg.get('out_root', '.'),
                                       f'task_{args.task}_status.json')
            with open(status_path, 'w') as f:
                json.dump({'ok': n_ok, 'errors': n_err,
                           'failed_ids': list(merged_errors.keys())}, f)
        if merged_errors:
            for sid, tb in merged_errors.items():
                _log(f"FAILED catchment {sid}:\n{tb}")


if __name__ == "__main__":
    main()
