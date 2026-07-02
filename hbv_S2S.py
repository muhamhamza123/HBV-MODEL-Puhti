# -*- coding: utf-8 -*-
"""
hbv_S2S.py — HBV hydrological model (vectorised NumPy rewrite).

Original author : Harri Koivusalo
Vectorisation   : MPI parallelisation branch

Literature:
  Seibert & Vis (2012) Hydrol. Earth Syst. Sci. 16, 3315-3325
  Koskela et al. (2012) Water Resources Research 48, W11513

The outer time-step loop (for i in range(ndays)) has been replaced with
NumPy array operations.  Each HBV sub-model (agric / forest / urban) is
still run sequentially in *state* (snow / soil / zone stores carry forward),
but all arithmetic within each step is vectorised across the three land-use
columns simultaneously via np.where / np.clip / np.minimum.

The public function signature is unchanged:
    run_hbv_model(met_input_csv, catchment_id=None) -> dict[str, str]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame


# ═══════════════════════════════════════════════════════════════════════════
# Parameter accessors (para shape: (n_rows, 3) — columns = agric/forest/urban)
# ═══════════════════════════════════════════════════════════════════════════

def _p(para: np.ndarray, row: int) -> np.ndarray:
    """Return a 1-D array [agric, forest, urban] for parameter row `row`."""
    return para[row, :]   # shape (3,)


# ═══════════════════════════════════════════════════════════════════════════
# Snow module — vectorised across 3 land-use columns per time step
# ═══════════════════════════════════════════════════════════════════════════

def _snow_step(
    sweice: np.ndarray, sweliq: np.ndarray,
    pr: float, airt: float, para: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One sub-time-step of the degree-day snow model for all 3 land-use cols.

    Returns (sweice, sweliq, rainmelt, quickr, interception, totalprecip).
    """
    # Scalar parameters per land-use column (shape (3,))
    tt    = _p(para, 0)   # threshold temp (snow/rain split lower)
    tt2   = _p(para, 1)   # threshold temp upper (all rain)
    crfac = _p(para, 2)   # rain correction factor
    cfac  = _p(para, 3)   # snow correction factor
    whc   = _p(para, 4)   # interception / liquid water holding capacity
    cmelt = _p(para, 5)   # degree-day melt factor  (mm/°C/d)
    cfreez= _p(para, 6)   # degree-day freeze factor
    wfrac = _p(para, 7)   # liquid water fraction in snow

    # ── Partition precip into rain / snow ────────────────────────────────
    # Three zones: airt < tt (all snow), airt > tt2 (all rain), between
    all_snow = airt < tt
    all_rain = airt > tt2
    mixed    = ~all_snow & ~all_rain

    f_rain  = np.where(all_rain, 1.0,
              np.where(mixed, (airt - tt) / np.maximum(tt2 - tt, 1e-9), 0.0))
    f_rain  = np.clip(f_rain, 0.0, 1.0)

    prr = f_rain       * pr * crfac   # liquid precip (m/sub-step)
    prs = (1 - f_rain) * pr * cfac    # solid precip

    totalprecip = prr + prs

    # ── Melt / freeze ────────────────────────────────────────────────────
    melt  = np.where(airt > tt, cmelt  / 1000 * (airt - tt),  0.0) * dt
    freez = np.where(airt < tt, cfreez / 1000 * (tt - airt),  0.0) * dt

    # Interception (fraction of total precip withheld by canopy)
    interception = whc * (prr + prs)

    # ── Update ice / liquid stores ────────────────────────────────────────
    # If melt exceeds available ice → drain everything
    melt_capped   = np.minimum(melt, sweice + (1 - whc) * prs * dt)
    freez_capped  = np.minimum(freez, sweliq)

    sweliq_new = sweliq + ((1 - whc) * prr + melt_capped - freez_capped) * dt
    sweice_new = sweice + ((1 - whc) * prs + freez_capped - melt_capped) * dt

    # Outflow from snowpack when liquid exceeds holding capacity
    outflow      = np.maximum(sweliq_new / dt - wfrac * sweice_new / dt, 0.0)
    sweliq_final = np.minimum(sweliq_new, wfrac * sweice_new)

    # Full melt-out: if all ice gone, push everything out
    fully_melted = sweice_new <= 0
    outflow      = np.where(fully_melted, (sweliq_new + sweice_new) / dt, outflow)
    sweice_final = np.where(fully_melted, 0.0, sweice_new)
    sweliq_final = np.where(fully_melted, 0.0, sweliq_final)

    quickr   = _p(para, 23) * outflow
    rainmelt = outflow - quickr

    return sweice_final, sweliq_final, rainmelt, quickr, interception, totalprecip


# ═══════════════════════════════════════════════════════════════════════════
# HBV soil / routing module — vectorised across 3 land-use columns
# ═══════════════════════════════════════════════════════════════════════════

def _hbv_step(
    sbox: np.ndarray, suz: np.ndarray, slz: np.ndarray,
    swe: np.ndarray,
    rainmelt: np.ndarray, quickr: np.ndarray,
    potetr: float,
    qdel: np.ndarray,           # shape (maxbas_max, 3)
    c:    np.ndarray,           # routing triangle (maxbas_max, 3)
    para: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    One sub-time-step HBV soil/zone/routing update — all 3 land-use cols.

    Returns (sbox, suz, slz, eactual, q_total, qdel, delayedrunoff).
    """
    fc   = _p(para, 8)    # field capacity
    lp   = _p(para, 9)    # limit for potential ET
    beta = _p(para, 10)   # shape of recharge function
    pmax = _p(para, 11)   # percolation threshold (upper zone)
    alpha= _p(para, 12)   # percolation rate
    k0   = _p(para, 13)   # fast runoff coefficient
    k1   = _p(para, 14)   # slow upper-zone runoff coefficient
    k2   = _p(para, 15)   # base-flow coefficient

    # Actual ET — suppressed under snow
    et_fraction = np.where(swe > 0, 0.0,
                  np.where(sbox / np.maximum(fc, 1e-9) > lp, 1.0,
                           sbox / np.maximum(fc * lp, 1e-9)))
    eactual = potetr * et_fraction

    # Recharge from soil to upper zone
    recharge = rainmelt * (sbox / np.maximum(fc, 1e-9)) ** beta
    inf      = rainmelt - recharge
    sbox_new = sbox + dt * (inf - eactual)

    # Upper zone
    q0  = np.where(suz > pmax, k0 * (suz - pmax), 0.0)
    q1  = k1 * suz
    perc_max = suz / dt + recharge - q0 - q1
    perc = np.minimum(alpha, np.maximum(perc_max, 0.0))
    suz_new = np.maximum(suz + dt * (recharge - perc - q0 - q1), 0.0)

    # Lower zone
    q2      = k2 * slz
    slz_new = slz + dt * (perc - q2)

    q_total = q0 + q1 + q2 + quickr   # shape (3,)

    # ── Triangular routing filter (convolution) ───────────────────────────
    # qdel shape: (maxbas, 3); c shape: (maxbas, 3)
    qdel_new        = qdel.copy()
    qdel_new       += c * q_total[np.newaxis, :]   # broadcast over maxbas
    delayedrunoff   = qdel_new[0, :]               # first element = today's output
    qdel_shifted    = np.roll(qdel_new, -1, axis=0)
    qdel_shifted[-1, :] = 0.0

    return sbox_new, suz_new, slz_new, eactual, q_total, qdel_shifted, delayedrunoff


# ═══════════════════════════════════════════════════════════════════════════
# Routing triangle builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_routing(para: np.ndarray, maxbas_max: int = 999) -> np.ndarray:
    """
    Build triangular weighting arrays for all 3 land-use columns.
    Returns c of shape (maxbas_max+1, 3).
    """
    c = np.zeros((maxbas_max + 1, 3), dtype=float)
    for j in range(3):
        mb = min(int(para[16, j]) + 1, maxbas_max)
        for i in range(mb):
            v = 2 / mb - (((i + 1) - mb / 2) ** 2) ** 0.5 * 4 / mb ** 2
            c[i, j] = max(v, 0.0)
        s = c[:mb, j].sum()
        if s > 0:
            c[:mb, j] /= s
    return c


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_hbv_model(met_input_csv: str, catchment_id: str | None = None) -> dict[str, str]:
    """
    Run HBV model — vectorised inner loop.

    Parameters
    ----------
    met_input_csv : path to met_input CSV (Year,Month,Day,Prec_mm/d,Tair_oC,Epot_mm/d)
    catchment_id  : optional ID string, used only in log messages

    Returns
    -------
    dict mapping output key → absolute file path
    """
    label = f"catchment {catchment_id}" if catchment_id else "catchment"
    print(f"[HBV] {label} — reading {met_input_csv}")

    hydroinput = pd.read_csv(met_input_csv).values   # (ndays, 6)
    ndays      = len(hydroinput)

    # hbv_para.csv must be in cwd (hbv_worker.py chdir's there per rank)
    hydropara = pd.read_csv("hbv_para.csv").iloc[:, 0:3].values.astype(float)
    # shape: (n_params, 3) — columns = agric, forest, urban

    n_luse     = 3
    subdtnhbv  = 24
    dt         = 1.0 / subdtnhbv

    MAXBAS_MAX = 999
    c    = _build_routing(hydropara, MAXBAS_MAX)   # (MAXBAS_MAX+1, 3)
    qdel = np.zeros((MAXBAS_MAX + 1, n_luse), dtype=float)

    # ── State vectors — one value per land-use column ─────────────────────
    sweice    = hydropara[18, :] / 1000.0
    sweliq    = np.zeros(n_luse)
    sbox      = hydropara[19, :] / 1000.0
    suz       = hydropara[20, :] / 1000.0
    slz       = hydropara[21, :] / 1000.0

    # ── Output arrays (daily, aggregated from sub-steps) ──────────────────
    totpr_arr      = np.zeros((ndays, n_luse))
    einterc_arr    = np.zeros((ndays, n_luse))
    eactual_arr    = np.zeros((ndays, n_luse))
    delrunoff_arr  = np.zeros((ndays, n_luse))
    runinput_arr   = np.zeros((ndays, n_luse))
    swe_arr        = np.zeros((ndays, n_luse))
    sbox_arr       = np.zeros((ndays, n_luse))
    suz_arr        = np.zeros((ndays, n_luse))
    slz_arr        = np.zeros((ndays, n_luse))

    # ── Main time loop — sub-daily sub-steps, vectorised over 3 land uses ─
    for i in range(ndays):
        row      = hydroinput[i, :]
        airt     = float(row[4])
        pr_day   = float(row[3]) / 1000.0      # mm → m
        potetr = float(np.mean(_p(hydropara, 17)) * row[5] / 1000.0)

        # Accumulate over sub-time-steps
        acc_totpr   = np.zeros(n_luse)
        acc_einterc = np.zeros(n_luse)
        acc_eactual = np.zeros(n_luse)
        acc_delroff = np.zeros(n_luse)
        acc_runinp  = np.zeros(n_luse)

        for _ in range(subdtnhbv):
            pr_sub = pr_day / subdtnhbv   # precip per sub-step (m)

            # Snow
            sweice, sweliq, rainmelt, quickr, interc, totprec = _snow_step(
                sweice, sweliq, pr_sub, airt, hydropara, dt
            )

            # HBV soil/routing
            sbox, suz, slz, eact, q_total, qdel, delroff = _hbv_step(
                sbox, suz, slz,
                sweice + sweliq,
                rainmelt, quickr,
                potetr * dt,
                qdel, c, hydropara, dt,
            )

            acc_totpr   += dt * totprec
            acc_einterc += dt * interc
            acc_eactual += dt * eact
            acc_delroff += dt * delroff
            acc_runinp  += dt * q_total

        totpr_arr[i]     = acc_totpr
        einterc_arr[i]   = acc_einterc
        eactual_arr[i]   = acc_eactual
        delrunoff_arr[i] = acc_delroff
        runinput_arr[i]  = acc_runinp
        swe_arr[i]       = sweice + sweliq
        sbox_arr[i]      = sbox
        suz_arr[i]       = suz
        slz_arr[i]       = slz

    # ── Write per-land-use output CSVs ────────────────────────────────────
    fnames    = ["hbv_output_agric.csv", "hbv_output_forest.csv", "hbv_output_urban.csv"]
    col_scale = 1000.0   # m → mm

    output_files: dict[str, str] = {}

    for j, fout in enumerate(fnames):
        df = DataFrame({
            "Year":                          hydroinput[:, 0].astype(int),
            "Month":                         hydroinput[:, 1].astype(int),
            "Day":                           hydroinput[:, 2].astype(int),
            "total_precipitation mm":        col_scale * totpr_arr[:, j],
            "interception mm":               col_scale * einterc_arr[:, j],
            "actual_evapotranspiration mm":  col_scale * eactual_arr[:, j],
            "delayedrunoff mm":              col_scale * delrunoff_arr[:, j],
            "runoffinput mm":                col_scale * runinput_arr[:, j],
            "snow_water_equivalent mm":      col_scale * swe_arr[:, j],
            "soil_water_storage mm":         col_scale * (sbox_arr[:, j]
                                                          + suz_arr[:, j]
                                                          + slz_arr[:, j]),
            "soilbox mm":                    col_scale * sbox_arr[:, j],
            "supperzone mm":                 col_scale * suz_arr[:, j],
            "slowerzone mm":                 col_scale * slz_arr[:, j],
            "snow depth cm":                 100.0 * swe_arr[:, j] / hydropara[22, j],
        })
        df.to_csv(fout, index=False)
        key = f"landuse_{j}"
        output_files[key] = os.path.abspath(fout)
        print(f"[HBV] {label} — {fout} written")

    # ── Total runoff (weighted sum of delayed runoffs) ────────────────────
    lf = hydropara[24, :]   # land-use area fractions [agric, forest, urban]
    total_runoff = (  delrunoff_arr[:, 0] * lf[0]
                    + delrunoff_arr[:, 1] * lf[1]
                    + delrunoff_arr[:, 2] * lf[2]) * col_scale

    total_fout = "hbv_output_totalrunoff.csv"
    df_total = DataFrame({
        "Year":            hydroinput[:, 0].astype(int),
        "Month":           hydroinput[:, 1].astype(int),
        "Day":             hydroinput[:, 2].astype(int),
        "totalrunoff mm":  total_runoff,
        "agricrunoff mm":  col_scale * delrunoff_arr[:, 0] * lf[0],
        "forestrunoff mm": col_scale * delrunoff_arr[:, 1] * lf[1],
        "urbanrunoff mm":  col_scale * delrunoff_arr[:, 2] * lf[2],
    })
    df_total.to_csv(total_fout, index=False)
    output_files["total_runoff"] = os.path.abspath(total_fout)
    print(f"[HBV] {label} — {total_fout} written")

    return output_files


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point — used by Slurm job array
# ═══════════════════════════════════════════════════════════════════════════
# Slurm calls:
#   python hbv_S2S.py --met-csv /data/hbv/inputs/catchment_123.csv \
#                     --task 2 --total 4 --output /data/hbv/output/12345
#
# --task / --total split a list of met-csv files across nodes:
#   node 0 → files[0::4], node 1 → files[1::4], etc.
# A single --met-csv bypasses the split (single-catchment runs).

if __name__ == '__main__':
    import glob
    import shutil

    parser = argparse.ArgumentParser(description='HBV-S2S model runner')
    parser.add_argument('--met-csv',  help='Path to one met-input CSV file')
    parser.add_argument('--met-dir',  help='Directory of met-input CSV files (batch mode)')
    parser.add_argument('--para-csv', default=None,
                        help='Path to hbv_para.csv (defaults to the copy beside this script)')
    parser.add_argument('--task',     type=int, default=0,
                        help='Slurm array task index (0-based)')
    parser.add_argument('--total',    type=int, default=1,
                        help='Total number of Slurm array tasks')
    parser.add_argument('--output',   default='.',
                        help='Output directory for result CSVs')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # hbv_S2S.py reads hbv_para.csv from cwd — copy it there before chdir
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    para_src = args.para_csv or os.path.join(_script_dir, 'hbv_para.csv')
    para_dst = os.path.join(os.path.abspath(args.output), 'hbv_para.csv')
    if not os.path.exists(para_dst):
        shutil.copy2(para_src, para_dst)

    os.chdir(args.output)

    if args.met_csv:
        # single-catchment run
        files = [args.met_csv]
    elif args.met_dir:
        # batch run: each task processes its slice
        all_files = sorted(glob.glob(os.path.join(args.met_dir, '*.csv')))
        files = all_files[args.task::args.total]
        if not files:
            print(f'[HBV] task {args.task}/{args.total}: no files to process', flush=True)
            sys.exit(0)
    else:
        parser.error('Provide --met-csv or --met-dir')

    for csv_path in files:
        catchment_id = os.path.splitext(os.path.basename(csv_path))[0]
        print(f'[HBV] task {args.task}/{args.total} processing {catchment_id}', flush=True)
        out = run_hbv_model(csv_path, catchment_id=catchment_id)
        for k, v in out.items():
            print(f'[HBV]   {k}: {v}', flush=True)

    print(f'[HBV] task {args.task}/{args.total} done', flush=True)
