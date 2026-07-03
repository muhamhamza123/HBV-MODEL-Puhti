"""
api_client.py — thin wrapper around the HBV FastAPI gateway.

The dashboard imports this instead of calling hbv_worker directly.
Set HBV_API_URL in the notebook environment to point at the HPC head node,
e.g. HBV_API_URL=http://hpc-head.example.fi:8000
"""

from __future__ import annotations

import os
import time

import requests

API_URL = os.environ.get('HBV_API_URL', 'http://localhost:8000').rstrip('/')
_TIMEOUT = 30        # seconds for non-streaming requests
_SUBMIT_TIMEOUT = 600  # submit can take up to 10 min (rsync large files to Puhti)


def _headers(json=True) -> dict:
    """
    Send X-User header for API auth.
    On JupyterHub, JUPYTERHUB_USER is set automatically.
    For local dev, set HBV_DEV_USER.
    """
    h = {}
    if json:
        h['Content-Type'] = 'application/json'
    user = (os.environ.get('HBV_DEV_USER')
            or os.environ.get('JUPYTERHUB_USER')
            or '')
    if user:
        h['X-User'] = user
    return h


def upload_file(local_path: str, progress_cb=None) -> str:
    """
    Upload a local file to the HPC NFS via POST /upload.
    Returns the NFS path the API assigned.

    progress_cb(bytes_sent, total_bytes) is called periodically if provided.
    Files are cached by MD5 on the server — re-uploading the same file is instant.
    """
    file_size = os.path.getsize(local_path)
    filename   = os.path.basename(local_path)

    with open(local_path, 'rb') as fh:
        resp = requests.post(
            f'{API_URL}/upload',
            headers=_headers(json=False),
            files={'file': (filename, fh, 'application/octet-stream')},
            timeout=3600,
        )
    if progress_cb:
        progress_cb(file_size, file_size)
    resp.raise_for_status()
    return resp.json()['path']


def upload_shapefile_dir(shp_path: str, progress_cb=None) -> str:
    """
    Zip the directory containing the shapefile (picks up .dbf .shx .prj etc.)
    and upload as a single zip. Returns the NFS path of the extracted .shp file.
    """
    import tempfile, zipfile as _zf

    shp_dir  = os.path.dirname(shp_path)
    basename = os.path.splitext(os.path.basename(shp_path))[0]

    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        tmp_zip = tmp.name

    with _zf.ZipFile(tmp_zip, 'w', _zf.ZIP_DEFLATED) as zf:
        for fn in os.listdir(shp_dir):
            if os.path.splitext(fn)[0] == basename:
                zf.write(os.path.join(shp_dir, fn), fn)

    try:
        return upload_file(tmp_zip, progress_cb=progress_cb)
    finally:
        os.remove(tmp_zip)


def get_shapefiles() -> list[dict]:
    """GET /shapefiles — returns pre-registered shapefiles on shared NFS."""
    resp = requests.get(f'{API_URL}/shapefiles',
                        headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_cluster_info() -> dict:
    """GET /cluster/info — live Slurm node/CPU availability."""
    resp = requests.get(f'{API_URL}/cluster/info',
                        headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def submit_job(
    catchment_ids: list[str],
    shapefile_path: str,
    id_col: str = 'TASO_ID',
    precipitation_nc: str = '',
    evapotranspiration_nc: str = '',
    temperature_nc: str = '',
    urban_land_path: str = '',
    agricultural_land_path: str = '',
    hbvpara_path: str | None = None,
    n_nodes: int = 4,
    cpus_per_task: int = 4,
    partition: str = 'small',
) -> dict:
    """
    POST /submit — returns the job dict with job_id and initial status.
    Raises requests.HTTPError on API errors (including 429 rate-limit).
    """
    payload: dict = {
        'catchment_ids':          catchment_ids,
        'shapefile_path':         shapefile_path,
        'id_col':                 id_col,
        'precipitation_nc':       precipitation_nc,
        'evapotranspiration_nc':  evapotranspiration_nc,
        'temperature_nc':         temperature_nc,
        'urban_land_path':        urban_land_path,
        'agricultural_land_path': agricultural_land_path,
        'n_nodes':                n_nodes,
        'cpus_per_task':          cpus_per_task,
        'partition':              partition,
    }
    if hbvpara_path:
        payload['hbvpara_path'] = hbvpara_path

    resp = requests.post(f'{API_URL}/submit', json=payload,
                         headers=_headers(), timeout=_SUBMIT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_status(job_id: str) -> dict:
    """GET /status/{job_id}"""
    resp = requests.get(f'{API_URL}/status/{job_id}',
                        headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_results(job_id: str) -> dict:
    """GET /results/{job_id} — call only after status == 'done'."""
    resp = requests.get(f'{API_URL}/results/{job_id}',
                        headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def download_csv(job_id: str, catchment: str, filename: str) -> bytes:
    """GET /download/{job_id}/{catchment}/{filename} — returns CSV bytes."""
    resp = requests.get(f'{API_URL}/download/{job_id}/{catchment}/{filename}',
                        headers=_headers(json=False), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def my_jobs() -> list[dict]:
    """GET /jobs/mine"""
    resp = requests.get(f'{API_URL}/jobs/mine',
                        headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def cancel_job(job_id: str) -> None:
    """DELETE /jobs/{job_id}"""
    resp = requests.delete(f'{API_URL}/jobs/{job_id}',
                           headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()


def get_job_resources(job_id: str) -> dict:
    """GET /jobs/{job_id}/resources — live Slurm node/CPU/memory stats."""
    resp = requests.get(f'{API_URL}/jobs/{job_id}/resources',
                        headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_system_stats() -> dict:
    """GET /system/stats — disk usage for current user and overall NFS."""
    resp = requests.get(f'{API_URL}/system/stats',
                        headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_logs(job_id: str, offset: int = 0) -> dict:
    """GET /logs/{job_id}?offset=N — returns {lines: [...], next_offset: N}"""
    resp = requests.get(f'{API_URL}/logs/{job_id}', params={'offset': offset},
                        headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def wait_for_job(job_id: str, poll_interval: float = 10.0,
                 on_tick=None) -> dict:
    """
    Block until the job reaches a terminal state (done/failed/cancelled).
    on_tick(status_dict) is called on each poll if provided — use it to
    update a progress widget.
    Returns the final status dict.
    """
    while True:
        status = get_status(job_id)
        if on_tick:
            on_tick(status)
        if status['status'] in ('done', 'failed', 'cancelled'):
            return status
        time.sleep(poll_interval)
