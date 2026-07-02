"""
puhti.py — SSH-based Slurm wrapper for Puhti supercomputer.

Replaces direct sbatch/squeue/scancel calls with SSH equivalents.
All file transfers go via rsync over SSH.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

PUHTI_HOST    = os.environ.get('PUHTI_HOST',    'puhti.csc.fi')
PUHTI_USER    = os.environ.get('PUHTI_USER',    'javedham')
PUHTI_KEY     = os.environ.get('PUHTI_SSH_KEY', '/home/hbv/.ssh/id_puhti')
PUHTI_SCRATCH = os.environ.get('PUHTI_SCRATCH', '/scratch/project_XXXXXXX/hbv')
PUHTI_PROJECT = os.environ.get('PUHTI_PROJECT', 'project_XXXXXXX')

_SSH_BASE = [
    'ssh', '-i', PUHTI_KEY,
    '-o', 'StrictHostKeyChecking=no',
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=15',
    f'{PUHTI_USER}@{PUHTI_HOST}',
]


def ssh_run(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a shell command on Puhti via SSH."""
    return subprocess.run(
        _SSH_BASE + [cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def sbatch(script: str, env: dict[str, str], array: str | None = None,
           extra_args: list[str] | None = None) -> str:
    """
    Submit a job to Puhti Slurm. Returns the Slurm job ID string.
    env vars are passed as --export=KEY=VALUE arguments.
    """
    export = ','.join(f'{k}={v}' for k, v in env.items())
    cmd_parts = [f'sbatch --export={export}']
    if array:
        cmd_parts.append(f'--array={array}')
    if extra_args:
        cmd_parts.extend(extra_args)
    cmd_parts.append(script)

    r = ssh_run(' '.join(cmd_parts))
    if r.returncode != 0:
        raise RuntimeError(f'sbatch failed: {r.stderr.strip()}')
    # "Submitted batch job 12345" → "12345"
    return r.stdout.strip().split()[-1]


def squeue(job_id: str) -> str | None:
    """Return Slurm state string for a job, or None if not found."""
    r = ssh_run(f'squeue -j {job_id} -h --format=%T', timeout=15)
    out = r.stdout.strip()
    return out if out else None


def scancel(job_id: str) -> None:
    """Cancel a Slurm job on Puhti."""
    r = ssh_run(f'scancel {job_id}', timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f'scancel failed: {r.stderr.strip()}')


def sinfo() -> subprocess.CompletedProcess:
    """Get cluster node info from Puhti."""
    return ssh_run('sinfo -h --Node -o "%N %t %C %m"', timeout=15)


def rsync_to_puhti(local_path: str, remote_path: str, timeout: int = 300) -> None:
    """Copy a file or directory from head node to Puhti scratch."""
    subprocess.run([
        'rsync', '-az', '--mkpath',
        '-e', f'ssh -i {PUHTI_KEY} -o StrictHostKeyChecking=no -o BatchMode=yes',
        local_path,
        f'{PUHTI_USER}@{PUHTI_HOST}:{remote_path}',
    ], check=True, timeout=timeout)


def rsync_from_puhti(remote_path: str, local_path: str, timeout: int = 300) -> None:
    """Copy results from Puhti scratch back to head node."""
    os.makedirs(local_path, exist_ok=True)
    subprocess.run([
        'rsync', '-az',
        '-e', f'ssh -i {PUHTI_KEY} -o StrictHostKeyChecking=no -o BatchMode=yes',
        f'{PUHTI_USER}@{PUHTI_HOST}:{remote_path}/',
        local_path + '/',
    ], check=True, timeout=timeout)


def remote_path(local_path: str) -> str:
    """
    Map a local /data/hbv/... path to its Puhti scratch equivalent.
    e.g. /data/hbv/uploads/abc → /scratch/project_XXXXXXX/hbv/uploads/abc
    """
    rel = os.path.relpath(local_path, '/data/hbv')
    return os.path.join(PUHTI_SCRATCH, rel)
