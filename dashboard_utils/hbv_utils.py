import os, sys, shutil


def parse_predata(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line == 'input_text':
                continue
            parts = line.split(',', 1)
            if len(parts) == 2:
                rows.append((parts[0].strip(), parts[1].strip()))
    return dict(rows)


def generate_predata_csv(cfg, temp_dir):
    path = os.path.join(temp_dir, 'predata.csv')
    with open(path, 'w') as f:
        f.write('input_text\n')
        for k, v in cfg.items():
            f.write(f'{k},{v}\n')
    return path


def find_mpirun():
    venv_bin = os.path.join(sys.prefix, 'bin')
    for name in ('mpiexec.hydra', 'mpiexec', 'mpirun'):
        p = os.path.join(venv_bin, name)
        if os.path.exists(p):
            return p
    for name in ('mpiexec.hydra', 'mpiexec', 'mpirun'):
        p = shutil.which(name)
        if p:
            return p
    return None
