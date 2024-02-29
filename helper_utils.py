import os
import subprocess
from typing import Tuple, Optional, List

NO_SOLUTION_PREFIX = "NO_SOLUTION"
NO_SOLUTION_BAD_FORMAT = f"{NO_SOLUTION_PREFIX} [BAD FORMAT]"
NO_SOLUTION_DIMS = f"{NO_SOLUTION_PREFIX} [DIMS]"
NO_SOLUTION_EXCEED_CONTEXT = f"{NO_SOLUTION_PREFIX} [EXCEED CONTEXT]"

def download_extract(url: str, unzipped_dir: str, downloaded_targz: str, root: str, gz: bool = True):
    assert not os.path.exists(unzipped_dir), f'Cannot download extract to {unzipped_dir}. Directory already exists'
    tar_flags = '-zxvf' if gz else '-xvf'
    get = ['curl', url, '-o', downloaded_targz] if url.startswith("http") else ['cp', url, downloaded_targz]
    unzip = ['tar', tar_flags, downloaded_targz]
    mv = ['mv', unzipped_dir, f'{root}']
    for cmd in [get, unzip, mv]:
        print(f'Running: {cmd}')
        subprocess.call(cmd)

def targz(dirname: str):
    tar_name = dirname + ".tar"
    targz_name = tar_name + ".gz"
    assert not os.path.exists(tar_name), f'Cannot tar to {tar_name}. File already exists'
    assert not os.path.exists(targz_name), f'Cannot gz to {targz_name}. File already exists'
    tar = ['tar', '-cvf', tar_name, dirname]
    gzip = ['gzip', tar_name]
    subprocess.call(tar)
    subprocess.call(gzip)

def untargz(targz_name: str, dst: str):
    ext = ".tar.gz"
    assert targz_name.endswith(ext)
    extract_name = targz_name[:-len(ext)]
    assert not os.path.exists(extract_name), f'Cannot extract to {extract_name}. Directory already exists'
    tar_flags = '-zxvf'
    unzip = ['tar', tar_flags, targz_name]
    subprocess.call(unzip)

    if extract_name != dst:
        print(f'Moving {extract_name} to {dst}')
        assert not os.path.exists(dst), f'Cannot mv to {dst}. Directory already exists'
        mv = ['mv', extract_name, dst]
        subprocess.call(mv)
