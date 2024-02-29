import sys
import importlib.util

def load_mod(fname):
    nm = fname
    spec = importlib.util.spec_from_file_location(nm, fname)
    assert spec, f'Could not load from file: {fname}'
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nm] = mod
    assert spec.loader, f'Spec does not have loader'
    spec.loader.exec_module(mod)
    return mod
