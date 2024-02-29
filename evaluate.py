from loader import load_mod
from eval.measure_math import Eval
import json
import argparse
from typing import List, Dict, Any
from persist import Persist


DEFAULT_MODEL_SPECS_FILE = 'evaluated_models.json'
DEFAULT_SNAPSHOTS_FILE = 'monthly_snapshots.json'
DEFAULT_EVAL_PICKLE_FILE = DEFAULT_MODEL_SPECS_FILE + '.pickle'


class ModelSpec:
    def __init__(self, model: str, script: str):
        self.model = model
        self.script = script


class ModelRunners:
    def __init__(self, spec_file: str, snapshots_specs: str, verbose: bool, save_snapshot: bool):
        self.verbose = verbose
        self.save_snapshot = save_snapshot
        self.snapshots_specs = snapshots_specs

        with open(spec_file, 'r') as f:
            js = json.load(f)
        script_names = set(spec["script"] for spec in js if spec["include"] )
        model_specs = [ ModelSpec(spec["name"], spec["script"]) for spec in js if spec["include"] ]

        models_for = lambda sc: [ ms for ms in model_specs if ms.script == sc ]

        self.models: Dict[str, List[ModelSpec]] = { script: models_for(script) for script in script_names }
        self.scripts: Dict[str, Any] = { script: load_mod(script) for script in script_names }

    def run(self, ms: ModelSpec) -> Eval:
        self.banner(f"Evaluating {ms.model}")
        return self.scripts[ms.script].Answers.run(ms.model, self.snapshots_specs, self.verbose, self.save_snapshot)

    def run_models_for(self, script: str) -> Dict[str, Eval]:
        return { ms.model: self.run(ms) for ms in self.models[script] }

    def api_script_names(self) -> List[str]:
        return list(self.scripts.keys())

    def banner(self, msg):
        hr = '\n' + '*' * 80 + '\n'
        print(f'{hr}{msg}{hr}')


class Evaluate:
    def __init__(self, spec_file: str, snapshots_specs: str, verbose: bool, save_snapshot: bool):
        self.model_runners = ModelRunners(spec_file, snapshots_specs, verbose, save_snapshot)

    def build_evals(self) -> Dict[str, Eval]:
        api_scripts: List[str] = self.model_runners.api_script_names()
        evals: List[Dict[str, Eval]] = list(map(self.rfn, api_scripts))
        return { m: e for evd in evals for m, e in evd.items() }

    def rfn(self, sc):
        return self.model_runners.run_models_for(sc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_specs", type=str, default=DEFAULT_MODEL_SPECS_FILE,
            help = f"JSON of model specs, name + run script + enabled")
    parser.add_argument("--snapshots_specs", type=str, default=DEFAULT_SNAPSHOTS_FILE,
            help = f"JSON of static and monthly snapshots")
    parser.add_argument("--verbose", action='store_true', default=False,
                        help="Outcomes of each test, one per line, on console")
    parser.add_argument("--save_snapshot", action='store_true', default=False,
                        help="Save API query snapshot as .tar.gz file")
    args = parser.parse_args()

    model_specs = args.model_specs if args.model_specs else DEFAULT_MODEL_SPECS_FILE
    outfile = f'{model_specs}.pickle'

    e = Evaluate(args.model_specs, args.snapshots_specs, args.verbose, args.save_snapshot)
    evals = e.build_evals()
    Persist.save(evals, outfile, force_overwrite = True)
    ofile_msg = f'--in_file {outfile}' if outfile != DEFAULT_EVAL_PICKLE_FILE else ''
    print(f'Evals written. Run python3 -m summarize_evals {ofile_msg} to see summary.')
