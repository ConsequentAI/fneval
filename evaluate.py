from loader import load_mod
from eval.measure_math import Eval
import json
import argparse
from typing import List, Dict, Any, Tuple
from persist import Persist


DEFAULT_MODEL_SPECS_FILE = 'evaluated_models.json'
DEFAULT_SNAPSHOTS_FILE = 'monthly_snapshots.json'
DEFAULT_EVAL_PICKLE_FILE = DEFAULT_MODEL_SPECS_FILE + '.pickle'

DEFAULT_FEW_SHOT_NUM = 5
DEFAULT_TEMPERATURE = 0.0

SPEC_SCRIPT = 'script'
SPEC_INCLUDE = 'include'
SPEC_NAME = 'name'
SPEC_COT = 'CoT'
SPEC_FEW_SHOT = 'few shot'
SPEC_TEMP = 'temperature'
SPEC_EXTRA_PARAMS = 'extra_params'
SPEC_EXTRA_PARAMS_MODE_WRITE_OFFLINE = 'write_qs'


class ModelSpec:
    def __init__(self, spec):
        self.script = spec[SPEC_SCRIPT]
        self.model = spec[SPEC_NAME]
        self.chain_of_thought = spec[SPEC_COT]
        self.few_shot_num = spec[SPEC_FEW_SHOT] if SPEC_FEW_SHOT in spec else DEFAULT_FEW_SHOT_NUM
        self.temperature = spec[SPEC_TEMP] if SPEC_TEMP in spec else DEFAULT_TEMPERATURE
        self.extra_params = json.loads(spec[SPEC_EXTRA_PARAMS]) if SPEC_EXTRA_PARAMS in spec else {}

    def ident(self) -> str:
        cot_yn = 'yes' if self.chain_of_thought else 'no'
        return self.model + f'_cot={cot_yn}_fs={self.few_shot_num}_temp={self.temperature}'

    def __repr__(self) -> str:
        return self.ident()

    def spec_csv(self) -> Tuple[str,str]:
        hdr = f'model,cot,fewshot,temp'
        row = f'{self.model},{self.chain_of_thought},{self.few_shot_num},{self.temperature}'
        return hdr,row

class ModelRunners:
    def __init__(self, spec_file: str, snapshots_specs: str, verbose: bool, save_snapshot: bool):
        self.verbose = verbose
        self.save_snapshot = save_snapshot
        self.snapshots_specs = snapshots_specs

        with open(spec_file, 'r') as f:
            js = json.load(f)
        script_names = set(spec[SPEC_SCRIPT] for spec in js if spec[SPEC_INCLUDE] )
        model_specs = [ ModelSpec(spec) for spec in js if spec[SPEC_INCLUDE] ]

        models_for = lambda sc: [ ms for ms in model_specs if ms.script == sc ]

        self.models: Dict[str, List[ModelSpec]] = { script: models_for(script) for script in script_names }
        self.scripts: Dict[str, Any] = { script: load_mod(script) for script in script_names }

    def run(self, ms: ModelSpec) -> Eval:
        self.banner(f"Evaluating {ms.ident()}")
        runner = self.scripts[ms.script].Answers.run
        return runner(ms.model,
                self.snapshots_specs,
                ms.chain_of_thought,
                ms.few_shot_num,
                ms.temperature,
                self.verbose,
                self.save_snapshot,
                ms.extra_params)

    def run_models_for(self, script: str) -> Dict[ModelSpec, Eval]:
        return { ms: self.run(ms) for ms in self.models[script] }

    def api_script_names(self) -> List[str]:
        return list(self.scripts.keys())

    def banner(self, msg):
        hr = '\n' + '*' * 80 + '\n'
        print(f'{hr}{msg}{hr}')


class Evaluate:
    def __init__(self, spec_file: str, snapshots_specs: str, verbose: bool, save_snapshot: bool):
        self.model_runners = ModelRunners(spec_file, snapshots_specs, verbose, save_snapshot)

    def build_evals(self) -> Dict[ModelSpec, Eval]:
        api_scripts: List[str] = self.model_runners.api_script_names()
        evals: List[Dict[ModelSpec, Eval]] = list(map(self.rfn, api_scripts))
        return { ms: e for evd in evals for ms, e in evd.items() }

    def rfn(self, sc) -> Dict[ModelSpec, Eval]:
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

    outfile = f'{args.model_specs}.pickle'
    e = Evaluate(args.model_specs, args.snapshots_specs, args.verbose, args.save_snapshot)
    evals: Dict[ModelSpec, Eval] = e.build_evals()
    Persist.save(evals, outfile, force_overwrite = True)
    ofile_msg = f'--in_file {outfile}' if outfile != DEFAULT_EVAL_PICKLE_FILE else ''
    print(f'Evals written. Run python3 -m summarize_evals {ofile_msg} to see summary.')
