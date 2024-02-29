from __future__ import annotations
from typing import List, Optional, Tuple
from eval.measure_math import Eval
from helper_utils import NO_SOLUTION_BAD_FORMAT
from fn_snapshot import FnSnapshot

HR = "\n" + '-' * 80 + "\n"
BENCHMARK = "MATH"

class RunEval:
    def __init__(self):
        self.model_name = None
        self.snapshots = None
        self.verbose = True

    def do(self):
        return Eval(self.solve_fn,
                self.snapshots,
                self.model_name,
                verbose = self.verbose)

    def solve_fn(self, prb: str) -> List[Tuple[str, str]]:
        answers: List[Tuple[str, str]] = self.answers(prb)
        # If no answers return indicator that we got no answer section
        if not answers:
            return [(NO_SOLUTION_BAD_FORMAT, "")]
        return answers

    def answers(self, prb: str) -> List[Tuple[str, str]]: # type: ignore [empty-body]
        pass

mk_dir_safe = lambda x: x.replace('/', '_')

class EvalRunner(RunEval):
    def __init__(self, answerer, snapshots_specs, verbose):
        print(f'Reading specs from: {snapshots_specs}')
        self.answers = answerer.answers # type: ignore [method-assign]
        self.model_name = mk_dir_safe(answerer.model.name)
        self.snapshots = FnSnapshot.load(BENCHMARK, snapshots_specs)
        self.verbose = verbose
