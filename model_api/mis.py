from eval.runner import EvalRunner, mk_dir_safe
import argparse
import sys
from typing import Dict, List, Tuple
from math_utils.math_helpers import rm_latex_math
from unformatted_llm import UnformattedLLM
from model_api.closed_api import KnownModel, ParameterizedModel
from model_api.mistral import Mistral, MISTRAL_PARAMS
from evaluate import DEFAULT_SNAPSHOTS_FILE

_MISTRAL = MISTRAL_PARAMS()

class Answers:
    def __init__(self, model: KnownModel, few_shot_num: int = -1):
        self.model = model
        few_shot: Dict[str, str] = UnformattedLLM().few_shot_limited(few_shot_num)
        params = ParameterizedModel(
                temp = 0.7,
                instruction = UnformattedLLM.INSTRUCTION,
                who_are_you = "",
                few_shot = few_shot,
                model = model,
                agent_name = f"Mistral_Eval_{mk_dir_safe(model.name)}",
        )
        self.mistral = Mistral(params)

    def answers(self, prb: str) -> List[Tuple[str, str]]:
        completion = self.mistral.query([prb], rate_limited = False)[0]
        answer = self.mistral.extract_answer(completion)
        return [(answer, completion)]

    @classmethod
    def run(cls, name: str, snapshots_specs: str, verbose: bool = False, save_snaphot: bool = False):
        known_model = _MISTRAL.models[name]
        answerer = Answers(known_model, few_shot_num = 5)
        e = EvalRunner(answerer, snapshots_specs, verbose).do()
        if save_snaphot:
            answerer.mistral.snapshot_api_query_cache()
        return e


if __name__ == "__main__":
    models_msg = f"Model has to be one of {list(_MISTRAL.models.keys())}"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help=models_msg)
    parser.add_argument("--snapshots_specs", type=str, default=DEFAULT_SNAPSHOTS_FILE,
            help = f"JSON of static and monthly snapshots")
    parser.add_argument("--verbose", action='store_true',
                        help="Outcomes of each test, one per line, on console")
    parser.add_argument("--save_snapshot", action='store_true',
                        help="Save API query snapshot as .tar.gz file")
    args = parser.parse_args()

    if not args.model in _MISTRAL.models:
        print(models_msg)
        exit(-1)

    Answers.run(args.model, args.snapshots_specs, args.verbose, args.save_snapshot)
