from eval.runner import EvalRunner, mk_dir_safe
import argparse
import sys
from typing import Dict, List, Tuple
from math_utils.math_helpers import rm_latex_math
from unformatted_llm import UnformattedLLM
from model_api.closed_api import KnownModel, ParameterizedModel
from model_api.claude2 import Claude2, ANTHROPIC_PARAMS
from evaluate import DEFAULT_SNAPSHOTS_FILE

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
                agent_name = f"Anthropic_Eval_{mk_dir_safe(model.name)}",
        )
        self.claude2 = Claude2(params)

    def first_word_opt(self, answer: str) -> str:
        # claude2 outputs a lot of junk after the answer
        # split on whitespace and make the first word the answer if a first word exists
        many = answer.split()
        if many and many[0]:
            answer = many[0]
        return answer

    def answers(self, prb: str) -> List[Tuple[str, str]]:
        completion = self.claude2.query([prb], rate_limited = False)[0]
        answer = self.claude2.extract_answer(completion)
        answer = self.first_word_opt(answer)
        return [(answer, completion)]

    @classmethod
    def run(cls, name: str, snapshots_specs: str, verbose: bool = False, save_snaphot: bool = False):
        known_model = ANTHROPIC_PARAMS.SHORT_NAMES[name]
        answerer = Answers(known_model, few_shot_num = 5)
        e = EvalRunner(answerer, snapshots_specs, verbose).do()
        if save_snaphot:
            answerer.claude2.snapshot_api_query_cache()
        return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
            help = f"Model has to be one of {list(ANTHROPIC_PARAMS.SHORT_NAMES.keys())}")
    parser.add_argument("--snapshots_specs", type=str, default=DEFAULT_SNAPSHOTS_FILE,
            help = f"JSON of static and monthly snapshots")
    parser.add_argument("--verbose", action='store_true',
                        help="Outcomes of each test, one per line, on console")
    parser.add_argument("--save_snapshot", action='store_true',
                        help="Save API query snapshot as .tar.gz file")
    args = parser.parse_args()

    Answers.run(args.model, args.snapshots_specs, args.verbose, args.save_snapshot)
