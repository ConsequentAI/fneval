from eval.runner import EvalRunner, mk_dir_safe
import argparse
import sys
from typing import Dict, List, Tuple
from math_utils.math_helpers import rm_latex_math
from unformatted_llm import UnformattedLLM
from model_api.closed_api import KnownModel, ParameterizedModel
from model_api.claude import Claude, ANTHROPIC_PARAMS
from evaluate import DEFAULT_SNAPSHOTS_FILE
from chain_of_thought import COT_INSTRUCTION, ChainOfThought

PREAMBLE = f"QQuery. Answer the MATH query below. "
INSTRUCTION = PREAMBLE + COT_INSTRUCTION

class Answers:
    def __init__(self, model: KnownModel, few_shot_num: int = -1):
        self.model = model
        self.cot = ChainOfThought()
        few_shot: Dict[str, str] = self.cot.few_shot_limited(few_shot_num)
        params = ParameterizedModel(
                temp = 0.7,
                instruction = INSTRUCTION,
                who_are_you = "",
                few_shot = few_shot,
                model = model,
                agent_name = f"Anthropic_Eval_{mk_dir_safe(model.name)}",
        )
        self.claude = Claude(params)

    def answers(self, prb: str) -> List[Tuple[str, str]]:
        completion = self.claude.query([prb], rate_limited = False)[0]
        answer = self.claude.extract_answer(completion)
        answer = self.cot.extract_answer(answer)
        return [(answer, completion)]

    @classmethod
    def run(cls, name: str, snapshots_specs: str, verbose: bool = False, save_snaphot: bool = False):
        known_model = ANTHROPIC_PARAMS.SHORT_NAMES[name]
        # 0-shot apparently works; so no examples needed
        answerer = Answers(known_model, few_shot_num = 0) # few_shot_num = 5
        e = EvalRunner(answerer, snapshots_specs, verbose).do()
        if save_snaphot:
            answerer.claude.snapshot_api_query_cache()
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
