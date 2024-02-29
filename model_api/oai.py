from eval.runner import EvalRunner, mk_dir_safe
import argparse
import sys
from typing import Dict, List, Tuple
from math_utils.math_helpers import rm_latex_math
from unformatted_llm import UnformattedLLM
from model_api.gpt import GPT, OAI_MODELS
from model_api.closed_api import KnownModel, ParameterizedModel
from evaluate import DEFAULT_SNAPSHOTS_FILE

WHO_ARE_YOU = "You are a expert in mathematics, logic, and formal reasoning. " \
              "You are incredibly concise and formal at writing math statements."

class Answers:
    def __init__(self, model: KnownModel, few_shot_num: int = -1):
        self.model = model
        self.unformatted = UnformattedLLM()
        few_shot: Dict[str, str] = self.unformatted.few_shot_limited(few_shot_num)
        params = ParameterizedModel(
            temp = 0.0,
            instruction = self.unformatted.INSTRUCTION,
            who_are_you = WHO_ARE_YOU,
            few_shot = few_shot,
            model = model,
            agent_name = f"OAI_Eval_{mk_dir_safe(model.name)}",
        )
        self.gpt = GPT(params)

    def answers(self, prb: str) -> List[Tuple[str, str]]:
        try:
            completion = self.gpt.query([prb], rate_limited = False)[0]
        except Exception as e:
            if self.known_error(e):
                completion = ""
            else:
                raise e
        answer = self.unformatted.extract_answer(completion)
        return [(answer, completion)]

    def known_error(self, e) -> bool:
        KNOWN = [
                "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."
        ]
        print(f'API ERROR: {str(e)}')
        return str(e) in KNOWN

    @classmethod
    def run(cls, name: str, snapshots_specs: str, verbose: bool = False, save_snaphot: bool = False):
        known_model = OAI_MODELS.SHORT_NAMES[name]
        answerer = Answers(known_model, few_shot_num = 5)
        e = EvalRunner(answerer, snapshots_specs, verbose).do()
        if save_snaphot:
            answerer.gpt.snapshot_api_query_cache()
        return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
            help = f"Model has to be one of {list(OAI_MODELS.SHORT_NAMES.keys())}")
    parser.add_argument("--snapshots_specs", type=str, default=DEFAULT_SNAPSHOTS_FILE,
            help = f"JSON of static and monthly snapshots")
    parser.add_argument("--verbose", action='store_true',
                        help="Outcomes of each test, one per line, on console")
    parser.add_argument("--save_snapshot", action='store_true',
                        help="Save API query snapshot as .tar.gz file")
    args = parser.parse_args()

    Answers.run(args.model, args.snapshots_specs, args.verbose, args.save_snapshot)
