from eval.runner import EvalRunner, mk_dir_safe
import argparse
import sys
from typing import Dict, List, Tuple
from math_utils.math_helpers import rm_latex_math
from unformatted_llm import UnformattedLLM
from model_api.closed_api import KnownModel, ParameterizedModel
from model_api.oss import OSS, TOGETHER_PARAMS
from evaluate import DEFAULT_SNAPSHOTS_FILE, DEFAULT_FEW_SHOT_NUM, DEFAULT_TEMPERATURE
from chain_of_thought import COT_INSTRUCTION, ChainOfThought

_TOGETHER = TOGETHER_PARAMS()

KNOWN_CRASH="400 Client Error: Bad Request for url: https://api.together.xyz/api/inference"

class Answers:
    def __init__(self, model: KnownModel, use_cot: bool, few_shot_num: int, temperature: float):
        self.model = model
        self.use_cot = use_cot
        self.cot = ChainOfThought()
        unformatted = UnformattedLLM()

        few_shot_builder = self.cot if use_cot else unformatted
        instruction = COT_INSTRUCTION if use_cot else unformatted.INSTRUCTION
        few_shot: Dict[str, str] = few_shot_builder.few_shot_limited(few_shot_num)
        cache_infer_params = f'temp={temperature}_cot={use_cot}_fs={few_shot_num}'
        cache_model_name = mk_dir_safe(model.name)
        params = ParameterizedModel(
                temp = temperature,
                instruction = instruction,
                who_are_you = "",
                few_shot = few_shot,
                model = model,
                agent_name = f"Together_Eval_{cache_model_name}_{cache_infer_params}",
        )
        self.oss = OSS(params)

    def answers(self, prb: str) -> List[Tuple[str, str]]:
        try:
            completion = self.oss.query([prb], rate_limited = False)[0]
        except Exception as he:
            if f'{he}' == KNOWN_CRASH:
                # print(f'[WARN] CRASH on prb below\n-----\n{prb}\n-----\nIgnoring and Continue!')
                completion = ""
            else:
                raise he
        answer = self.oss.extract_answer(completion)
        if self.use_cot:
            answer = self.cot.extract_answer(answer)
        return [(answer, completion)]

    @classmethod
    def run(cls, name: str, snapshots_specs: str,
            cot: bool, few_shot_num: int, temperature: float,
            verbose: bool = False, save_snaphot: bool = False):
        known_model = _TOGETHER.models[name]
        answerer = Answers(known_model, cot, few_shot_num, temperature)
        e = EvalRunner(answerer, snapshots_specs, verbose).do()
        if save_snaphot:
            answerer.oss.snapshot_api_query_cache()
        return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
            help = f"Model has to be one of {list(_TOGETHER.models.keys())}")
    parser.add_argument("--snapshots_specs", type=str, default=DEFAULT_SNAPSHOTS_FILE,
            help = f"JSON of static and monthly snapshots")
    parser.add_argument("--verbose", action='store_true',
            help="Outcomes of each test, one per line, on console")
    parser.add_argument("--use_chain_of_thought", action='store_true',
            help="Use chain of thought instruction and postprocessing")
    parser.add_argument("--few_shot_num", type=str, default=DEFAULT_FEW_SHOT_NUM,
            help=f"Default few shot count: {DEFAULT_FEW_SHOT_NUM}")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
            help=f"Default temperature: {DEFAULT_TEMPERATURE}")
    parser.add_argument("--save_snapshot", action='store_true',
            help="Save API query snapshot as .tar.gz file")
    args = parser.parse_args()

    Answers.run(args.model, args.snapshots_specs,
            args.use_chain_of_thought, args.few_shot_num, args.temperature,
            args.verbose, args.save_snapshot)
