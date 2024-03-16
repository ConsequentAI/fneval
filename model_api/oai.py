from eval.runner import EvalRunner, mk_dir_safe
import argparse
import sys
from typing import Dict, List, Tuple, Any
from math_utils.math_helpers import rm_latex_math
from unformatted_llm import UnformattedLLM
from model_api.gpt import GPT, OAI_MODELS
from model_api.closed_api import KnownModel, ParameterizedModel, Runner
from evaluate import DEFAULT_SNAPSHOTS_FILE, DEFAULT_FEW_SHOT_NUM, DEFAULT_TEMPERATURE
from chain_of_thought import COT_INSTRUCTION, ChainOfThought

WHO_ARE_YOU = "You are a expert in mathematics, logic, and formal reasoning. "

class Answers(Runner):
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
            who_are_you = WHO_ARE_YOU,
            few_shot = few_shot,
            model = model,
            agent_name = f"OAI_Eval_{cache_model_name}_{cache_infer_params}",
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
        answer = self.gpt.extract_answer(completion)
        if self.use_cot:
            answer = self.cot.extract_answer(answer)
        return [(answer, completion)]

    def known_error(self, e) -> bool:
        KNOWN = [
                "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."
        ]
        print(f'API ERROR: {str(e)}')
        return str(e) in KNOWN

    @classmethod
    def run(cls, name: str, snapshots_specs: str,
            cot: bool, few_shot_num: int, temperature: float,
            verbose: bool = False, save_snaphot: bool = False, extra_params: Dict[str, Any] = {}):
        known_model = OAI_MODELS.SHORT_NAMES[name]
        answerer = Answers(known_model, cot, few_shot_num, temperature)
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
