from eval.runner import EvalRunner, mk_dir_safe
import argparse
import sys
import json
from typing import Dict, List, Tuple, Optional, Any
from math_utils.math_helpers import rm_latex_math
from unformatted_llm import UnformattedLLM
from model_api.closed_api import KnownModel, ParameterizedModel
from model_api.file_api import FileAPI, OFFLINE_PARAMS
from evaluate import DEFAULT_SNAPSHOTS_FILE, DEFAULT_FEW_SHOT_NUM, DEFAULT_TEMPERATURE, SPEC_EXTRA_PARAMS_MODE_WRITE_OFFLINE
from chain_of_thought import COT_INSTRUCTION, ChainOfThought

class Answers:
    def __init__(self, model: KnownModel, use_cot: bool, few_shot_num: int, temperature: float, mode_write_questions: bool):
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
                agent_name = f"Offline_Eval_{cache_model_name}_{cache_infer_params}",
        )
        self.file_api = FileAPI(params, mode_write_questions)

    def answers(self, prb: str) -> List[Tuple[str, str]]:
        completion = self.file_api.query([prb], rate_limited = False)[0]
        answer = self.file_api.extract_answer(completion)
        if self.use_cot:
            answer = self.cot.extract_answer(answer)
        return [(answer, completion)]

    def write_questions(self, e):
        def write(data):
            fname = self.file_api.params.agent_name + '.json'
            with open(fname, "w") as jfile:
                json.dump(data, jfile, indent=4)

        duplicates = 0
        for ref in e.problems_tested:
            matched = None
            for prompt in self.file_api.offline_prompts:
                if ref['problem'] in prompt['prompt']:
                    if matched and not ref['is_static']:
                        duplicates += 1
                    matched = prompt
            assert matched, f'Failed to match: {prompt}'
            for k, v in matched.items():
                ref[k] = v
        print(f'Duplicates found: {duplicates}. Investigate!')

        write(e.problems_tested)

    @classmethod
    def run(cls, name: str, snapshots_specs: str,
            cot: bool, few_shot_num: int, temperature: float,
            verbose: bool = False, save_snaphot: bool = False, extra_params: Dict[str, Any] = {}):
        mode_write_questions = extra_params[SPEC_EXTRA_PARAMS_MODE_WRITE_OFFLINE] if SPEC_EXTRA_PARAMS_MODE_WRITE_OFFLINE in extra_params else False
        known_model = OFFLINE_PARAMS.SHORT_NAMES[name]
        answerer = Answers(known_model, cot, few_shot_num, temperature, mode_write_questions)
        e = EvalRunner(answerer, snapshots_specs, verbose).do()
        if mode_write_questions:
            answerer.write_questions(e)
            return None
        if save_snaphot:
            answerer.file_api.snapshot_api_query_cache()
        return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
            help = f"Model has to be one of {list(OFFLINE_PARAMS.SHORT_NAMES.keys())}")
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
    parser.add_argument("--mode_write_questions", action='store_true',
            help="Write questions to output dump, to be solved offline")
    args = parser.parse_args()

    extra_params = { SPEC_EXTRA_PARAMS_MODE_WRITE_OFFLINE: args.mode_write_questions }
    Answers.run(args.model, args.snapshots_specs,
            args.use_chain_of_thought, args.few_shot_num, args.temperature,
            args.verbose, args.save_snapshot, extra_params)
