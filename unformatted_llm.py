from typing import Dict, List, Tuple, Optional
from few_shot import FewShotAnswerSamples
from math_utils.math_helpers import rm_latex_math
from helper_utils import NO_SOLUTION_PREFIX

HDR = "###"
PRB_TAG = f"{HDR} Problem"
ANS_TAG = f"{HDR} Answer"
END_TAG = "\n\n"

class UnformattedLLM:
    INSTRUCTION = f"Given a mathematics problem, determine the answer. "\
                  f"Simplify your answer as much as possible. "\
                  f"If the answer cannot be computed, or you are not confident, say {NO_SOLUTION_PREFIX}"

    def few_shot_dict(self, count: int = -1) -> Dict[str, str]:
        ios: Dict[str, str] = self.few_shot_limited(count)
        return ios

    def few_shot_format(self, ios: Dict[str, str]) -> str:
        return "".join([self.format(i, o) for i, o in ios.items()])

    def prompt_for(self, prb: str, solns: List[str] = []) -> str:
        assert solns == [], f'Unformatted does not format soln steps to model prompt'
        return self.format(prb)

    def extract_answer(self, completion: str) -> str:
        completion = completion.lstrip()
        end = completion.find(END_TAG)
        answer = completion[:end] if end != -1 else completion
        answer = answer.strip()
        return answer

    def __init__(self):
        mk_io = lambda fs: (fs.prb, rm_latex_math(fs.outcome))
        self.few_shot_io: List[Tuple[str, str]] = [ mk_io(fs) for fs in FewShotAnswerSamples ]

    def few_shot_limited(self, count: int) -> Dict[str, str]:
        limited_samples = self.few_shot_io[:count] if count != -1 else self.few_shot_io
        return dict(limited_samples)

    def format_fn(self, prb: str, solns: List[str], outcome: str, explain: Optional[List[str]]):
        assert solns == [] and explain == None
        return self.format(prb, outcome)

    def format(self, inp: str, out: Optional[str] = None) -> str:
        lines = [
                UnformattedLLM.INSTRUCTION,
                PRB_TAG,
                inp,
                ANS_TAG,
        ]
        if out:
            lines += [out, END_TAG]
        return "\n".join(lines)


