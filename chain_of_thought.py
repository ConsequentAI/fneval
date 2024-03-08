# Ref: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models https://arxiv.org/abs/2201.11903

from typing import Dict, List, Tuple
from math_utils.math_helpers import rm_latex_math
from few_shot import CoTFewShotAnswerSamples, FewShotBuilder
from helper_utils import NO_SOLUTION_PREFIX

PREAMBLE = "Answer the MATH query below. " \
           f"If the answer cannot be computed, or you are not confident, say {NO_SOLUTION_PREFIX}. "
COT_PROMPT = "Think step by step. "
ANSWER_TAG = "The answer is: "
END_WITH_ANSWER = f"End the answer with \"{ANSWER_TAG} \" followed by only the answer you compute and no extra words.\n"

COT_INSTRUCTION = PREAMBLE + COT_PROMPT + END_WITH_ANSWER

class ChainOfThought(FewShotBuilder):
    def __init__(self):
        mk_io = lambda fs: (fs.prb, fs.sol + '\n' + ANSWER_TAG + rm_latex_math(fs.outcome))
        self.few_shot_io: List[Tuple[str, str]] = [ mk_io(fs) for fs in CoTFewShotAnswerSamples ]

    def few_shot_limited(self, count: int) -> Dict[str, str]:
        limited_samples = self.few_shot_io[:count] if count != -1 else self.few_shot_io
        return dict(limited_samples)

    def extract_answer(self, answer):
        # locate the first ANSWER_TAG
        ans_index = answer.find(ANSWER_TAG)
        extracted = answer[ans_index + len(ANSWER_TAG):] if ans_index != -1 else ''

        # expecting only a single line in the answer
        empty_line = extracted.find('\n')
        extracted = extracted[:empty_line] if empty_line != -1 else extracted

        # remove any whitespaces, and any ending periods
        extracted = extracted.strip()
        extracted = extracted[:-1] if extracted.endswith('.') else extracted

        # Most of time this processing works:
        # The answer is: $68/125$.
        # The answer is: $\frac{\pi}{4}$
        # The answer is: 165.
        # The answer is: 6

        # But then there are other were Opus insists on outputting english.
        # The answer is: $n=5$.
        # The answer is: The probability that Phil and Sarah get the same number is $\frac{3}{10}$ or 0.3 or 30%.
        # The answer is: There are 28 ways for Mary to put the 6 identical basil plants on the 3 window sills
        # The answer is: 3/8 or approximately 0.375.
        # The answer is: 18/343
        # The answer is: The probability is approximately 0.2616 or about 26.16%.
        # The answer is: 1 slice has both pepperoni and mushrooms.
        # The answer is: 110 students take physics.
        #
        # Anthropic team: Is there a way to force it to not do that?
        #                 Or is there a more sophisticated math equivalence check you are using?

        console_confirm = False
        if console_confirm:
            print(f'\n---------------\n')
            print(f'From:{answer}')
            print(f'\n---------------\n')
            print(f'Extracted: {extracted}')
            print(f'\n---------------\n')
            input('Continue? ')
        if ' ' in extracted:
            print(f'Extracted: {extracted}')
        if not extracted:
            print(f'Missing extraction; no answer tag; likely exceeded token length.')
        return extracted
