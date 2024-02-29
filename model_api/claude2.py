import os
from model_api.closed_api import ClosedAPI, RateLimited, KnownModel, PERIODS_PER_MIN, ParameterizedModel
from unformatted_llm import UnformattedLLM
from typing import Any, Awaitable, Dict, Optional

from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT

CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
END_TAG = "\n~~~\n"

mk_model = lambda name: KnownModel(
        name,
        is_chat = False,
        api_org = None,
        api_key = CLAUDE_API_KEY,
        api_base = "")

class ANTHROPIC_PARAMS:
    NAMES = [ "claude-2.1" ]
    SHORT_NAMES = { n: mk_model(n) for n in NAMES }

class Claude2(ClosedAPI):
    def __init__(self, params: ParameterizedModel):
        super().__init__(params)

        self.params = params
        self.prelude = self.few_shot_format(params.few_shot)
        self.anthropic = AsyncAnthropic(api_key = params.model.api_key)

    def deobject(self, obj_resp: Any) -> str:
        completion = obj_resp if isinstance(obj_resp, str) else obj_resp.completion
        assert isinstance(completion, str), f'claude2: model output is not str: {obj_resp}'
        return completion

    def extract_answer(self, response) -> str:
        completion = response.lstrip()
        end = completion.find(END_TAG)
        answer = completion[:end] if end != -1 else completion
        answer = answer.strip()
        return answer

    def to_prompt_task(self, task: str) -> Any:
        return self.prelude + self.params.instruction + self.format(task, None)

    def reset_library(self, model: KnownModel) -> None:
        return

    def model_ask(self, prompt: str) -> Awaitable[Any]:
        return self.anthropic.completions.create(
            model = self.params.model.name,
            max_tokens_to_sample = 20,
            prompt = prompt
        )

    def spent(self, response) -> float:
        return 0.0

    def format(self, inp: str, out: Optional[str]) -> str:
        fmt = f"{HUMAN_PROMPT} {inp}{AI_PROMPT}"
        extra = out if out else ""
        return fmt + extra

    def few_shot_format(self, ios: Dict[str, str]) -> str:
        fs = ""
        for inp, out in ios.items():
            fs += self.format(inp, out) + END_TAG
        return fs

