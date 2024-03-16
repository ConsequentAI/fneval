import os
from model_api.closed_api import ClosedAPI, RateLimited, KnownModel, PERIODS_PER_MIN, ParameterizedModel, OFFLINE_DEFER
from typing import Any, Awaitable, List, Dict, Optional

END_TAG = "\n~~~\n"
MAX_TOKENS = 1024
DEFAULT_PROMPT_FORMAT = "Problem:\n{prompt}\nSolution:\n{response}"

mk_model = lambda name: KnownModel(
        name,
        is_chat = False,
        api_org = None,
        api_key = "",
        api_base = "",
        prompt_format = DEFAULT_PROMPT_FORMAT)

class OFFLINE_PARAMS:
    NAMES = {
                'gemini-ultra': 'gemini-ultra',
                'gemini-pro': 'gemini-pro',
                'gemini-nano': 'gemini-nano',
            }
    SHORT_NAMES = { short: mk_model(model) for short, model in NAMES.items() }

class FileAPI(ClosedAPI):
    def __init__(self, params: ParameterizedModel, mode_write_questions: bool):
        super().__init__(params)

        self.params = params
        self.prelude = self.few_shot_format(params.few_shot)

        self.mode_write_questions = mode_write_questions
        self.offline_prompts: List[Dict[str, Any]] = []
        self.offline_answers = self.read_offline(params.model.name) if not mode_write_questions else None

    def read_offline(self, model_name: str) -> Dict[str, str]:
        assert False, f'unimplemented'

    def deobject(self, message: Any) -> str:
        completion = message
        assert isinstance(completion, str), f'model output is not str: {completion}'
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
        if self.mode_write_questions:
            # question mode
            return self.async_write_prompt(prompt,
                                          model = self.params.model.name,
                                          max_tokens = MAX_TOKENS,
                                          temperature = self.params.temperature)
        else:
            # answer mode
            return self.async_read_response(prompt)

    async def async_write_prompt(self, prompt,
                                 model: str,
                                 max_tokens: int,
                                 temperature: float) -> str:
        prompt_json = {
                'prompt': prompt,
                'model': model,
                'max_tokens': max_tokens,
                'temperature': temperature
        }
        self.offline_prompts.append(prompt_json)
        return OFFLINE_DEFER

    async def async_read_response(self, prompt) -> str:
        assert self.offline_answers, f'No offline answers available'
        assert prompt in self.offline_answers, f'Prompt not completed: {prompt}'
        message = self.offline_answers[prompt]
        return message

    def spent(self, response) -> float:
        return 0.0

    def format(self, inp: str, out: Optional[str]) -> str:
        resp = out if out else ""
        assert self.params.model.prompt_format, f'prompt format not set'
        formatted = self.params.model.prompt_format.format(prompt = inp, response = resp)
        return formatted

    def few_shot_format(self, ios: Dict[str, str]) -> str:
        fs = ""
        for inp, out in ios.items():
            fs += self.format(inp, out) + END_TAG
        return fs

