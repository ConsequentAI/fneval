from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
from model_api.closed_api import ClosedAPI, RateLimited, KnownModel, PERIODS_PER_MIN, ParameterizedModel
from unformatted_llm import UnformattedLLM
from typing import Any, Awaitable, Dict, Optional, List, Tuple
from tqdm import tqdm # type: ignore

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DEFAULT_END_TAGS = ['\n\n', '<question>']
DEFAULT_PROMPT_FORMAT = "<question>: {prompt}\n<answer>: {response}"

class MISTRAL_PARAMS:
    def __init__(self):
        # https://docs.mistral.ai/platform/endpoints/
        models_names: List[str] = ["mistral-tiny", "mistral-small", "mistral-medium"]
        self.models: Dict[str, KnownModel] = { n: self.known_from_config(n) for n in models_names }

    def known_from_config(self, name: str) -> KnownModel:
        is_chat = True
        stops: List[str] = DEFAULT_END_TAGS
        prompt_format: str = DEFAULT_PROMPT_FORMAT

        return KnownModel(name,
                is_chat,
                api_org = None,
                api_key = MISTRAL_API_KEY,
                api_base = "",
                stops = stops,
                prompt_format = prompt_format)


class Mistral(ClosedAPI):
    def __init__(self, params: ParameterizedModel):
        super().__init__(params)

        self.params = params
        self.prelude = self.few_shot_format(params.few_shot)
        self.client = MistralClient(api_key = params.model.api_key)

    def deobject(self, obj_resp: Any) -> str:
        extract = lambda resp: resp.choices[0].message.content
        completion = obj_resp if isinstance(obj_resp, str) else extract(obj_resp)
        assert isinstance(completion, str), f'model output is not str: {obj_resp}'
        return completion

    def extract_answer(self, response) -> str:
        completion = response.lstrip()
        assert self.params.model.stops, f'stop tokens not set'
        ends = [completion.find(s) for s in self.params.model.stops if completion.find(s) != -1]
        answer = completion[:ends[0]] if ends else completion
        answer = answer.strip()
        return answer

    async def async_ask(self, prompt: str):
        return self.client.chat(
            model = self.params.model.name,
            messages = [
                ChatMessage(role="user", content=prompt)
            ],
            max_tokens = 20,
        )

    def to_prompt_task(self, task: str) -> Any:
        return self.prelude + self.params.instruction + self.format(task, None)

    def reset_library(self, model: KnownModel) -> None:
        return

    def model_ask(self, prompt: str) -> Awaitable[Any]:
        return self.async_ask(prompt)

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
            assert self.params.model.stops, f'stop tokens not set'
            fs += self.format(inp, out) + self.params.model.stops[0]
        return fs
