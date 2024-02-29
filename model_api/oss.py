import together # type: ignore
import os
from model_api.closed_api import ClosedAPI, RateLimited, KnownModel, PERIODS_PER_MIN, ParameterizedModel
from unformatted_llm import UnformattedLLM
from typing import Any, Awaitable, Dict, Optional, List, Tuple
from tqdm import tqdm # type: ignore

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
DEFAULT_END_TAGS = ['\n\n', '<human>']
DEFAULT_PROMPT_FORMAT = "<human>: {prompt}\n<model>: {response}"

# Model list: https://docs.together.ai/docs/inference-models
WORTH_IT_LANG = [
        "togethercomputer/falcon-40b",
        "togethercomputer/llama-2-70b",
        "EleutherAI/llemma_7b",
        "mistralai/Mixtral-8x7B-v0.1",
        "togethercomputer/Qwen-7B",
        "togethercomputer/StripedHyena-Hessian-7B",
        "WizardLM/WizardLM-70B-V1.0",
        "zero-one-ai/Yi-34B",
]

WORTH_IT_CODE = [
        "togethercomputer/CodeLlama-34b-Python",
        "togethercomputer/CodeLlama-34b",
        "WizardLM/WizardCoder-Python-34B-V1.0",
]

WORTH_IT_CHAT = [
        "zero-one-ai/Yi-34B-Chat",
        "togethercomputer/CodeLlama-34b-Instruct",
        "togethercomputer/llama-2-70b-chat",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Open-Orca/Mistral-7B-OpenOrca",
        "togethercomputer/Pythia-Chat-Base-7B-v0.16",
        "togethercomputer/Qwen-7B-Chat",
        "togethercomputer/StripedHyena-Nous-7B",
        "lmsys/vicuna-13b-v1.5",
        "togethercomputer/CodeLlama-34b-Instruct",
        "upstage/SOLAR-0-70b-16bit",
]

def worth_it_exclude(names: List[str]) -> Tuple[List[str], List[str]]:
    keep, exclude = [], []
    for n in names:
        if n in WORTH_IT_CHAT or n in WORTH_IT_LANG or n in WORTH_IT_CODE:
            keep.append(n)
        else:
            exclude.append(n)
    return keep, exclude

class TOGETHER_PARAMS:
    def __init__(self):
        models_names: List[str] = [m['name'] for m in together.Models.list()]
        print(f'## Available models: {len(models_names)}\n')

        # keep the "excluded" print, it'll show whats available on together as new models get added
        worth_it, excluded = worth_it_exclude(models_names)
        print(f'## Worth trying: {worth_it}\n')
        print(f'## Excluded: {excluded}\n')

        self.models: Dict[str, KnownModel] = { n: self.known_from_config(n) for n in tqdm(worth_it, desc = 'Getting info for "worth it" models') }

    def known_from_config(self, name: str) -> KnownModel:
        info = together.Models.info(name)

        # https://docs.together.ai/docs
        # >>> together.Models.info(model='mistralai/Mixtral-8x7B-v0.1')
        # info: {'modelInstanceConfig': {'appearsIn': [], 'order': 0}, '_id': '6577bf1034e6c1e2bb5283d9', 'name': 'mistralai/Mixtral-8x7B-v0.1', 'display_name': 'Mixtral-8x7B', 'display_type': 'language', 'description': 'The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts.', 'license': 'apache-2.0', 'link': 'https://huggingface.co/mistralai/Mixtral-8x7B-v0.1', 'creator_organization': 'mistralai', 'pricing_tier': 'Featured', 'access': 'open', 'num_parameters': '56000000000', 'show_in_playground': True, 'isFeaturedModel': True, 'context_length': 32768, 'pricing': {'input': 150, 'output': 150, 'hourly': 0}, 'created_at': '2023-12-12T02:01:52.674Z', 'update_at': '2023-12-12T02:01:52.674Z', 'instances': [{'avzone': 'us-east-1a', 'cluster': 'happypiglet'}], 'renamed': 'mistralai/mixtral-8x7b-32kseqlen', 'hardware_label': '', 'descriptionLink': '', 'depth': {'num_asks': 1, 'num_bids': 0, 'num_running': 0, 'qps': 0, 'throughput_in': 0, 'throughput_out': 0, 'error_rate': 0, 'retry_rate': 0, 'stats': [{'avzone': 'us-east-1a', 'cluster': 'happypiglet', 'capacity': 0, 'qps': 0, 'throughput_in': 0, 'throughput_out': 0, 'error_rate': 0, 'retry_rate': 0}]}}

        is_chat = info["display_type"] == 'chat'

        stops: List[str] = DEFAULT_END_TAGS
        prompt_format: str = DEFAULT_PROMPT_FORMAT
        # override the stops and prompt_format if they are specified (which is the case for chat models)
        if is_chat:
            # >>> together.Models.info(model='mistralai/Mixtral-8x7B-Instruct-v0.1')["config"]
            # {'stop': ['</s>', '[INST]'], 'prompt_format': '[INST] {prompt} [/INST]', 'chat_template_name': 'llama'}
            stops = info["config"]["stop"]
            prompt_format = info["config"]["prompt_format"] + "{response}"

        return KnownModel(name,
                is_chat,
                api_org = None,
                api_key = TOGETHER_API_KEY,
                api_base = "https://api.together.xyz",
                stops = stops,
                prompt_format = prompt_format)


class OSS(ClosedAPI):
    def __init__(self, params: ParameterizedModel):
        super().__init__(params)

        self.params = params
        self.prelude = self.few_shot_format(params.few_shot)

    def deobject(self, obj_resp: Any) -> str:
        extract = lambda output: output['output']['choices'][0]['text']
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

    def to_prompt_task(self, task: str) -> Any:
        return self.prelude + self.params.instruction + self.format(task, None)

    def reset_library(self, model: KnownModel) -> None:
        return

    def model_ask(self, prompt: str) -> Awaitable[Any]:
        return self.async_ask(prompt)

    async def async_ask(self, prompt: str):
        return together.Complete.create(
            model = self.params.model.name,
            max_tokens = 20,
            prompt = prompt,
            stop = self.params.model.stops
        )

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

