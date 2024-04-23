import sys
import os
from copy import deepcopy
from typing import Dict, List, Tuple, Callable, Any, Awaitable, Optional

import openai # type: ignore

from model_api.closed_api import ClosedAPI, RateLimited, KnownModel, PERIODS_PER_MIN, ParameterizedModel
from model_api.closed_api import OracleTaskQ
from unformatted_llm import UnformattedLLM
from persist import Persist

# In debug mode will only make limited API queries
DEBUG_MODE_NUM_LIMIT = None # 200 # or None for non-debug mode

# end few shot instances with special tag
END_TAG = "\n~~~\n"

# how much buffer on being nice with API calls; e.g., if 100/min allowed then we do 50/min if factor = 2
BATCH_SZ_FACTOR_PER_MIN = 8
# extra nice on API calls, by reducing batch size "permitted" based on API rate, divided by this factor
EXTRA_NICE = 4

# find these keys at https://platform.openai.com/account/api-keys
# ... org-... at url above Organization->Settings
# ... sk- at url above User->API Keys
API_ORG = os.getenv("OPENAI_ORG")
API_KEY = os.getenv("OPENAI_API_KEY")

def ensure_keys():
    if API_KEY and API_ORG:
        return
    print(f'OAI keys missing. export OPENAI_API_KEY and OPENAI_ORG. Aborting.')
    print(f'[WARN] You might have to purge the oracle task queue at {OracleTaskQ.TASKS_DIR}')
    print(f'[WARN] Some no-key tasks might have been written to queue')
    exit(-1)

class OAI_PARAMS:
    # gpt rate limits: https://platform.openai.com/account/rate-limits
    QUERY_PER_MIN_LIMIT = {
            "gpt4": 200,
            "gpt3": 3500,
    }
    if isinstance(DEBUG_MODE_NUM_LIMIT, int):
        QUERY_PER_MIN_LIMIT = {
                "gpt4": min(200, DEBUG_MODE_NUM_LIMIT),
                "gpt3": min(3500, DEBUG_MODE_NUM_LIMIT),
        }

    # https://openai.com/pricing#language-models
    MODEL_COST_PER_TOKEN = {
            "gpt4": 0.03 / 1000,
            "gpt3": 0.0015 / 1000,
    }

    # Note, legacy text models, e.g., DAVINCI are being deprecated Jan 2024
    # https://platform.openai.com/docs/deprecations/base-gpt-models
    @classmethod
    def cps(cls, m):
        return cls.MODEL_COST_PER_TOKEN[m]
    @classmethod
    def bsz(cls, m):
        return int(cls.QUERY_PER_MIN_LIMIT[m] / (PERIODS_PER_MIN * BATCH_SZ_FACTOR_PER_MIN))
    @classmethod
    def np(cls, m):
        return int(cls.bsz(m) / EXTRA_NICE) if DEBUG_MODE_NUM_LIMIT is None else 10
    @classmethod
    def rates(cls, m):
        return { "batch_size": cls.bsz(m), "num_parallel": cls.np(m), "cps": cls.cps(m) }
    LOC = { "api_org": API_ORG, "api_key": API_KEY, "api_base": "https://api.openai.com/v1" }

class OAI_MODELS:
    GPT4Turbo = KnownModel("gpt-4-turbo-2024-04-09", is_chat = True, **OAI_PARAMS.LOC, **OAI_PARAMS.rates("gpt4")) # type: ignore
    GPT4 = KnownModel("gpt-4", is_chat = True, **OAI_PARAMS.LOC, **OAI_PARAMS.rates("gpt4")) # type: ignore
    GPT3 = KnownModel("gpt-3.5-turbo", is_chat = True, **OAI_PARAMS.LOC, **OAI_PARAMS.rates("gpt3")) # type: ignore

    SHORT_NAMES = {"gpt4turbo": GPT4Turbo, "gpt4": GPT4, "gpt3": GPT3}

    DEFAULT_MODEL = GPT3

class GPT(ClosedAPI):
    def __init__(self, params: ParameterizedModel):
        self.params = params
        super().__init__(params)

        a, t, c, p = GPT.model_fns(params.model, params.temperature, params.few_shot, params.instruction, params.who_are_you)
        self.model_ask = a # type: ignore [method-assign]
        self.to_prompt_task = t # type: ignore [method-assign]
        self.deobject = c # type: ignore [method-assign]
        self.prelude = p

    def extract_answer(self, response) -> str:
        completion = response.lstrip()
        end = completion.find(END_TAG)
        answer = completion[:end] if end != -1 else completion
        answer = answer.strip()
        return answer

    def reset_library(self, model_params):
        # reset openai library
        openai.organization = model_params.api_org
        openai.api_key = model_params.api_key
        openai.api_base = model_params.api_base

    def spent(self, response) -> float:
        return response['usage']['total_tokens'] * self.params.model.cost_per_token

    @classmethod
    def model_fns(cls, model, temp, few_shot, instruction, who_are_you) -> Tuple[Any, Any, Any, Any]:
        # note `acreate` instead of `create` for async
        # docs: https://github.com/openai/openai-python#async-api

        # chat ask, raw api query
        def chat_ask(msgs):
            # print(f'Sending query: {msgs}')
            return openai.ChatCompletion.acreate(
                model = model.name,
                messages = msgs,
                temperature = temp,
            )
        # prelude added to each query
        who = {"role": "system", "content": who_are_you}
        chat_prelude = [who]
        for inp, out in few_shot.items():
            que = {"role": "user", "content": instruction + inp}
            chat_prelude.append(que)
            ans = {"role": "assistant", "content": out + END_TAG}
            chat_prelude.append(ans)
        # extract
        chat_txt_from_obj = lambda response: response['choices'][0]['message']['content']

        def chat_task(task: str):
            messages = deepcopy(chat_prelude)
            que = {"role": "user", "content": instruction + task}
            messages.append(que)
            return messages

        # pick appropriate functions based on model type
        callfn = chat_ask
        promptfn = chat_task
        deobject = chat_txt_from_obj
        prelude = chat_prelude
        return callfn, promptfn, deobject, prelude

    def check_model_access(self):
        # check that we have access to this model
        self.reset_library(self.params.model)
        models = openai.Model.list()["data"]
        assert list(filter(lambda x: x["id"] == self.params.model.name, models)), "No access to {self.params.model.name}"
