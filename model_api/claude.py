import os
from model_api.closed_api import ClosedAPI, RateLimited, KnownModel, PERIODS_PER_MIN, ParameterizedModel
from typing import Any, Awaitable, Dict, Optional

from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT

CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
END_TAG = "\n~~~\n"
MAX_TOKENS = 1024

mk_model = lambda name: KnownModel(
        name,
        is_chat = False,
        api_org = None,
        api_key = CLAUDE_API_KEY,
        api_base = "")

class ANTHROPIC_PARAMS:
    NAMES = {
            'c2-2.1': 'claude-2.1',
            'c3-opus': 'claude-3-opus-20240229',
            'c3-sonnet': 'claude-3-sonnet-20240229'
            }
    SHORT_NAMES = { short: mk_model(model) for short, model in NAMES.items() }

class Claude(ClosedAPI):
    def __init__(self, params: ParameterizedModel):
        super().__init__(params)

        self.params = params
        self.prelude = self.few_shot_format(params.few_shot)
        self.anthropic = AsyncAnthropic(api_key = params.model.api_key)

    def deobject(self, message: Any) -> str:
        # message is a Message(
        #                   id = 'msg_..',
        #                   content = [ContentBlock(text='', type='text')],
        #                   model='claude-3-opus-20240229',
        #                   role='assistant',
        #                   stop_reason='end_turn',
        #                   stop_sequence=None,
        #                   type='message',
        #                   usage=Usage(input_tokens=84, output_tokens=409))
        content_list = message.content

        # From docs: https://docs.anthropic.com/claude/reference/messages_post
        # [{"type": "text", "text": "Hi, I'm Claude."}]

        # Actual output..
        # [ContentBlock(text='To solve this, we ca.......9647\n\nThe answer is: 0.9647', type='text')]
        assert len(content_list) == 1, f'Expecting a single content response, got: {content_list}'
        single_content = content_list[0]
        assert single_content.type == 'text', f'Only expecting text content, got: {single_content}'
        completion = single_content.text

        assert isinstance(completion, str), f'claude: model output is not str: {completion}'
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
        # print(f'Sending query: {prompt}')
        message = self.anthropic.messages.create(
                model = self.params.model.name,
                max_tokens = MAX_TOKENS,
                temperature = self.params.temperature,
                system = "You are a MATH expert.",
                messages = [{
                    'role': 'user',
                    'content': prompt,
                }]
        )
        return message

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

