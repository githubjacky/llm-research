from langchain_core.prompts import PromptTemplate
from pathlib import Path
import orjson


from .model.utils import read_jsonl
from .model.prompt import Prompt


def finetune_format(prompt: Prompt, fpath_train: str):
    train_instances = read_jsonl(fpath_train)

    dir_output = Path(f'data/finetune/')
    dir_output.mkdir(parents=True, exist_ok=True)
    fpath_request_name = fpath_train.split('/')[-1].split('.')[0]
    f_output = dir_output / f'train_{fpath_request_name}.jsonl'

    system_message = prompt.system_message
    human_message = (
        PromptTemplate
        .from_template(prompt.human_prompt_template)
        .partial(
            instructions = "",
            output_instructions = prompt.output_instructions,
        )
    )

    with f_output.open('wb') as f:
        for i in train_instances:

            r_human = (
                human_message
                .format(
                    **{
                        prompt.request_variable: i.get(prompt.request_variable)
                    }
                )
            )

            r_assistant = (
                orjson
                .dumps({j: i[j] for j in prompt.response_variables})
                .decode()
            )

            item = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": r_human
                    },
                    {
                        "role": "assistant",
                        "content": r_assistant
                    },
                ]
            }
            f.write(orjson.dumps(item, option = orjson.OPT_APPEND_NEWLINE))
