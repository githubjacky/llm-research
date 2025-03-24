from functools import cached_property
import mlflow
from loguru import logger
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
    RunnableSerializable
)
from langchain.globals import set_verbose
from langchain.memory.buffer import ConversationBufferMemory
import numpy as np
import orjson
from operator import itemgetter
from pathlib import Path
import sys
from shutil import rmtree
from tqdm import trange
import time
from typing import Tuple, List, Dict, Optional

from .prompt import Prompt
from .utils import read_jsonl


class BaseModel:
    def __init__(self, llm, verbose = False) -> None:
        set_verbose(verbose)
        self.llm = llm

        # these attributes will be initialized later in the self.request_batch method
        self.prompt = None
        self.request_list = None  # stored the jsonl instances for sending requests to LLMs
        self.is_fewshot_prompt = False


    @cached_property
    def max_context_length(self) -> int:
        if hasattr(self.llm, 'num_ctx'):
            logger.info(f"Using Ollama model, {self.llm.model} with context length of {self.llm.num_ctx} tokens")
            context_length = self.llm.num_ctx
        
        # For OpenAI and other models - use hardcoded context lengths based on model name
        else:
            # Common context lengths for various models
            context_lengths = {
                # OpenAI models
                'gpt-4o': 128000,
                # Claude models
                # ref: https://docs.anthropic.com/en/docs/about-claude/models/all-models
                'claude-3-7-sonnet-latest': 200000,
                # Gemini models
                # ref: https://ai.google.dev/gemini-api/docs/models
                'gemini-2.0-flash': 1048576,
            }

            if hasattr(self.llm, 'model_name'):  # OpenAI
                model_name = self.llm.model_name
            else:
                if 'models' in self.llm.model:  # Gemini
                    model_name = self.llm.model.split('models/')[1]
                else:
                    model_name = self.llm.model
            
            # Try to find a matching model
            if model_name in context_lengths:
                logger.info(f"Using model {model_name} with context length of {context_lengths[model_name]} tokens")
                context_length = context_lengths[model_name]
            else:
                logger.warning(f"Unknown model: {model_name}. Using default context length of 16k tokens.")
                context_length = 16000

        return context_length - 200  # 200 is for the memory message tokens


    def init_request(self, experiment_name: str, run_name: str) -> None:
        self.experiment_name = experiment_name
        self.__experiment_id = (
            mlflow
            .set_experiment(experiment_name)
            .experiment_id
        )
        self.run_name = run_name

        log_dir = Path(f'log/{self.experiment_name}')
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = log_dir / f'{self.run_name}.log'
        if self.log_file_path.exists():
            self.log_file_path.unlink()
        logger.remove()
        logger.add(self.log_file_path, level = "INFO")

        llm_response_dir = Path(f'data/request_results/{self.experiment_name}')
        llm_response_dir.mkdir(parents=True, exist_ok=True)
        self.llm_response_file_path = llm_response_dir / f'{self.run_name}.jsonl'


    def _get_prompt_token_count(self, chain, instructions):
        chat_prompt_template = chain.steps[1]
        prompt_message = chat_prompt_template.format(instructions=instructions, history=[])
        return self.llm.get_num_tokens(prompt_message)


    def __trim_fewshot_prompt(self, chain, instructions, i):
        token_count = self._get_prompt_token_count(chain, instructions)
        n_examples = self.prompt.max_n_fewshot_examples
        news_content = self.request_list[i]
    
        while token_count > self.max_context_length and n_examples > 0:
            logger.info(f"{i+1} th sample has {token_count} prompt tokens with {n_examples} fewshot examples")

            n_examples -= 1
            message_prompt = self.prompt.few_shot(None, n_examples)
            chain = (
                message_prompt.partial(**{self.prompt.request_variable: news_content})
                | self.llm
                | self.prompt.parser
            )
            chain = self.__add_memory(chain)

            token_count = self._get_prompt_token_count(chain, instructions)

        logger.info(f"{i+1} th sample has {token_count} prompt tokens with {n_examples} fewshot examples")

        # reduce the size of the news article when even the number of fewshot examples is 0,  the
        # prompt size still exceed the context length
        if n_examples == 0 and token_count > self.max_context_length:
            logger.info("The number of fewshot examples is 0, but the prompt size still exceeds the context length. Therefore, the news content is reduced.")
            n_exceed_tokens = token_count - self.max_context_length
            seq_tokens = self.llm.get_token_ids(news_content)
            n_retain_tokens = len(seq_tokens) - n_exceed_tokens

            end = -1
            new_news_content = " ".join(news_content.split(" ")[:end])
            while len(new_news_content) > 0 and self.llm.get_num_tokens(new_news_content) > n_retain_tokens:
                end -= 1
                new_news_content = " ".join(news_content.split(" ")[:end])

            message_prompt = self.prompt.zero_shot()
            chain = (
                message_prompt.partial(**{self.prompt.request_variable: new_news_content})
                | self.llm
                | self.prompt.parser
            )
            chain = self.__add_memory(chain)
    
        return chain


    def __format_handler(self,
                         chain: RunnableSequence | RunnableSerializable,
                         instructions: str = "",
                         i: int = 1, # the ith request
                        ):
        retry = True
        n_retries = 0
        while retry:
            if self.is_fewshot_prompt:
                chain = self.__trim_fewshot_prompt(chain, instructions, i)

            try:
                res = chain.invoke({'instructions': instructions})

                if set(res.keys()) == set(self.prompt.response_variables):
                    retry = False
                else:
                    logger.info(f"formatting error(KeyError) for {i+1} th sample, re-generate")
                    instructions = " ".join((
                        f"This is the {n_retries} th retry.",
                        "Your answer which is a json string has missing keys or extra keys.",
                        "Follow the schema carefully.",
                    ))

            except orjson.JSONDecodeError:
                logger.info(f"formatting error(JSONDecodeError) for {i+1} th sample, re-generate")
                instructions = " ".join((
                    f"This is the {n_retries} th retry.",
                    "Formatting error. It might because",
                    "not all single quotes have been escaped or",
                    "the answering has been truncated.ry to answer precisely",
                    "and reduce the number of token.",
                ))
            # AttributeError: 'NoneType' object has no attribute 'get'
            except AttributeError:
                continue
            except:
                logger.info(f"Please reduce the length of the prompt for the {i+1}th sample")
                sys.exit()

            n_retries += 1  # count the number of retries

        return res


    @staticmethod
    def __add_memory(chain: RunnableSequence):
        memory = ConversationBufferMemory(return_messages=True)
        return (
            RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
            )
            | chain
        )


    def request_instance(self,
                         instructions: str,
                         chain: RunnableSequence,
                         i: int
                        ):
        return self.__format_handler(
            self.__add_memory(chain),
            instructions,
            i,
        )


    @property
    def __run_id(self):
        run_info = mlflow.search_runs(self.__experiment_id, filter_string=f"run_name='{self.run_name}'")
        return run_info['run_id'].values[0]


    def __hook_process(self, request_file_path: str) -> Tuple[int, int, List]:
        data = read_jsonl(request_file_path)

        if not self.llm_response_file_path.exists():
            _i = 0
            n_request = len(data)
        else:
            _i = len(read_jsonl(self.llm_response_file_path))
            if _i == len(data):
                rmtree(f'mlruns/{self.__experiment_id}/{self.__run_id}')
                self.log_file_path.unlink(missing_ok=True)
                self.llm_response_file_path.unlink(missing_ok=True)

                _i = 0
                n_request = len(data)
            else:
                n_request = len(data) - _i
                logger.info(f"restart the process form the {_i+1}th request")

        mlflow.start_run(experiment_id=self.__experiment_id, run_name=self.run_name)
        return _i, n_request, data


    def mlflow_logging(self, data: List[Dict]):
        mlflow.log_artifact(self.prompt.system_prompt_path)
        mlflow.log_artifact(self.prompt.human_prompt_path)

        res = read_jsonl(self.llm_response_file_path)
        keys = list(data[0].keys()) + list(res[0].keys())
        values = (
            np.array([list(i.values()) for i in data]).transpose().tolist()
            +
            np.array([list(i.values()) for i in res]).transpose().tolist()
        )
        table_dict = {
            key: value
            for key, value in zip(keys, values)
        }
        mlflow.log_table(table_dict, "request_response.json")


    def request_batch(self,
                      prompt: Prompt,
                      request_file_path: str,
                      fewshot_examples_path: Optional[str] = None,
                      sleep: int = 3,
                     ) -> None:
        self.prompt = prompt
        _i, n_request, data = self.__hook_process(request_file_path)
        logger.info(f"has fininshed {_i} requests and there are {n_request} requests left")
        self.request_list = [i.get(prompt.request_variable) for i in data]

        if fewshot_examples_path is not None:
            message_prompt = prompt.few_shot(fewshot_examples_path)
            mlflow.log_artifact(fewshot_examples_path)

            self.is_fewshot_prompt = True

        else:
            message_prompt = prompt.zero_shot()

        logger.info('start the request process')
        with self.llm_response_file_path.open('ab') as f:
            for i in trange(n_request, position=0, leave=True):
                chain = (
                     message_prompt.partial(**{prompt.request_variable: self.request_list[_i+i]})
                    | self.llm
                    | prompt.parser
                )
                res = self.request_instance("", chain, _i+i)
                f.write(orjson.dumps(res, option=orjson.OPT_APPEND_NEWLINE))
                f.flush()
                time.sleep(sleep)

        text_list = self.llm_response_file_path.read_text().split("\n")[:-1]  # remove the last empty line
        self.llm_response_file_path.unlink()

        with self.llm_response_file_path.open('w') as f:
            for text in text_list[:-1]:
                f.write(text)
                f.write("\n")
            f.write(text_list[-1])

        logger.info('finish the request process')
        self.mlflow_logging(data)


    @staticmethod
    def end_request():
        mlflow.end_run()