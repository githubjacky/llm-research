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
import openai
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
    def __init__(self, verbose = False) -> None:
        set_verbose(verbose)


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
        logger.remove()
        logger.add(self.log_file_path, level = "INFO")

        llm_response_dir = Path(f'data/model_results/{self.experiment_name}')
        llm_response_dir.mkdir(parents=True, exist_ok=True)
        self.llm_response_file_path = llm_response_dir / f'{self.run_name}.jsonl'


    @staticmethod
    def __add_memory(chain: RunnableSequence):
        memory = ConversationBufferMemory(return_messages=True)
        return (
            RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
            )
            | chain
        )


    def __format_handler(self,
                         chain: RunnableSequence | RunnableSerializable,
                         instructions: str = "",
                         i: int = 1, # the ith request
                        ):
        retry = True
        while retry:
            try:
                res = chain.invoke({'instructions': instructions})

                bad = False
                for key in self.prompt.response_keys:
                    if res.get(key) is None:
                        bad = True
                        break
                if not bad:
                    retry = False
                else:
                    logger.info(f"formatting error(KeyError) for {i+1} th sample, re-generate")
                    instructions = " ".join((
                        "Your answer which is a json string don't",
                        "have the specified key. Follow the schema carefully.",
                    ))

            except orjson.JSONDecodeError:
                logger.info(f"formatting error(JSONDecodeError) for {i+1} th sample, re-generate")
                instructions = " ".join((
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

        return res


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


    def __hook_process(self, query_file_path: str) -> Tuple[int, int, List]:
        data = read_jsonl(query_file_path)

        if not self.llm_response_file_path.exists():
            _i = 0
            n_query = len(data)
        else:
            _i = len(read_jsonl(self.llm_response_file_path))
            if _i == len(data):
                rmtree(f'mlruns/{self.__experiment_id}/{self.__run_id}')
                self.log_file_path.unlink(missing_ok=True)
                self.llm_response_file_path.unlink(missing_ok=True)

                _i = 0
                n_query = len(data)
            else:
                n_query = len(data) - _i
                logger.info(f"restart the process form the {_i+1}th query")

        mlflow.start_run(experiment_id=self.__experiment_id, run_name=self.run_name)
        return _i, n_query, data


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
        mlflow.log_table(table_dict, "query_response.json")


    def request_batch(self,
                      prompt: Prompt,
                      query_file_path: str,
                      fewshot_examples_path: Optional[str] = None,
                      sleep: int = 3,
                     ) -> None:
        self.prompt = prompt
        _i, n_query, data = self.__hook_process(query_file_path)
        query_list = [i.get(prompt.query_key) for i in data]

        if fewshot_examples_path is not None:
            message_prompt = prompt.few_shot(fewshot_examples_path)
            mlflow.log_artifact(fewshot_examples_path)
        else:
            message_prompt = prompt.zero_shot()

        logger.info('start the query process')
        with self.llm_response_file_path.open('ab') as f:
            for i in trange(n_query, position=0, leave=True):
                chain = (
                     message_prompt.partial(**{prompt.query_key: query_list[_i+i]})
                    | self.llm
                    | prompt.parser
                )
                res = self.request_instance("", chain, _i+i)
                f.write(orjson.dumps(res , option=orjson.OPT_APPEND_NEWLINE))
                time.sleep(sleep)

        logger.info('finish the query process')
        self.mlflow_logging(data)


    @staticmethod
    def end_request():
        mlflow.end_run()
