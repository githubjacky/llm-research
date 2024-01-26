from loguru import logger
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.prompts import load_prompt
import orjson
from pydantic._internal._model_construction import ModelMetaclass
from typing import List
import sys

from .utils import read_jsonl


class Prompt:
    def __init__(self,
                 pydantic_object: ModelMetaclass,
                 system_prompt_path: str,
                 human_prompt_path:str ,
                 **system_message_input_variables
                ):
        self.response_keys = list(pydantic_object.__fields__.keys())
        self.parser = JsonOutputParser(pydantic_object=pydantic_object)
        self.output_instructions = self.parser.get_format_instructions()

        self.system_prompt_path = system_prompt_path
        self.system_prompt_template = load_prompt(system_prompt_path).template
        self.system_message = system_message_input_variables

        self.human_prompt_path = human_prompt_path
        human_prompt = load_prompt(human_prompt_path)
        self.human_prompt_template = human_prompt.template

        query_variables = human_prompt.input_variables
        if not 'instructions' in query_variables:
            logger.debug('"instuctions" is not the input variable of human prompt')
            sys.exit()
        elif not 'output_instructions' in query_variables:
            logger.debug('"output_instructions" is not the input variable of human prompt')
            sys.exit()
        else:
            query_variables.remove('instructions')
            query_variables.remove('output_instructions')
            self.query_key = query_variables[0]


    def zero_shot(self) -> ChatPromptTemplate:
        return (
            ChatPromptTemplate
            .from_messages([
                ("system", self.system_prompt_template),
                MessagesPlaceholder(variable_name = "history"),
                ("human", self.human_prompt_template)
            ])
            .partial(
                **self.system_message,
                output_instructions = self.output_instructions,
            )
        )


    def __create_fewshot_prompt(self,
                                query_examples: List[str],
                                response_examples: List[str],
                               ) -> ChatPromptTemplate:
        few_shot_example = [
            {
                self.query_key: query_examples[i],
                "response": response_examples[i],
                "instructions": "",
                "output_instructions": "",
            }
            for i in range(len(response_examples))
        ]

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", self.human_prompt_template),
            ("ai", "{response}"),
        ])
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt = example_prompt,
            examples = few_shot_example,
        )

        return (
            ChatPromptTemplate
            .from_messages([
                ("system", self.system_prompt_template),
                few_shot_prompt,
                MessagesPlaceholder(variable_name = "history"),
                ("human", self.human_prompt_template),
            ])
            .partial(
                **self.system_message,
                output_instructions = self.output_instructions
            )
        )


    def few_shot(self, fewshot_examples_path: str) -> ChatPromptTemplate:
        """Create fewshot prompt.
        Args:
            `fewshot_examples_path`: It must be json lines file. 

            `query_key`: Each json instance must contains this key and for all
                the other keys are the keys of json format of LLM response.

        Example:
            .. code-block:: python

                fewshot_examples = [
                    {"query": "", "label": "", "reason": ""},
                    {"query": "", "label": "", "reason": ""},
                ]
                with Path('fewshot_examples.jsonl').open('w') as f:
                    for i in fewshot_examples:
                        f.write(orjson.dumps(i, option=orjson.OPT_APPEND_NEWLINE))

                Prompt.few_shot('fewshot_examples.jsonl')
        """
        fewshot_examples = read_jsonl(fewshot_examples_path)
        query_examples = []
        response_examples = []

        for item in fewshot_examples:
            query_examples.append(item[self.query_key])
            response_examples.append(
                orjson.dumps(
                    {k: item[k] for k in self.response_keys}
                )
                .decode()
            )

        return self.__create_fewshot_prompt(query_examples, response_examples)
