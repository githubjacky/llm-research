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
        self.response_variables = list(pydantic_object.__fields__.keys())
        self.parser = JsonOutputParser(pydantic_object=pydantic_object)
        self.output_instructions = self.parser.get_format_instructions()

        self.system_prompt_path = system_prompt_path
        self.system_prompt_template = load_prompt(system_prompt_path).template
        self.system_message_input_variables = system_message_input_variables

        self.human_prompt_path = human_prompt_path
        human_prompt = load_prompt(human_prompt_path)
        self.human_prompt_template = human_prompt.template

        request_variables = human_prompt.input_variables
        if not 'instructions' in request_variables:
            logger.debug('"instuctions" is not the input variable of human prompt')
            sys.exit()
        elif not 'output_instructions' in request_variables:
            logger.debug('"output_instructions" is not the input variable of human prompt')
            sys.exit()
        else:
            request_variables.remove('instructions')
            request_variables.remove('output_instructions')
            self.request_variable = request_variables[0]


    @property
    def system_message(self):
        return (
            self
            .system_prompt_template
            .format(**self.system_message_input_variables)
        )


    def zero_shot(self) -> ChatPromptTemplate:
        return (
            ChatPromptTemplate
            .from_messages([
                ("system", self.system_prompt_template),
                MessagesPlaceholder(variable_name = "history"),
                ("human", self.human_prompt_template)
            ])
            .partial(
                **self.system_message_input_variables,
                output_instructions = self.output_instructions,
            )
        )


    def __create_fewshot_prompt(self,
                                request_examples: List[str],
                                response_examples: List[str],
                               ) -> ChatPromptTemplate:
        few_shot_example = [
            {
                self.request_variable: request_examples[i],
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
                **self.system_message_input_variables,
                output_instructions = self.output_instructions
            )
        )


    def few_shot(self, fewshot_examples_path: str) -> ChatPromptTemplate:
        """Create fewshot prompt.
        Args:
            `fewshot_examples_path`: It must be json lines file. 

        Example:
            .. code-block:: python

                fewshot_examples = [
                    {"request_varialbe": "", "label": "", "reason": ""},
                    {"request_variable": "", "label": "", "reason": ""},
                ]
                with Path('fewshot_examples.jsonl').open('w') as f:
                    for i in fewshot_examples:
                        f.write(orjson.dumps(i, option=orjson.OPT_APPEND_NEWLINE))

                Prompt.few_shot('fewshot_examples.jsonl')
        """
        fewshot_examples = read_jsonl(fewshot_examples_path)
        request_examples = []
        response_examples = []

        for item in fewshot_examples:
            request_examples.append(item[self.request_variable])
            response_examples.append(
                orjson.dumps(
                    {k: item[k] for k in self.response_variables}
                )
                .decode()
            )

        return self.__create_fewshot_prompt(request_examples, response_examples)
