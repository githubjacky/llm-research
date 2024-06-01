# llm-research
*A minimum Python package built on top of the LangChain framework to interact with LLM.*

Some of the core features of this package includes: easy-to-use API, support fewshot prompting strategy, structual output(json) and MLflow integration to better inspect the output and one can also easily add the logging metrics/artifacts. For more infomation, check out the examples in the `examples` folder

# Prompt
1. prompt template

Users can play around with the prompt and save it as the json file through LangChain PromptTemplate API.
```python
from langchain.prompts import PromptTemplate

system_template = """\
You are an experienced expert in translating English addresses to Traditional Chinese.
Your task is to translate the English address to Traditional Chinese using Json format.
"Notice: Do not include the country and postal code in your response".\
"""
system_prompt_template = PromptTemplate.from_template(system_template)
system_prompt_template.save('system_prompt.json')
```

The human prompt's input variables must have `instructions` and `output_instructions` plus one variable as the rquest variable. In the following case, `owner_address` is the request variable
```python
human_template = """\
{instructions}
Translate the following address in Traditional Chinese:
{owner_address}
Output Instructions:
{output_instructions}
Besides, don't forget to escape a single quote in your response json string.
"""
```

2. output data class

Users can specify the responses' keys from LLM through LangChain pydantic module. In the following case, LLM will respone a json which contains `translated_address` key.
```python
from langchain_core.pydantic_v1 import BaseModel, Field

class LLMResponse(BaseModel):
    translated_address: str = Field(description="the translated address in Traditional Chinese")
```

3. create the prompt object

After Combining the system and human promt with pydantic data class, users are ready to create the prompt
```python
# notice that users should also specify the input variables in the system prompt as keyword arguments in this class
prompt = Prompt(LLMResponse, 'system_promp.json', 'human_prompt')
```


# Model
1. create the LLM class and specify the `experiment_name` and `run_name` for MLflow logging service
```python
from llm_research import OpenAILLM
model = OpenAILLM(model="gpt-3.5-turbo-1106", temperature=0., timeout=120, verbose=True)
model.init_request(experiment_name='10-test', run_name='chatgpt3.5')
```

2. start requesting

The input data should be formatted in json line files and each json instance should contain keys for request variable. For example, if users has `owner_address` as the request variable in the human prompt, there must be a key `owner_address` in each json instance. Notice that `instructions` and `output_instructions` shouldn't be excluded.
For fewshot examples, it should also be formatted as json lines and besides the request variable of human prompt should be one of the keys in each json instances, all fields of the pydantic data class should also be included. Based on the previous example, ther json must have two keys-`owner_address` and `translated_address`.
```python
model.request_batch(
    prompt,
    'data.jsonl',
    'fewshot_examples.jsonl'
)
```

3. end the requests
Dont' forget to end the request procedure to close the MLflow logging service.
```python
model.end_request()
```
