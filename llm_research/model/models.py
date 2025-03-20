from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from .base_model import BaseModel
from .utils import env_setup


class OpenAILLM(BaseModel):
    def __init__(self,
                 *,
                 model: str = 'gpt-3.5-turbo-1106',
                 seed: int = 1126,
                 temperature: float = 0.8,
                 max_completion_tokens: int|None = None,
                 timeout: int = 120,
                 verbose = False
                ) -> None:
        env_setup("OPENAI")
        llm = ChatOpenAI(
            model = model,
            seed = seed,
            temperature = temperature,
            max_completion_tokens = max_completion_tokens,
            # ref: https://github.com/langchain-ai/langchainjs/issues/4555#issuecomment-1975270399
            model_kwargs={"response_format": {"type": "json_object"}},
            timeout = timeout,
            verbose = verbose
        )
        super().__init__(llm, verbose)


class OllamaLLM(BaseModel):
    def __init__(self,
                 *,
                 model: str = 'phi4:14b-q4_K_M',
                 num_ctx: int = 16384,  # context length, can be check by the command: ollama show {model}
                 seed: int = 1126,  # random seed
                 temperature: float = 0.8,
                 num_predict: int = -1,  # number of tokens to generate, -1: unlimited, -2: full context
                 timeout: int = 120,
                 verbose = False
                ) -> None:
        env_setup()
        llm = ChatOllama(
            model = model,
            num_ctx = num_ctx,
            seed = seed,
            temperature = temperature,
            num_predict = num_predict,
            timeout = timeout,
            verbose = verbose,
            format = 'json',
        )
        super().__init__(llm, verbose)