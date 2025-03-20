from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from .base_model import BaseModel
from .utils import env_setup


class OpenAILLM(BaseModel):
    def __init__(self,
                 *,
                 model: str = 'gpt-3.5-turbo-1106',
                 temperature: float = 0.,
                 timeout: int = 120,
                 verbose = False
                ) -> None:
        super().__init__(verbose)
        env_setup("OPENAI")
        self.llm = ChatOpenAI(
            model = model,
            temperature = temperature,
            timeout = timeout,
            verbose = verbose
        )


class OllamaLLM(BaseModel):
    def __init__(self,
                 *,
                 model: str = 'phi4:14b-q4_K_M',
                 num_ctx: int = 16384,  # context length, default to 16k, which is the case for phi4 model
                 num_predict: int = -1,  # number of tokens to generate, -1: unlimited, -2: full context
                 seed: int = 1126,  # random seed
                 timeout: int = 120,
                 verbose = False
                ) -> None:
        super().__init__(verbose)
        env_setup()
        self.llm = ChatOllama(
            model = model,
            num_ctx = num_ctx,
            num_predict = num_predict,
            seed = seed,
            timeout = timeout,
            verbose = verbose
        )