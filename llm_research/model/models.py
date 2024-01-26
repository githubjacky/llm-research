from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI

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


class VertexAILLM(BaseModel):
    def __init__(self,
                 *,
                 model: str = "gemini-pro",
                 verbose = False
                 ):
        super().__init__(verbose)
        env_setup("GOOGLE")
        self.llm = ChatVertexAI(
            model_name=model,
            verbose=verbose,
            convert_system_message_to_human = True
        )
