from dotenv import load_dotenv
import orjson
from os import getenv
from pathlib import Path
from typing import Dict, List



def env_setup(tag: str = "OPENAI", env_file: str = '.env') -> None:
    load_dotenv(env_file)
    match tag:
        case "GOOGLE":
            key_name = 'GOOGLE_APPLICATION_CREDENTIALS'
        case _:
            key_name = 'OPENAI_API_KEY'

    key = getenv(key_name)

    if key is None:
        key = input(f'input your {key_name}: ')

        file = Path(".env")
        if file.is_file():
            with file.open("a") as f:
                f.write(f"{key_name}={key}\n")
        else:
            with file.open("w") as f:
                f.write(f"{key_name}={key}\n")


def read_jsonl(path: str | Path,
               n: int = -1,
               return_str: bool = False) -> List[Dict] | List[str]:

    res = Path(path).read_text().split('\n')
    if n == len(res):
        n = -1

    if not return_str and n == -1 and res[-1] != '':
        return [orjson.loads(i) for i in res]
    if return_str and n == -1 and res[-1] != '':
        return res
    else:
         return (
            [orjson.loads(i) for i in res[:n]]
            if not return_str
            else
            res[:n]
        )


