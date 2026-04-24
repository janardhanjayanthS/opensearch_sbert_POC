from os import getenv

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.categorizer.prompt import SYSTEM_PROMPT

# from prompt import SYSTEM_PROMPT # while testing

load_dotenv()

openai = ChatOpenAI(model="gpt-4o-mini", api_key=getenv("OPENAI_API_KEY"))


def get_category(text: str) -> str:
    response = openai.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "human", "content": text},
        ]
    )
    print(response.content)


# if __name__ == "__main__":
#     get_category(text=input("enter text to get category: "))
