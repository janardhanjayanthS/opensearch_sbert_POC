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
    return response.content


def check_similar_existing_category_else_return_new(
    new_category: str, existing_categories: list[str]
):
    response = openai.invoke(
        [
            {"role": "system", "content": "the system message"},
            {
                "role": "human",
                "content": f"new category: {new_category} and existing categories: {existing_categories}",
            },
        ]
    )
    return response.content


# if __name__ == "__main__":
#     get_category(text=input("enter text to get category: "))
