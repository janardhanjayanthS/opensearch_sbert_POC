from os import getenv

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.categorizer.prompt import CATEGORIZE_SYSTEM_PROMPT, COMPARE_CATEGORIES_SYSTEM_PROMPT

# from prompt import CATEGORIZE_SYSTEM_PROMPT, COMPARE_CATEGORIES_SYSTEM_PROMPT # while testing

load_dotenv()

openai = ChatOpenAI(model="gpt-4o-mini", api_key=getenv("OPENAI_API_KEY"))


def get_category(text: str) -> str:
    """Use LLM to assign a generic category label to the given text.

    Args:
        text: Raw text to categorize.

    Returns:
        Single category name string, or "uncategorizable".
    """
    response = openai.invoke(
        [
            {"role": "system", "content": CATEGORIZE_SYSTEM_PROMPT},
            {"role": "human", "content": text},
        ]
    )
    return response.content


def check_similar_existing_category_else_return_new(
    new_category: str, existing_categories: list[str]
) -> str:
    """Compare a new category against existing ones and return the best match or the new category.

    Uses LLM to decide if the new category is semantically similar to any existing one.
    If similar, returns the matching existing category name. Otherwise returns new_category.

    Args:
        new_category: Category label just produced by get_category().
        existing_categories: Category names already stored in category_index.

    Returns:
        Matched existing category name, or new_category if no close match found.
    """
    response = openai.invoke(
        [
            {"role": "system", "content": COMPARE_CATEGORIES_SYSTEM_PROMPT},
            {
                "role": "human",
                "content": f"new category: {new_category} and existing categories: {existing_categories}",
            },
        ]
    )
    return response.content


# if __name__ == "__main__":
#     get_category(text=input("enter text to get category: "))
