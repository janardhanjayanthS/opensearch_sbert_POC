from os import getenv

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.embeddings.create(
    input="hello world", model="text-embedding-3-large"
)
embedding = response.data[0].embedding
print(embedding)
print(len(embedding))
