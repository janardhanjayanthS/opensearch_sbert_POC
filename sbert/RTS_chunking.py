from typing import Optional

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_file_contents(filepath: str) -> Optional[list[str]]:
    try:
        file = pymupdf.open(filename=filepath)
        contents = []
        for page in file:
            text = page.get_text()
            contents.append(text)
        return contents
    except Exception as e:
        print(f"ERROR: {e}")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=len,
)

if __name__ == "__main__":
    file_contents = get_file_contents(
        "..\\files\\pdf\\Extended Leave and Career Breaks Policy United Kingdom .pdf"
    )
    text = "".join(list(file_contents))
    split_text = splitter.split_text(text=text)
    for each_text in split_text:
        print(each_text, len(each_text))
        print("-" * 50)
    print(len(split_text))
