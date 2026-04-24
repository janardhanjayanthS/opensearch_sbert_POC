import fitz  # PyMuPDF
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download the NLTK sentence tokenizer data (only needs to run once)
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")


def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF file."""
    text = ""
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            print("Page text: ", page.get_text("text"))
            text += page.get_text("text") + " "
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")

    # Clean up excessive whitespace and newlines
    return " ".join(text.split())


def semantic_chunking(text, model_name="all-MiniLM-L12-v2", similarity_threshold=0.3):
    """
    Splits text into semantic chunks using SBERT.
    A new chunk is created when the similarity between adjacent sentences
    drops below the similarity_threshold.
    """
    # 1. Split text into individual sentences
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    print(f"Loaded SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    # 2. Generate embeddings for all sentences
    print("Generating embeddings...")
    embeddings = model.encode(sentences)

    # 3. Group sentences into semantic chunks
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # Calculate cosine similarity between the previous sentence and the current one
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]

        # If the similarity is above the threshold, they likely belong to the same topic
        if sim >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            # Topic shift detected. Save the current chunk and start a new one.
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    # Don't forget to add the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


if __name__ == "__main__":
    # --- Configuration ---
    # Replace with the path to your actual PDF file
    PDF_FILE_PATH = "..\\files\\pdf\\English\\Canada Vacation-Flexible Time Off (FTO) Policy 2025.pdf"

    # Threshold determines how strictly chunks are grouped.
    # Lower = larger chunks (more forgiving). Higher = smaller, highly specific chunks.
    SIMILARITY_THRESHOLD = 0.35

    # --- Execution ---
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(PDF_FILE_PATH)

    if pdf_text:
        print("Chunking text...")
        semantic_chunks = semantic_chunking(
            pdf_text, similarity_threshold=SIMILARITY_THRESHOLD
        )

        print("\n--- Results ---")
        print(f"Total chunks created: {len(semantic_chunks)}\n")

        # Print the first few chunks to verify
        for idx, chunk in enumerate(semantic_chunks):
            print(f"--- Chunk {idx + 1} ---")
            print(chunk)
            print("-" * 40)
    else:
        print("No text extracted. Please check the PDF path and file format.")
