import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read original text
with open('extracted_text.txt', 'r', ) as file:
    text = file.read()

# Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Create chunks
texts = text_splitter.split_text(text)

# Save all chunks in ONE file
with open("chunks.pkl", "wb") as f:
    pickle.dump(texts, f)

print(f"Saved {len(texts)} chunks to chunks.pkl")

for index, value in enumerate(texts):
    print(f"\n[Chunk {index}]:\n{value[:100]}")
