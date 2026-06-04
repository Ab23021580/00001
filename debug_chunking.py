import re
from pypdf import PdfReader
from config import CHUNK_SIZE

reader = PdfReader('docs/6261123f68ff3747511986.pdf')
text = reader.pages[0].extract_text()

print(f"Total text length: {len(text)}")

# Step 1: Check for double newlines
blocks = text.split('\n\n')
print(f"Number of blocks after split by '\\n\\n': {len(blocks)}")

# Check each block
for i, block in enumerate(blocks):
    print(f"\nBlock {i} length: {len(block.strip())}")

# Step 2: Apply article splitting to the main block
main_block = blocks[0].strip()
article_pattern = r'\n?\s*第\s*[一二三四五六七八九十百千\d]+\s*條\s*[:：]?'

splits = re.split(f'({article_pattern})', main_block)
print(f"\nNumber of splits: {len(splits)}")
for i, s in enumerate(splits):
    print(f"Split {i} (len={len(s)}): {repr(s[:80])}...")

# Step 3: Reassemble into paragraphs
paragraphs = []
current_article = ""
for part in splits:
    if re.match(article_pattern, part):
        if current_article.strip():
            paragraphs.append(current_article.strip())
        current_article = part
    else:
        current_article += part
if current_article.strip():
    paragraphs.append(current_article.strip())

print(f"\nNumber of paragraphs after reassembly: {len(paragraphs)}")
for i, p in enumerate(paragraphs):
    print(f"Para {i} (len={len(p)}): {repr(p[:80])}...")

# Check if article 6 is in any paragraph
for i, p in enumerate(paragraphs):
    if '第六' in p or '待遇' in p or '時薪' in p:
        print(f"\n>>> Article 6 found in paragraph {i} <<<")

# Step 4: Check what gets merged into chunks
chunks = []
current_chunk = ""
for para in paragraphs:
    if len(para) > CHUNK_SIZE:
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        # Would be further split
        chunks.append(f"[LONG para, len={len(para)}, would be split]")
        current_chunk = ""
    elif len(current_chunk) + len(para) + 2 > CHUNK_SIZE and current_chunk:
        chunks.append(current_chunk.strip())
        current_chunk = para
    else:
        if current_chunk:
            current_chunk += "\n" + para
        else:
            current_chunk = para

if current_chunk.strip():
    chunks.append(current_chunk.strip())

print(f"\nFinal chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"Chunk {i} (len={len(c)}): {repr(c[:80])}...")
    if '第六' in c or '待遇' in c or '時薪' in c:
        print("  >>> CONTAINS TARGET <<<")
