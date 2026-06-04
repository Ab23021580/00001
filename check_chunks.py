import chromadb
import sys

client = chromadb.PersistentClient(path='chroma_db')
col = client.get_collection('rag_docs')
res = col.get(where={'source': {'$eq': '6261123f68ff3747511986.pdf'}})

with open('chunk_check.txt', 'w', encoding='utf-8') as f:
    f.write(f'Found {len(res["documents"])} chunks for this file\n\n')
    for i, doc in enumerate(res['documents']):
        f.write(f'--- Chunk {i} (len={len(doc)}) ---\n')
        f.write(doc[:500] + '\n\n')
        if '第六' in doc or '待遇' in doc or '時薪' in doc or '給付' in doc:
            f.write('  >>> CONTAINS TARGET KEYWORDS <<<\n\n')

print('Done, wrote chunk_check.txt')
