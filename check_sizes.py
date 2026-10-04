import os
for f in ['eureka_forbes.index', 'eureka_embeddings.npy', 'chunks.json']:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"{f}: {size_mb:.2f} MB")
        