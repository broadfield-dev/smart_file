import os
import datetime
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class SemanticExplorer:
    """A class to manage filesystem indexing and semantic search using ChromaDB."""

    def __init__(self, db_path="./chroma_db", collection_name="filesystem_index"):
        """
        Initializes the model and the persistent ChromaDB client.
        """
        print("Initializing SemanticExplorer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("SemanticExplorer initialized.")
        self.is_cancelled = False

    def get_status(self) -> str:
        """Returns the current status of the database."""
        count = self.collection.count()
        if count > 0:
            return f"Persistent index loaded with {count} items."
        return "Index is empty. Build the index to get started."

    def _get_file_snippet(self, path, max_len=500) -> str:
        """Safely reads the beginning of a file to use as a content snippet."""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                if '\x00' in f.read(1024): return ""
                f.seek(0)
                return f.read(max_len)
        except Exception:
            return ""

    def index_directory(self, root_path: str):
        """
        A generator that walks a directory, creates rich documents, and upserts them into ChromaDB.
        Yields status updates for progress tracking.
        """
        self.is_cancelled = False
        if not os.path.isdir(root_path):
            yield "Error: Provided path is not a valid directory."
            return

        yield "Scanning directories..."
        all_paths = []
        for root, dirs, files in os.walk(root_path):
            if self.is_cancelled: break
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.vscode', 'node_modules', '.idea', 'chroma_db']]
            for name in files + dirs:
                all_paths.append(os.path.join(root, name))
        
        if self.is_cancelled:
            yield f"Build cancelled. DB size: {self.collection.count()}"
            return

        total_paths = len(all_paths)
        yield f"Scan complete. Found {total_paths} items to process."

        batch_size = 128
        for i in tqdm(range(0, total_paths, batch_size), desc="Indexing Batches"):
            if self.is_cancelled: break
            
            batch_paths = all_paths[i:i+batch_size]
            docs, metadatas, ids = [], [], []

            for path_str in batch_paths:
                try:
                    stat = os.stat(path_str)
                    is_dir = os.path.isdir(path_str)
                    relative_path = os.path.relpath(path_str, root_path)
                    
                    doc = f"Type: {'Folder' if is_dir else 'File'}. Path: {relative_path.replace(os.sep, ' ')}. Tree: {' > '.join(Path(relative_path).parts)}. "
                    if not is_dir: doc += f"Content Snippet: {self._get_file_snippet(path_str)}"

                    docs.append(doc)
                    metadatas.append({
                        "full_path": path_str, "relative_path": relative_path, "is_dir": is_dir,
                        "size_bytes": stat.st_size, "modified_time": stat.st_mtime
                    })
                    ids.append(path_str)
                except FileNotFoundError:
                    continue

            if docs:
                self.collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
            
            yield f"Processing batch {i//batch_size + 1}... ({min(i+batch_size, total_paths)}/{total_paths})"

        final_count = self.collection.count()
        if self.is_cancelled:
            yield f"Build cancelled. The database now contains {final_count} items."
        else:
            yield f"Index build complete. The database now contains {final_count} items."

    def search(self, query: str, n_results: int = 20) -> list[dict]:
        """Performs a semantic search and returns a list of formatted results."""
        if self.collection.count() == 0: return []
        
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self.collection.count())
        )

        output = []
        if results['ids'][0]:
            for i, dist in enumerate(results['distances'][0]):
                meta = results['metadatas'][0][i]
                output.append({
                    "similarity": 1 - dist,
                    "path": meta['relative_path'],
                    "full_path": meta['full_path'], # <-- ADD THIS LINE
                    "type": "📁 Folder" if meta['is_dir'] else "📄 File",
                    "size": meta['size_bytes'] if not meta['is_dir'] else None,
                    "modified": datetime.datetime.fromtimestamp(meta['modified_time'])
                })
        return output


    def clear_index(self) -> int:
        """Deletes all items from the collection."""
        count = self.collection.count()
        if count > 0:
            ids_to_delete = self.collection.get(include=[])['ids']
            self.collection.delete(ids=ids_to_delete)
        return count
    
    def cancel_indexing(self):
        """Sets a flag to stop the current indexing process."""
        self.is_cancelled = True
