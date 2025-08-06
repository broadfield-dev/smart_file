import os
import datetime
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

class SemanticExplorer:
    def __init__(self, db_path="./chroma_db", collection_name="filesystem_index"):
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
        count = self.collection.count()
        if count > 0:
            return f"Persistent index loaded with {count} items."
        return "Index is empty. Build the index to get started."

    def _get_file_snippet(self, path, max_len=500) -> str:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                if '\x00' in f.read(1024): return ""
                f.seek(0)
                return f.read(max_len)
        except Exception:
            return ""

    def index_directory(self, root_path: str, progress_callback=None):
        self.is_cancelled = False
        if not os.path.isdir(root_path):
            yield "Error: Provided path is not a valid directory."
            return
        if progress_callback:
            progress_callback(0, desc="Scanning directories...")
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
        if progress_callback:
            progress_callback(0.05, desc=f"Scan complete. Found {total_paths} items.")
        yield f"Scan complete. Found {total_paths} items to process."
        batch_size = 128
        for i in range(0, total_paths, batch_size):
            if self.is_cancelled: break
            progress_fraction = i / total_paths
            status_message = f"Processing batch {i//batch_size + 1}... ({min(i+batch_size, total_paths)}/{total_paths})"
            if progress_callback:
                progress_callback(progress_fraction, desc=status_message)
            yield status_message
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
        final_count = self.collection.count()
        if self.is_cancelled:
            yield f"Build cancelled. The database now contains {final_count} items."
        else:
            if progress_callback:
                progress_callback(1, desc="Complete!")
            yield f"Index build complete. The database now contains {final_count} items."

    def search(self, query: str, n_results: int = 20, metadata_filters: dict = None) -> list[dict]:
        if self.collection.count() == 0: return []
        
        db_filters = metadata_filters.copy() if metadata_filters else {}
        path_contains_filter = None
        
        def extract_path_filter(conditions):
            nonlocal path_contains_filter
            remaining_conditions = []
            for cond in conditions:
                if 'relative_path' in cond and '$contains' in cond['relative_path']:
                    path_contains_filter = cond['relative_path']['$contains']
                else:
                    remaining_conditions.append(cond)
            return remaining_conditions

        if '$and' in db_filters:
            db_filters['$and'] = extract_path_filter(db_filters['$and'])
            if not db_filters['$and']:
                del db_filters['$and']
        elif 'relative_path' in db_filters and '$contains' in db_filters['relative_path']:
            path_contains_filter = db_filters['relative_path']['$contains']
            del db_filters['relative_path']

        query_params = {
            "query_embeddings": self.model.encode([query]).tolist(),
            "n_results": min(n_results * 5, self.collection.count())
        }
        if db_filters:
            query_params["where"] = db_filters
        
        results = self.collection.query(**query_params)
        
        output = []
        if not results['ids'][0]: return []

        for i, dist in enumerate(results['distances'][0]):
            meta = results['metadatas'][0][i]
            
            if path_contains_filter and path_contains_filter not in meta['relative_path']:
                continue

            output.append({
                "similarity": 1 - dist,
                "path": meta['relative_path'],
                "full_path": meta['full_path'],
                "type": "📁 Folder" if meta['is_dir'] else "📄 File",
                "size": meta['size_bytes'] if not meta['is_dir'] else None,
                "modified": datetime.datetime.fromtimestamp(meta['modified_time'])
            })
            
            if len(output) >= n_results:
                break
                
        return output

    def clear_index(self) -> int:
        count = self.collection.count()
        if count > 0:
            ids_to_delete = self.collection.get(include=[])['ids']
            self.collection.delete(ids=ids_to_delete)
        return count
    
    def cancel_indexing(self):
        self.is_cancelled = True
