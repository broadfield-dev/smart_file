# SmartFile: Semantic Filesystem Explorer

SmartFile is a powerful Python library that indexes your local filesystem, including file contents, to enable fast and intuitive semantic searching. It uses `sentence-transformers` for state-of-the-art text embeddings and `ChromaDB` for persistent, efficient vector storage and search.

This repository also includes a Gradio-based demo application to showcase the library's capabilities.

## Features

-   **Persistent Indexing**: Build the search index once; it's saved to disk for instant loading on subsequent runs.
-   **Rich Metadata**: Indexes file paths, directory structure, modification times, and file sizes.
-   **Content-Aware Search**: For text files, a snippet of the content is included in the index, allowing you to search for files based on what's *inside* them.
-   **Simple API**: A clean `SemanticExplorer` class abstracts away the complexity.
-   **Extensible**: Easily integrate into your own scripts and applications.

## Installation

You can install the library directly from GitHub:

```bash
pip install git+https://github.com/broadfield-dev/smartfile.git
```

### For Development

To set up the project for development (including the demo):

```bash
# 1. Clone the repository
git clone https://github.com/broadfield-dev/smartfile.git
cd filesem-explorer

# 2. Install the library in editable mode
pip install -e .

# 3. Install the demo dependencies
pip install -r demo/requirements.txt
```

## Usage

### As a Python Library

Here is a minimal example of how to use `filesem` in your own script:

```python
from filesem import SemanticExplorer
import os

# Initialize the explorer (it will use a './chroma_db' folder by default)
explorer = SemanticExplorer()

# Get the current status
print(explorer.get_status())

# Index a directory (this is a generator for progress tracking)
# The index will be updated, not rebuilt from scratch.
target_dir = os.getcwd()
print(f"Indexing directory: {target_dir}")
for status in explorer.index_directory(target_dir):
    print(status)

# Perform a search
query = "a function that processes user data"
results = explorer.search(query)

print(f"\nTop search results for: '{query}'")
for result in results:
    print(f"- Similarity: {result['similarity']:.3f}, Path: {result['path']}")
```

### Running the Gradio Demo
Clone this repository
```bash
git clone https://github.com/broadfield-dev/smartfile.git
```

Install the dependencies 
```bash
pip install -r smartfile/requirements.txt
```

Run the demo from the command line
```bash
python smartfile/demo.py
```

The application will launch on a local URL, providing a full UI for browsing, indexing, and searching your filesystem.
