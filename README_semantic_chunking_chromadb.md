# Semantic Chunking + LangChain + ChromaDB Lab

## What this lab demonstrates
This lab shows how to use semantic chunking instead of fixed-size chunking.
Semantic chunking splits documents based on meaning, topic shift, or semantic distance.

## Files
- `semantic_chunking_chromadb_demo.py` - main Python demo
- `semantic_chunking_experiment_text.txt` - sample document for experimentation
- `requirements_semantic_chromadb.txt` - Python dependencies

## Run
```bash
pip install -r requirements_semantic_chromadb.txt
python semantic_chunking_chromadb_demo.py
```

## Default embedding model
The demo uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2` locally.

## Optional OpenAI mode
```bash
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY="your-key"
python semantic_chunking_chromadb_demo.py
```

## Suggested experiments
1. Change `BREAKPOINT_AMOUNT` from 80 to 70 or 90.
2. Compare semantic chunks with fixed-size chunks.
3. Add unrelated sections to the text file and observe chunk boundaries.
4. Ask different retrieval questions.
5. Inspect the `chroma_semantic_db` folder after running the script.
