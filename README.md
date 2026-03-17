# Verito 
Verito adds local semantic search capabilities to obsidian using text embedding models (via ollama). 
Instead of keyword searching, this plugin allows you to search by meaning. 

# Features 
- Semantic search across your entire vault.
- Your choice of model.
- Fully local processing, your data never leaves your computer. 

# How it works 
The plugin connects to a local indexing backend that does the following:
- Parse markdown files.
- Chunk docs into semantic sections.
- Generate embeddings using the local model of your choice.
- Store vectors locally using ChromaDB.
- Perform semantic search at query time. 

No vault data is sent to external services. 

# Installation 
## Manual Installation 
1. Download the latest release.
2. Extract plugin folder into your vault path.
3. Enable plugin

# Requirements 
- Desktop version of obsidian. 
- Local backend binary included with the release. 

# Project Status 
Stable release to be published soon.

