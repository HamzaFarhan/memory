# Memory MCP

A knowledge-graph-based memory system for AI agents that enables persistent information storage between conversations.

## Features

- Persistent memory storage using a knowledge graph structure
- Entity-relation model for organizing information
- Tools for adding, searching, and retrieving memories

## Tools

The system provides the following MCP tools:

- `load_knowledge_graph()`: Retrieves the entire knowledge graph
- `get_knowledge_graph_size()`: Returns the current size category of the graph ("small", "medium", or "large")
- `add_entities(entities)`: Adds new entities to the memory
- `add_relations(relations)`: Creates relationships between entities
- `add_observations(entity_name, observations)`: Adds observations to existing entities
- `delete_entities(entity_names)`: Removes entities from memory
- `delete_relations(relations)`: Removes relationships
- `search_nodes(query, search_mode)`: Searches for entities and relations matching a query. Supports three search modes:
  - "exact_phrase": Matches the entire query as a substring
  - "any_token": Matches if any word in the query matches (default)
  - "all_tokens": Matches if all words in the query match
- `open_nodes(names)`: Retrieves specific entities and their relationships between them

## Usage

Run the agent with:

```
uv run memory_agent.py
```

The agent will automatically:
1. Load its memory at the start of conversations
2. Reference relevant information during interactions
3. Update its memory with new information when the conversation ends

Exit a conversation by typing `q`.

## Configuration

Set the memory storage location with the `MEMORY_FILE_PATH` environment variable (defaults to `memory.json`).
