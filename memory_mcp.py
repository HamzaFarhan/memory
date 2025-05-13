from __future__ import annotations

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from memory_tools import (
    add_entities,
    add_observations,
    add_relations,
    delete_entities,
    delete_relations,
    get_knowledge_graph_size,
    load_knowledge_graph,
    open_nodes,
    search_nodes,
)

server = FastMCP(name="memory_mcp")


load_dotenv()


server.add_tool(get_knowledge_graph_size)
server.add_tool(load_knowledge_graph)
server.add_tool(add_entities)
server.add_tool(add_relations)
server.add_tool(add_observations)
server.add_tool(delete_entities)
server.add_tool(delete_relations)
server.add_tool(search_nodes)
server.add_tool(open_nodes)

if __name__ == "__main__":
    server.run()
