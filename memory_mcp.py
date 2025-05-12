from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Self

from dotenv import load_dotenv
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, BeforeValidator, Field, model_validator
from pydantic_ai import ModelRetry

server = FastMCP(name="memory_mcp")


load_dotenv()


def load_memory_path() -> Path:
    return Path(os.getenv("MEMORY_FILE_PATH", "memory.json"))


class Entity(BaseModel):
    name: str
    entity_type: str = Field(..., description="For example, 'person', 'task', 'event'")
    observations: list[str]

    async def add_observations(self, observations: list[str]) -> None:
        self.observations = list(set(self.observations + observations))


class Relation(BaseModel):
    relation_from: str
    relation_to: str
    relation_type: str


def validate_relations(entities: dict[str, Entity], relations: dict[tuple[str, str, str], Relation]) -> None:
    for r in relations.values():
        if not entities.get(r.relation_from):
            raise ModelRetry(f"Entity '{r.relation_from}' not found in graph")
        if not entities.get(r.relation_to):
            raise ModelRetry(f"Entity '{r.relation_to}' not found in graph")


def validate_relation_key(v: tuple[str, str, str] | str) -> tuple[str, str, str]:
    if isinstance(v, str):
        relation_key = tuple(v.split(","))
        if len(relation_key) != 3:
            raise ValueError("Relation key must be a tuple of three strings")
        return relation_key
    return v


RelationKey = Annotated[tuple[str, str, str], BeforeValidator(validate_relation_key)]


class KnowledgeGraph(BaseModel):
    entities: dict[str, Entity] = Field(default_factory=dict)
    relations: dict[RelationKey, Relation] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_relations(self) -> Self:
        validate_relations(self.entities, self.relations)
        return self


async def save_knowledge_graph(graph: KnowledgeGraph) -> None:
    """
    Save the knowledge graph to the memory file.
    """
    memory_file_path = load_memory_path()
    memory_file_path.write_text(graph.model_dump_json())


@server.tool()
async def load_knowledge_graph() -> KnowledgeGraph:
    """
    Load the knowledge graph from the memory file.
    """
    memory_file_path = load_memory_path()
    if not memory_file_path.exists():
        return KnowledgeGraph()
    try:
        return KnowledgeGraph.model_validate_json(memory_file_path.read_text())
    except Exception as e:
        logger.error(f"Error loading graph from {memory_file_path}: {e}")
        return KnowledgeGraph()


@server.tool()
async def add_entities(entities: list[Entity]) -> None:
    """
    Add entities to the knowledge graph.

    Parameters
    ----------
    entities : list[Entity]
        The entities to add to the graph.
    """
    graph = await load_knowledge_graph()
    graph.entities.update({e.name: e for e in entities})
    await save_knowledge_graph(graph)


@server.tool()
async def add_relations(relations: list[Relation]) -> None:
    """
    Add relations to the graph.

    Parameters
    ----------
    relations : list[Relation]
        The relations to add to the graph.
        A relation is a tuple of (relation_from, relation_to, relation_type).
        The relation_from and relation_to are the names of the entities that are connected by the relation.
    """
    graph = await load_knowledge_graph()
    validate_relations(graph.entities, graph.relations)
    graph.relations.update({(r.relation_from, r.relation_to, r.relation_type): r for r in relations})
    await save_knowledge_graph(graph)


@server.tool()
async def add_observations(entity_name: str, observations: list[str]) -> None:
    """
    Add observations to an entity.

    Parameters
    ----------
    entity_name : str
        The name of the entity to add observations to.
    observations : list[str]
        The observations to add to the entity.
    """
    graph = await load_knowledge_graph()
    if not graph.entities.get(entity_name):
        raise ModelRetry(f"Entity {entity_name} not found in graph")
    await graph.entities[entity_name].add_observations(observations=observations)
    await save_knowledge_graph(graph)


@server.tool()
async def delete_entities(entity_names: list[str]) -> None:
    graph = await load_knowledge_graph()
    for entity_name in entity_names:
        if not graph.entities.get(entity_name):
            continue
        del graph.entities[entity_name]
    await save_knowledge_graph(graph)


@server.tool()
async def delete_relations(relations: list[Relation]) -> None:
    graph = await load_knowledge_graph()
    for relation in relations:
        relation_key = (relation.relation_from, relation.relation_to, relation.relation_type)
        if not graph.relations.get(relation_key):
            continue
        del graph.relations[relation_key]
    await save_knowledge_graph(graph)


@server.tool()
async def search_nodes(query: str) -> KnowledgeGraph:
    """
    Search for nodes in the graph that match the query and return a new graph with the results.

    Parameters
    ----------
    query : str
        The query string to search for in entity names, types, and observations.

    Returns
    -------
    KnowledgeGraph
        A new KnowledgeGraph containing only the entities and relations that match
        the search query. Entities match if the query appears in their name,
        type, or observations. Relations are included if they connect matching
        entities.
    """
    graph = await load_knowledge_graph()
    filtered_entities = {
        e.name: e
        for e in graph.entities.values()
        if query in e.name.lower()
        or query in e.entity_type.lower()
        or any(query in obs.lower() for obs in e.observations)
    }
    filtered_relations = {
        (r.relation_from, r.relation_to, r.relation_type): r
        for r in graph.relations.values()
        if query in r.relation_from.lower() or query in r.relation_to.lower() or query in r.relation_type.lower()
    }
    return KnowledgeGraph(entities=filtered_entities, relations=filtered_relations)


@server.tool()
async def open_nodes(names: list[str]) -> KnowledgeGraph:
    """
    Open nodes in the graph based on the names of the nodes.

    Parameters
    ----------
    names : list[str]
        The names of the nodes to open.
    """
    graph = await load_knowledge_graph()
    filtered_entities = {e.name: e for e in graph.entities.values() if e.name in names}
    filtered_relations = {
        (r.relation_from, r.relation_to, r.relation_type): r
        for r in graph.relations.values()
        if r.relation_from in names or r.relation_to in names
    }
    return KnowledgeGraph(entities=filtered_entities, relations=filtered_relations)


if __name__ == "__main__":
    server.run()
