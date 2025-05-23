from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, Self

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, BeforeValidator, Field, model_validator
from pydantic_ai import ModelRetry

load_dotenv()

DEFAULT_MEMORY_FILE_PATH = "memory.json"
KG_LIMITS = {"small": 25_000, "medium": 50_000, "large": 100_000}


def load_memory_path() -> Path:
    return Path(os.getenv("MEMORY_FILE_PATH", DEFAULT_MEMORY_FILE_PATH))


class Entity(BaseModel):
    name: str
    entity_type: str = Field(..., description="For example, 'person', 'task', 'event'")
    observations: list[str]

    async def add_observations(self, observations: list[str]) -> None:
        self.observations = list(set(self.observations + observations))

    async def overwrite_observations(self, observations: list[str]) -> None:
        self.observations = observations


class Relation(BaseModel):
    relation_from: str
    relation_to: str
    relation_type: str


class CondensedObservations(BaseModel):
    entity_name: str
    condensed_observations: list[str]


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


async def get_knowledge_graph_size() -> str:
    """
    Get the size of the knowledge graph.
    """
    graph = await load_knowledge_graph()
    kg_tokens = len(graph.model_dump_json()) * 1.3
    for kg_size, limit in KG_LIMITS.items():
        if kg_tokens <= limit:
            return kg_size
    return "large"


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

    # Validate incoming relations before attempting to add them
    # This checks if the entities referred to by the new relations exist.
    if relations:  # Only validate if there are relations to add
        temp_relations_dict_for_validation = {
            (r.relation_from, r.relation_to, r.relation_type): r for r in relations
        }
        validate_relations(graph.entities, temp_relations_dict_for_validation)

    graph.relations.update({(r.relation_from, r.relation_to, r.relation_type): r for r in relations})
    await save_knowledge_graph(graph)


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


async def delete_entities(entity_names: list[str]) -> None:
    graph = await load_knowledge_graph()
    actually_deleted_entities: set[str] = set()

    for entity_name in entity_names:
        if not graph.entities.get(entity_name):
            continue
        del graph.entities[entity_name]
        actually_deleted_entities.add(entity_name)

    if not actually_deleted_entities:
        return

    # Remove relations connected to any of the deleted entities
    relations_to_keep = {
        key: relation
        for key, relation in graph.relations.items()
        if relation.relation_from not in actually_deleted_entities
        and relation.relation_to not in actually_deleted_entities
    }
    graph.relations = relations_to_keep

    await save_knowledge_graph(graph)


async def delete_relations(relations: list[Relation]) -> None:
    graph = await load_knowledge_graph()
    for relation in relations:
        relation_key = (relation.relation_from, relation.relation_to, relation.relation_type)
        if not graph.relations.get(relation_key):
            continue
        del graph.relations[relation_key]
    await save_knowledge_graph(graph)


async def search_nodes(
    query: str,
    search_mode: Literal["any_token", "all_tokens", "exact_phrase"] = "any_token",
) -> KnowledgeGraph:
    """
    Search for nodes in the graph that match the query and return a new graph with the results.

    Parameters
    ----------
    query : str
        The query string to search for.
    search_mode : Literal["any_token", "all_tokens", "exact_phrase"], optional
        Defines how the query string is matched against entity fields:
        - "exact_phrase": The entire query string must be found as a substring (case-insensitive).
        - "any_token": The query string is split into tokens (words). An entity matches if
                       any token is found in its fields (case-insensitive). This is the default.
        - "all_tokens": The query string is split into tokens. An entity matches if
                        all tokens are found in its fields (case-insensitive).
                        Tokens can appear in different fields or multiple times in one field.

    Returns
    -------
    KnowledgeGraph
        A new KnowledgeGraph containing entities that match the search criteria
        and relations connecting these matched entities.
    """
    graph = await load_knowledge_graph()

    stripped_query = query.strip()
    if not stripped_query:
        return KnowledgeGraph()  # No query, no results

    normalized_query_phrase = stripped_query.lower()
    query_tokens = [token.lower() for token in stripped_query.split() if token]

    # If stripped_query is not empty, query_tokens should also not be empty.
    # This check is more of a safeguard or for future query processing logic.
    if not query_tokens and (search_mode == "any_token" or search_mode == "all_tokens"):
        # If for some reason (e.g. query was only punctuation split away) no tokens remain,
        # but the original phrase was not empty, fallback to exact_phrase search.
        search_mode = "exact_phrase"

    matched_entities: dict[str, Entity] = {}
    for entity_name, e in graph.entities.items():
        # All searchable text fields of an entity, lowercased once.
        searchable_texts = [e.name.lower(), e.entity_type.lower()]
        searchable_texts.extend(obs.lower() for obs in e.observations)

        entity_matches = False
        if search_mode == "exact_phrase":
            if any(normalized_query_phrase in text for text in searchable_texts):
                entity_matches = True
        elif search_mode == "any_token":
            # This mode requires query_tokens to be non-empty.
            if query_tokens and any(token in text for text in searchable_texts for token in query_tokens):
                entity_matches = True
        elif search_mode == "all_tokens":
            # This mode requires query_tokens to be non-empty.
            # Each token must be found in at least one of the entity's texts.
            if query_tokens and all(any(token in text for text in searchable_texts) for token in query_tokens):
                entity_matches = True

        if entity_matches:
            matched_entities[entity_name] = e

    # Relations are filtered to include only those connecting entities that were matched.
    filtered_relations = {
        (r.relation_from, r.relation_to, r.relation_type): r
        for r in graph.relations.values()
        if r.relation_from in matched_entities and r.relation_to in matched_entities
    }
    return KnowledgeGraph(entities=matched_entities, relations=filtered_relations)


async def open_nodes(names: list[str]) -> KnowledgeGraph:
    """
    Retrieve specific nodes by name and the relations *between* them.

    Parameters
    ----------
    names : list[str]
        The names of the nodes to retrieve.

    Returns
    -------
    KnowledgeGraph
        A new KnowledgeGraph containing:
        - Entities: Only the entities whose names were specified in the `names` list.
        - Relations: Only the relations where *both* the 'from' and 'to' entities
          are among the specified `names` and thus included in the returned entities.
    """
    graph = await load_knowledge_graph()

    # Filter entities to include only those specified by name
    selected_entities = {
        name: graph.entities[name]
        for name in names
        if name in graph.entities  # Ensure entity exists before trying to access
    }

    # Filter relations to include only those connecting the selected_entities
    selected_relations = {
        (r.relation_from, r.relation_to, r.relation_type): r
        for r in graph.relations.values()
        if r.relation_from in selected_entities and r.relation_to in selected_entities
    }
    return KnowledgeGraph(entities=selected_entities, relations=selected_relations)


kg = KnowledgeGraph(
    entities={
        "alice": Entity(
            name="alice", entity_type="person", observations=["alice is a person", "alice is 20 years old"]
        ),
        "bob": Entity(name="bob", entity_type="person", observations=["bob is a person", "bob is 25 years old"]),
    },
    relations={
        ("alice", "bob", "friend"): Relation(relation_from="alice", relation_to="bob", relation_type="friend")
    },
)
