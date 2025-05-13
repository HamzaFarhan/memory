import asyncio
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage, ToolCallPart

load_dotenv()

MEMORY_ITER_LIMIT = 10
MEMORY_FILE_PATH = "memories.json"


class NothingToRemember(BaseModel):
    pass


async def memorize(agent: Agent, message_history: list[ModelMessage], iter_limit: int = MEMORY_ITER_LIMIT):
    memorized = False
    iter_count = 0
    while not memorized and iter_count < iter_limit:
        user_prompt = (
            "<--- The conversation is over. Please update your knowledge graph --->\n"
            "If there is nothing to remember, just return a 'NothingToRemember' object."
        )
        async with agent.iter(
            user_prompt=user_prompt,
            message_history=message_history,
            output_type=str | NothingToRemember,  # type: ignore
        ) as run:
            async for node in run:
                if agent.is_call_tools_node(node):
                    for part in node.model_response.parts:
                        if isinstance(part, ToolCallPart) and part.tool_name in (
                            "add_entities",
                            "add_relations",
                            "add_observations",
                        ):
                            memorized = True
        if run.result and isinstance(run.result.output, NothingToRemember):
            break
        iter_count += 1


memory_server = MCPServerStdio(
    command="uv", args=["run", "memory_mcp.py"], env={"MEMORY_FILE_PATH": MEMORY_FILE_PATH}
)

agent = Agent(
    "google-gla:gemini-2.0-flash", instructions=Path("memory_prompt.txt").read_text(), mcp_servers=[memory_server]
)


async def run_agent():
    user_prompt = input("> ")
    message_history = None
    while True:
        async with agent.run_mcp_servers():
            res = await agent.run(user_prompt=user_prompt, message_history=message_history)
            user_prompt = input(f"{res.output.strip()} > ")
            message_history = res.all_messages()
            if user_prompt.lower().strip() == "q":
                await memorize(agent, message_history)
                break


if __name__ == "__main__":
    asyncio.run(run_agent())
