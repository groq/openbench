"""
Argument/embedding generation for synthetic MCP servers.

This generates embeddings for the synthetic servers defined in servers.json,
storing them in the same format as the live copilot embeddings.
"""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm  # type: ignore[import-untyped]

import mcp.types as types
import openai

from openbench.utils.text import LIVEMCPBENCH_TOOL_SUMMARY_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)

embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
embedding_api_url = os.getenv("EMBEDDING_BASE_URL")

abstract_api_key = os.getenv("ABSTRACT_API_KEY") or os.getenv("OPENAI_API_KEY")
abstract_model = os.getenv("ABSTRACT_MODEL", "gpt-4.1-2025-04-14")
abstract_api_url = os.getenv("ABSTRACT_BASE_URL")


def _synthetic_mcp_dir() -> Path:
    """Get the synthetic_mcp directory path."""
    current = Path(__file__).resolve()
    repo_root = current.parent.parent.parent.parent.parent.parent
    return repo_root / "synthetic_mcp"


def _user_cache_dir() -> Path:
    """Get the user cache directory for copilot embeddings."""
    return Path(
        os.path.expanduser("~/.openbench/progressivemcpbench/copilot")
    ).resolve()


def _compute_servers_hash(servers_config: dict[str, Any]) -> str:
    """Compute a hash of the servers configuration for cache invalidation.

    The hash is based on server names, descriptions, and tool definitions,
    so any change to these will bust the cache.
    """
    # Extract only the fields that affect embeddings
    hashable_data: list[dict[str, Any]] = []
    for server_name in sorted(servers_config.keys()):
        server = servers_config[server_name]
        server_data = {
            "name": server_name,
            "description": server.get("description", ""),
            "tools": [],
        }
        for tool in server.get("tools", []):
            server_data["tools"].append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                }
            )
        hashable_data.append(server_data)

    # Create deterministic JSON and hash it
    content = json.dumps(hashable_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _default_output_path(servers_config: dict[str, Any] | None = None) -> Path:
    """Get the default output path for embeddings.

    If servers_config is provided, includes a content hash in the filename
    to ensure cache invalidation on any config changes.
    """
    if servers_config:
        config_hash = _compute_servers_hash(servers_config)
        return (
            _user_cache_dir()
            / "config"
            / f"synthetic_mcp_arg_{embedding_model}_{abstract_model}_{config_hash}.json"
        )
    # Fallback without hash (for backwards compatibility)
    return (
        _user_cache_dir()
        / "config"
        / f"synthetic_mcp_arg_{embedding_model}_{abstract_model}.json"
    )


class SyntheticMcpArgGenerator:
    """Generate embeddings for synthetic MCP servers."""

    def __init__(
        self,
        servers_json_path: Path | None = None,
        output_file: Path | None = None,
    ):
        """Initialize the generator.

        Args:
            servers_json_path: Path to synthetic servers.json (default: synthetic_mcp/config/servers.json)
            output_file: Output path for embeddings (default: cache dir with content hash)
        """
        self.servers_json_path = servers_json_path or (
            _synthetic_mcp_dir() / "config" / "servers.json"
        )

        if not self.servers_json_path.exists():
            raise FileNotFoundError(
                f"Servers config not found: {self.servers_json_path}"
            )

        with self.servers_json_path.open("r", encoding="utf-8") as f:
            self.servers_config = json.load(f)

        # Use content-hash-based path if no explicit output_file provided
        self.output_file = output_file or _default_output_path(self.servers_config)

        self.embedding_client = openai.AsyncOpenAI(
            api_key=embedding_api_key, base_url=embedding_api_url
        )
        self.summary_client = openai.AsyncOpenAI(
            api_key=abstract_api_key, base_url=abstract_api_url
        )

    async def _get_embedding(
        self, text: str, model: str = embedding_model
    ) -> list[float]:
        if not text:
            raise ValueError("Empty text provided for embedding generation")
        response = await self.embedding_client.embeddings.create(
            model=model,
            input=[text],
            encoding_format="float",
        )
        embedding = response.data[0].embedding
        if not embedding or len(embedding) == 0:
            raise ValueError(f"Empty embedding returned for text: {text[:100]}...")
        return embedding

    async def _generate_summary(
        self,
        server_name: str,
        server_desc: str,
        tools: list[types.Tool],
        model: str = abstract_model,
    ) -> str:
        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in tools]
        )

        prompt = LIVEMCPBENCH_TOOL_SUMMARY_PROMPT.format(
            server_name=server_name,
            server_desc=server_desc,
            tool_descriptions=tool_descriptions,
        )
        try:
            response = await self.summary_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical writer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.error(f"Summary Generation Error for '{server_name}': {e}")
            return f"Error generating summary for {server_name}"

    def _format_tool_parameters(self, tool: types.Tool) -> dict[str, str]:
        formatted_params: dict[str, str] = {}
        schema = tool.inputSchema
        if not schema or "properties" not in schema:
            return formatted_params

        properties = schema.get("properties", {})
        required_params = schema.get("required", [])

        for param_name, param_details in properties.items():
            param_type = param_details.get("type", "any")
            param_desc = param_details.get("description", "")

            if param_name not in required_params:
                formatted_params[param_name] = f"(Optional, {param_type}) {param_desc}"
            else:
                formatted_params[param_name] = f"({param_type}) {param_desc}"
        return formatted_params

    async def generate(self, force: bool = False) -> Path:
        """Generate embeddings for all synthetic servers.

        Args:
            force: If True, regenerate even if output file exists

        Returns:
            Path to the generated embeddings file
        """
        # Check if we can skip
        if self.output_file.exists() and not force:
            logger.info(f"Embeddings file already exists: {self.output_file}")
            return self.output_file

        # Load existing servers if any (for incremental updates)
        existing_servers_info: list[dict[str, Any]] = []
        existing_server_names: set[str] = set()

        if self.output_file.exists() and not force:
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        existing_servers_info = content
                        for server_data in existing_servers_info:
                            if "server_name" in server_data:
                                existing_server_names.add(server_data["server_name"])
                        logger.info(
                            f"Loaded {len(existing_server_names)} existing servers from {self.output_file}."
                        )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading existing servers: {e}")

        all_servers_info = existing_servers_info.copy()
        new_servers_processed_count = 0

        for server_name, server_data in tqdm(
            self.servers_config.items(), desc="Generating embeddings"
        ):
            if server_name in existing_server_names:
                continue

            server_description = server_data.get("description", "")
            tools_data = server_data.get("tools", [])

            # Convert to types.Tool objects
            tools = []
            for tool_data in tools_data:
                tool = types.Tool(
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description", ""),
                    inputSchema=tool_data.get("inputSchema", {}),
                )
                tools.append(tool)

            logger.info(f"Indexing server: {server_name}")

            # Generate summary
            server_summary = await self._generate_summary(
                server_name, server_description, tools
            )

            # Generate embeddings in parallel
            embedding_tasks = {
                "server_desc": self._get_embedding(server_description),
                "server_summary": self._get_embedding(server_summary),
            }
            for i, tool in enumerate(tools):
                if not tool.description:
                    raise ValueError(
                        f"Tool '{tool.name}' in server '{server_name}' has no description"
                    )
                embedding_tasks[f"tool_{i}"] = self._get_embedding(tool.description)

            embeddings_results = await asyncio.gather(*embedding_tasks.values())
            embeddings = dict(zip(embedding_tasks.keys(), embeddings_results))

            # Format tools
            formatted_tools = []
            for i, tool in enumerate(tools):
                formatted_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "description_embedding": embeddings[f"tool_{i}"],
                        "parameter": self._format_tool_parameters(tool),
                    }
                )

            server_output = {
                "server_name": server_name,
                "server_summary": server_summary,
                "server_description": server_description,
                "description_embedding": embeddings["server_desc"],
                "summary_embedding": embeddings["server_summary"],
                "tools": formatted_tools,
            }

            all_servers_info.append(server_output)

            # Write incrementally
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(all_servers_info, f, indent=2, ensure_ascii=False)
            new_servers_processed_count += 1

        logger.info("Indexing completed.")
        if new_servers_processed_count > 0:
            logger.info(
                f"Added {new_servers_processed_count} new servers to {self.output_file}."
            )
        else:
            logger.info("No new servers were added.")

        return self.output_file


async def generate_synthetic_embeddings(force: bool = False) -> Path:
    """Generate embeddings for synthetic MCP servers.

    Args:
        force: If True, regenerate even if output file exists

    Returns:
        Path to the embeddings file
    """
    generator = SyntheticMcpArgGenerator()
    return await generator.generate(force=force)


def get_synthetic_embeddings_path() -> Path:
    """Get the path to the synthetic embeddings file for current servers.json.

    The path includes a content hash of the servers config, so it automatically
    points to a new file if the config has changed.
    """
    servers_json_path = _synthetic_mcp_dir() / "config" / "servers.json"
    if servers_json_path.exists():
        with servers_json_path.open("r", encoding="utf-8") as f:
            servers_config = json.load(f)
        return _default_output_path(servers_config)
    return _default_output_path()


def synthetic_embeddings_exist() -> bool:
    """Check if synthetic embeddings have been generated for current config."""
    return get_synthetic_embeddings_path().exists()


if __name__ == "__main__":
    asyncio.run(generate_synthetic_embeddings(force=True))
