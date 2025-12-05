#!/usr/bin/env python3
"""
Synthetic HTTP MCP Server

A single HTTP server that masquerades as multiple MCP servers.
Routes requests based on path: /mcp/{server_name}/tools/{tool_name}

Handler types:
- filesystem: Read files from the synthetic data directory
- table_lookup: Look up data by key from JSON files
- excel_reader: Read Excel files
- static_json: Return fixed JSON responses
- compute: Perform simple computations
"""

import csv
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Default paths - can be overridden
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "servers.json"
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data"


class SyntheticMCPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for synthetic MCP server."""

    servers_config: dict = {}
    data_path: Path = DEFAULT_DATA_PATH
    web_corpus_metadata: dict = {}  # Cached web corpus metadata

    def log_message(self, format: str, *args: Any) -> None:
        """Override to reduce logging noise."""
        pass  # Suppress default logging

    def send_json_response(self, data: Any, status: int = 200) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_error_response(self, message: str, status: int = 400) -> None:
        """Send an error response."""
        self.send_json_response({"error": message}, status)

    def do_GET(self) -> None:
        """Handle GET requests - list servers or tools."""
        parsed = urlparse(self.path)
        path_parts = parsed.path.strip("/").split("/")

        if path_parts == [""]:
            # Root - list all servers
            servers = list(self.servers_config.keys())
            self.send_json_response({"servers": servers})

        elif len(path_parts) == 2 and path_parts[0] == "mcp":
            # /mcp/{server_name} - list tools
            server_name = path_parts[1]
            if server_name not in self.servers_config:
                self.send_error_response(f"Unknown server: {server_name}", 404)
                return

            server = self.servers_config[server_name]
            tools = [{"name": t["name"], "description": t.get("description", "")} for t in server.get("tools", [])]
            self.send_json_response({"server": server_name, "tools": tools})

        else:
            self.send_error_response("Invalid path", 404)

    def do_POST(self) -> None:
        """Handle POST requests - execute tools."""
        parsed = urlparse(self.path)
        path_parts = parsed.path.strip("/").split("/")

        # Expected: /mcp/{server_name}/tools/{tool_name}
        if len(path_parts) != 4 or path_parts[0] != "mcp" or path_parts[2] != "tools":
            self.send_error_response("Invalid path. Expected: /mcp/{server}/tools/{tool}", 400)
            return

        server_name = path_parts[1]
        tool_name = path_parts[3]

        # Find server and tool
        if server_name not in self.servers_config:
            self.send_error_response(f"Unknown server: {server_name}", 404)
            return

        server = self.servers_config[server_name]
        tool = None
        for t in server.get("tools", []):
            if t["name"] == tool_name:
                tool = t
                break

        if not tool:
            self.send_error_response(f"Unknown tool: {tool_name} in server {server_name}", 404)
            return

        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            params = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error_response("Invalid JSON in request body", 400)
            return

        # Execute handler
        handler = tool.get("handler", {})
        handler_type = handler.get("type", "static_json")

        try:
            result = self.execute_handler(handler_type, handler, params, tool)
            self.send_json_response({"result": result})
        except Exception as e:
            self.send_error_response(f"Handler error: {e!s}", 500)

    def execute_handler(self, handler_type: str, handler: dict, params: dict, tool: dict) -> Any:
        """Execute a handler based on its type."""
        if handler_type == "static_json":
            return handler.get("response", {})

        elif handler_type == "compute":
            return self.handle_compute(handler, params)

        elif handler_type == "filesystem":
            return self.handle_filesystem(handler, params, tool)

        elif handler_type == "table_lookup":
            return self.handle_table_lookup(handler, params)

        elif handler_type == "excel_reader":
            return self.handle_excel_reader(handler, params, tool)

        elif handler_type == "web_corpus":
            return self.handle_web_corpus(handler, params, tool)

        elif handler_type == "url_search":
            return self.handle_url_search(handler, params, tool)

        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

    def handle_compute(self, handler: dict, params: dict) -> Any:
        """Handle compute operations."""
        operation = handler.get("operation", "")

        if operation == "fixed_value":
            return handler.get("value")

        raise ValueError(f"Unknown compute operation: {operation}")

    def handle_filesystem(self, handler: dict, params: dict, tool: dict) -> Any:
        """Handle filesystem operations (read_file, list_directory, etc.)."""
        root = handler.get("root", "/root")
        tool_name = tool.get("name", "")

        # Map /root paths to our synthetic data directory
        files_root = self.data_path / "files" / "root"

        if tool_name in ("read_file", "document_reader", "get_document_text"):
            # Get the path parameter (different tools use different param names)
            file_path = params.get("path") or params.get("filePath") or params.get("filename", "")

            # Handle head/tail parameters for read_file
            head = params.get("head")
            tail = params.get("tail")

            # Normalize path - remove /root prefix and resolve
            if file_path.startswith("/root/"):
                rel_path = file_path[6:]  # Remove /root/
            elif file_path.startswith("/"):
                rel_path = file_path[1:]
            else:
                rel_path = file_path

            actual_path = files_root / rel_path

            if not actual_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Handle different file types
            suffix = actual_path.suffix.lower()

            if suffix == ".docx":
                # For Word documents, return extracted text
                return self.read_docx(actual_path)

            elif suffix == ".csv":
                # For CSV files, return as raw text (models can parse it)
                with open(actual_path, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                if head is not None:
                    lines = lines[: int(head)]
                elif tail is not None:
                    lines = lines[-int(tail) :]
                return "".join(lines)

            else:
                # For text files
                with open(actual_path, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()

                if head is not None:
                    lines = lines[: int(head)]
                elif tail is not None:
                    lines = lines[-int(tail) :]

                return "".join(lines)

        elif tool_name == "read_multiple_files":
            paths = params.get("paths", [])
            results = {}
            for file_path in paths:
                try:
                    result = self.handle_filesystem(handler, {"path": file_path}, {"name": "read_file"})
                    results[file_path] = {"content": result, "error": None}
                except Exception as e:
                    results[file_path] = {"content": None, "error": str(e)}
            return results

        elif tool_name == "list_directory":
            dir_path = params.get("path", "")

            # Normalize path
            if dir_path.startswith("/root/"):
                rel_path = dir_path[6:]
            elif dir_path.startswith("/root"):
                rel_path = dir_path[5:] if len(dir_path) > 5 else ""
            elif dir_path.startswith("/"):
                rel_path = dir_path[1:]
            else:
                rel_path = dir_path

            actual_path = files_root / rel_path if rel_path else files_root

            if not actual_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")

            entries = []
            for entry in sorted(actual_path.iterdir()):
                prefix = "[DIR]" if entry.is_dir() else "[FILE]"
                entries.append(f"{prefix} {entry.name}")

            return "\n".join(entries)

        elif tool_name == "read_pdf":
            # Handle PDF reading - return metadata from our API data
            sources = params.get("sources", [])
            results = []

            pdf_metadata = self.load_api_data("pdf_metadata.json")

            for source in sources:
                path = source.get("path", "")
                # Normalize path
                if not path.startswith("/root"):
                    path = f"/root/{path}"

                if path in pdf_metadata:
                    meta = pdf_metadata[path]
                    # Check if metadata is empty or invalid
                    if not meta or not isinstance(meta, dict) or not meta.get('title'):
                        results.append({
                            "path": path,
                            "error": "PDF not found in metadata"
                        })
                    else:
                        results.append(
                            {
                                "path": path,
                                "metadata": meta,
                                "page_count": meta.get("page_count", 0),
                                "text": f"[PDF Content] Title: {meta.get('title', 'Unknown')}\nAuthors: {', '.join(meta.get('authors', []))}",
                            }
                        )
                else:
                    # Generic response for unknown PDFs
                    results.append({"path": path, "error": "PDF not found in metadata"})

            return results

        else:
            raise ValueError(f"Unknown filesystem tool: {tool_name}")

    def read_docx(self, path: Path) -> str:
        """Extract text from a Word document."""
        # For simplicity, check if we have pre-computed content
        # In production, would use python-docx
        filename = path.name.lower()

        if "exchange" in filename:
            return "Everyone gave a gift."

        return f"[DOCX content from {path.name}]"

    def read_csv(self, path: Path) -> list[dict]:
        """Read a CSV file and return as list of dicts."""
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def handle_table_lookup(self, handler: dict, params: dict) -> Any:
        """Handle table lookup from JSON data files."""
        dataset_path = handler.get("dataset", "")
        key_field = handler.get("key_field", "")

        # Load the dataset
        data = self.load_api_data(dataset_path)

        # Find the key in params - try common parameter names
        lookup_key = None
        for param_name in [key_field, "nct_id", "paper_id", "dependency", "artifact"]:
            if param_name in params:
                lookup_key = params[param_name]
                break

        if lookup_key is None:
            # Return all data if no key specified (for search operations)
            return data

        # Look up the value
        if lookup_key in data:
            return data[lookup_key]

        # Try with normalized key (remove version suffix for maven)
        base_key = lookup_key.rsplit(":", 1)[0] if ":" in lookup_key else lookup_key
        if base_key in data:
            result = data[base_key].copy()
            result["queried_key"] = lookup_key
            return result

        return {"error": f"Key not found: {lookup_key}"}

    def handle_excel_reader(self, handler: dict, params: dict, tool: dict) -> Any:
        """Handle Excel reading operations."""
        tool_name = tool.get("name", "")
        root = handler.get("root", "/root")

        # Get file path
        file_path = params.get("fileAbsolutePath") or params.get("inputPath", "")

        # Map to our data directory
        if file_path.startswith("/root/"):
            rel_path = file_path[6:]
        else:
            rel_path = file_path.lstrip("/")

        actual_path = self.data_path / "files" / "root" / rel_path

        if not actual_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        # Use openpyxl to read the file
        try:
            import openpyxl

            wb = openpyxl.load_workbook(actual_path, data_only=True)
        except ImportError:
            return {"error": "openpyxl not installed"}

        if tool_name in ("excel_describe_sheets", "describe_sheets"):
            sheets = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                sheets.append({"name": sheet_name, "rows": ws.max_row, "columns": ws.max_column})
            return {"sheets": sheets}

        elif tool_name in ("excel_read_sheet", "excel_read"):
            sheet_name = params.get("sheetName")
            if sheet_name:
                ws = wb[sheet_name]
            else:
                ws = wb.active

            # Read the data
            cell_range = params.get("range", "")
            include_headers = params.get("includeHeaders", True)

            rows = []
            for row in ws.iter_rows(values_only=True):
                rows.append(list(row))

            if include_headers and rows:
                headers = rows[0]
                data = []
                for row in rows[1:]:
                    data.append(dict(zip(headers, row)))
                return {"headers": headers, "data": data, "row_count": len(data)}

            return {"data": rows, "row_count": len(rows)}

        return {"error": f"Unknown Excel tool: {tool_name}"}

    def get_web_corpus_metadata(self) -> dict:
        """Load and cache web corpus metadata."""
        if not self.web_corpus_metadata:
            metadata_path = self.data_path / "web" / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    data = json.load(f)
                    # Filter out schema/comment fields
                    self.web_corpus_metadata = {
                        k: v for k, v in data.items() if not k.startswith("_")
                    }
        return self.web_corpus_metadata

    def get_web_search_index(self) -> list:
        """Load web search index."""
        index_path = self.data_path / "web" / "search_index.json"
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
                return data.get("entries", [])
        return []

    def handle_web_corpus(self, handler: dict, params: dict, tool: dict) -> Any:
        """Handle web corpus operations (Playwright-like fetch)."""
        operation = handler.get("operation", "get_visible_html")
        url = params.get("url", "")

        if not url:
            return {"error": "URL parameter is required", "status": 400}

        # Load corpus metadata
        metadata = self.get_web_corpus_metadata()

        if operation == "navigate":
            # Check if URL exists in corpus
            if url in metadata:
                entry = metadata[url]
                return {
                    "page_id": entry.get("id", "unknown"),
                    "url": url,
                    "title": entry.get("title", ""),
                    "success": True,
                }
            else:
                return {
                    "error": "URL not found in synthetic web corpus",
                    "status": 404,
                    "url": url,
                }

        elif operation == "get_visible_html":
            if url not in metadata:
                return {
                    "error": "URL not found in synthetic web corpus",
                    "status": 404,
                    "url": url,
                }

            entry = metadata[url]
            html_path = entry.get("html_path", "")
            if not html_path:
                return {"error": "No HTML path configured for URL", "status": 500, "url": url}

            # Load HTML content
            full_html_path = self.data_path / html_path
            if not full_html_path.exists():
                return {"error": "HTML file not found", "status": 500, "url": url}

            with open(full_html_path, encoding="utf-8", errors="replace") as f:
                html_content = f.read()

            return {
                "url": url,
                "html": html_content,
                "title": entry.get("title", ""),
            }

        elif operation == "screenshot":
            # For screenshot, just return metadata about what would be captured
            if url not in metadata:
                return {
                    "error": "URL not found in synthetic web corpus",
                    "status": 404,
                    "url": url,
                }
            entry = metadata[url]
            return {
                "url": url,
                "title": entry.get("title", ""),
                "screenshot": "[Screenshot of page - see HTML content for actual data]",
            }

        else:
            return {"error": f"Unknown web_corpus operation: {operation}", "status": 400}

    def handle_url_search(self, handler: dict, params: dict, tool: dict) -> Any:
        """Handle URL search operations.

        This performs a simple keyword-based search over the web corpus.
        For more sophisticated search, an external LLM can be integrated.
        """
        query = params.get("query", "")
        max_results = params.get("max_results", 5)

        if not query:
            return {"error": "Query parameter is required", "results": []}

        # Load search index
        search_index = self.get_web_search_index()
        metadata = self.get_web_corpus_metadata()

        # Simple keyword matching
        query_lower = query.lower()
        query_terms = query_lower.split()

        scored_results = []
        for entry in search_index:
            score = 0
            searchable_text = " ".join(
                [
                    entry.get("title", ""),
                    entry.get("short_description", ""),
                    " ".join(entry.get("tags", [])),
                    " ".join(entry.get("example_queries", [])),
                ]
            ).lower()

            # Score based on term matches
            for term in query_terms:
                if term in searchable_text:
                    score += 1
                # Bonus for exact phrase match
                if query_lower in searchable_text:
                    score += 2

            if score > 0:
                scored_results.append((score, entry))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Build results
        results = []
        for score, entry in scored_results[:max_results]:
            url = entry.get("url", "")
            results.append(
                {
                    "id": entry.get("id", ""),
                    "url": url,
                    "title": entry.get("title", ""),
                    "description": entry.get("short_description", ""),
                }
            )

        return {
            "query": query,
            "results": results,
            "total_found": len(scored_results),
        }

    def load_api_data(self, dataset_path: str) -> dict:
        """Load a JSON dataset from the API data directory."""
        # Handle different path formats
        if dataset_path.startswith("data/"):
            rel_path = dataset_path[5:]  # Remove 'data/' prefix
        else:
            rel_path = dataset_path

        # Try in api/ subdirectory first
        full_path = self.data_path / "api" / Path(rel_path).name
        if not full_path.exists():
            full_path = self.data_path / rel_path

        if not full_path.exists():
            return {}

        with open(full_path, encoding="utf-8") as f:
            return json.load(f)


def create_server(
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_path: Path = DEFAULT_DATA_PATH,
    port: int = 8765,
) -> HTTPServer:
    """Create and configure the HTTP server."""
    # Load server configuration
    with open(config_path, encoding="utf-8") as f:
        servers_config = json.load(f)

    # Configure the handler class
    SyntheticMCPHandler.servers_config = servers_config
    SyntheticMCPHandler.data_path = data_path

    server = HTTPServer(("", port), SyntheticMCPHandler)
    return server


def main():
    """Run the synthetic MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic HTTP MCP Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to servers.json")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to data directory")
    args = parser.parse_args()

    server = create_server(config_path=args.config, data_path=args.data, port=args.port)

    print(f"Synthetic MCP Server starting on port {args.port}")
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Servers: {list(SyntheticMCPHandler.servers_config.keys())}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
