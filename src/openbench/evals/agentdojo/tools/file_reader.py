from inspect_ai.tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel


class Filesystem(BaseModel):
    files: dict[str, str]


@tool
def read_file() -> Tool:
    async def execute(file_path: str) -> str:
        """
        Reads the contents of the file at the given path.

        Args:
            file_path: The path to the file to read.

        Returns:
            The contents of the file
        """
        filesystem = store().get("filesystem")
        assert isinstance(filesystem, Filesystem)
        return filesystem.files.get(file_path, "")

    return execute
