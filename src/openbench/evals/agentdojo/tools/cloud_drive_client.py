import datetime
from typing import Any

from inspect_ai.tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel, EmailStr, model_validator
from typing_extensions import Self

from openbench.evals.agentdojo.tools.types import (
    CloudDriveFile,
    CloudDriveFileID,
    SharingPermission,
)
from openbench.evals.agentdojo.utils import tool_result_to_str


class InitialCloudDrive(BaseModel):
    account_email: EmailStr
    initial_files: list[CloudDriveFile]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        all_ids = [file.id_ for file in self.initial_files]
        if len(all_ids) != len(set(all_ids)):
            duplicate_ids = [id_ for id_ in all_ids if all_ids.count(id_) > 1]
            raise ValueError(f"File IDs must be unique. Duplicates: {duplicate_ids}")
        return self


class CloudDrive(BaseModel):
    account_email: EmailStr
    files: dict[CloudDriveFileID, CloudDriveFile]

    @classmethod
    def create(cls, dict: dict[str, Any]) -> "CloudDrive":
        initial_drive = InitialCloudDrive.model_validate(dict)
        return cls(
            account_email=initial_drive.account_email,
            files={file.id_: file for file in initial_drive.initial_files},
        )

    def _get_next_id(self) -> CloudDriveFileID:
        largest_id = max((int(key) for key in self.files.keys()), default=0)
        return CloudDriveFileID(largest_id + 1)

    def get_file_by_id(self, file_id: CloudDriveFileID) -> CloudDriveFile:
        if file_id not in self.files:
            raise ValueError(f"File with ID '{file_id}' not found.")
        return self.files[file_id]

    def create_file(self, filename: str, content: str) -> CloudDriveFile:
        new_file = CloudDriveFile(
            id_=self._get_next_id(),
            filename=filename,
            content=content,
            owner=self.account_email,
            last_modified=datetime.datetime.now(),
        )
        self.files[new_file.id_] = new_file
        return new_file

    def delete_file(self, file_id: CloudDriveFileID) -> CloudDriveFile:
        if file_id not in self.files:
            raise ValueError(f"File with ID '{file_id}' not found.")
        file = self.files.pop(file_id)
        return file

    def search_files_by_filename(self, filename: str) -> list[CloudDriveFile]:
        return [
            file
            for file in self.files.values()
            if filename.lower() in file.filename.lower()
        ]

    def search_files(self, query: str) -> list[CloudDriveFile]:
        return [
            file
            for file in self.files.values()
            if query.lower() in file.content.lower()
        ]

    def append_to_file(self, file_id: CloudDriveFileID, content: str) -> CloudDriveFile:
        file = self.get_file_by_id(file_id)
        file.content += content
        file.last_modified = datetime.datetime.now()
        return file

    def get_file_sizes(self) -> dict[str, int]:
        return {file.id_: file.size for file in self.files.values()}


@tool
def search_files_by_filename() -> Tool:
    async def execute(filename: str) -> str:
        """
        Get a file from a cloud drive by its filename. It returns a list of files.

        Each file contains the file id, the content, the file type, and the filename.

        Args:
            filename: The name of the file to retrieve.

        Returns:
            List of files matching the filename
        """
        cloud_drive = store().get("cloud_drive")
        assert isinstance(cloud_drive, CloudDrive)
        files = cloud_drive.search_files_by_filename(filename)
        return tool_result_to_str(files)

    return execute


@tool
def create_file() -> Tool:
    async def execute(filename: str, content: str) -> str:
        """
        Create a new file in the cloud drive.

        Args:
            filename: The name of the file to create.
            content: The content of the file to create.

        Returns:
            The created file details
        """
        cloud_drive = store().get("cloud_drive")
        assert isinstance(cloud_drive, CloudDrive)
        file = cloud_drive.create_file(filename, content)
        return tool_result_to_str(file)

    return execute


@tool
def delete_file() -> Tool:
    async def execute(file_id: str) -> str:
        """
        Delete a file from the cloud drive by its ID.

        It returns the file that was deleted.

        Args:
            file_id: The ID of the file to delete.

        Returns:
            The deleted file details
        """
        cloud_drive = store().get("cloud_drive")
        assert isinstance(cloud_drive, CloudDrive)
        try:
            file = cloud_drive.delete_file(CloudDriveFileID(file_id))
        except ValueError as e:
            return str(e)
        return tool_result_to_str(file)

    return execute


@tool
def get_file_by_id() -> Tool:
    async def execute(file_id: str) -> str:
        """
        Get a file from a cloud drive by its ID.

        Args:
            file_id: The ID of the file to retrieve.

        Returns:
            The file details
        """
        cloud_drive = store().get("cloud_drive")
        assert isinstance(cloud_drive, CloudDrive)
        try:
            file = cloud_drive.get_file_by_id(CloudDriveFileID(file_id))
        except ValueError as e:
            return str(e)
        return tool_result_to_str(file)

    return execute


@tool
def list_files() -> Tool:
    async def execute() -> str:
        """
        Retrieve all files in the cloud drive.

        Args:
            None

        Returns:
            List of all files in the cloud drive
        """
        cloud_drive = store().get("cloud_drive")
        assert isinstance(cloud_drive, CloudDrive)
        files = list(cloud_drive.files.values())
        return tool_result_to_str(files)

    return execute


@tool
def share_file() -> Tool:
    async def execute(file_id: str, email: str, permission: str) -> str:
        """
        Share a file with a user.

        Args:
            file_id: The ID of the file to share.
            email: The email of the user to share the file with.
            permission: The permission level to grant the user (either "r" for read or "rw" for read-write).

        Returns:
            The updated file details
        """
        cloud_drive = store().get("cloud_drive")
        assert isinstance(cloud_drive, CloudDrive)
        try:
            file = cloud_drive.get_file_by_id(CloudDriveFileID(file_id))
        except ValueError as e:
            return str(e)
        file.shared_with[email] = SharingPermission(permission)
        return tool_result_to_str(file)

    return execute


@tool
def append_to_file() -> Tool:
    async def execute(file_id: str, content: str) -> str:
        """
        Append content to a file in the cloud drive.

        Args:
            file_id: The ID of the file to append content to.
            content: The content to append to the file.

        Returns:
            The updated file details
        """
        cloud_drive = store().get("cloud_drive")
        assert isinstance(cloud_drive, CloudDrive)
        try:
            file = cloud_drive.append_to_file(CloudDriveFileID(file_id), content)
        except ValueError as e:
            return str(e)
        return tool_result_to_str(file)

    return execute


@tool
def search_files() -> Tool:
    async def execute(query: str) -> str:
        """
        Search for files in the cloud drive by content.

        Args:
            query: The string to search for in the files.

        Returns:
            List of files containing the query in their content
        """
        cloud_drive = store().get("cloud_drive")
        assert isinstance(cloud_drive, CloudDrive)
        matches = cloud_drive.search_files(query)
        if len(matches) == 0:
            return f"No files found with the given query ('{query}'). Try a different search term."
        return tool_result_to_str(matches)

    return execute
