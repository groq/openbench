from inspect_ai.tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel


class Web(BaseModel):
    web_content: dict[str, str]
    web_requests: list[str]


def standardize_url(url: str) -> str:
    url = url.removeprefix("https://")
    url = url.removeprefix("http://")
    return url


@tool
def post_webpage() -> Tool:
    async def execute(url: str, content: str) -> str:
        """
        Posts a webpage at a given URL with the given content.

        Args:
            url: The URL of the webpage.
            content: The content of the webpage.

        Returns:
            Confirmation message
        """
        web = store().get("web")
        assert isinstance(web, Web)

        url = standardize_url(url)
        web.web_requests.append(url)
        web.web_content[url] = content

        return f"Webpage posted at {url}"

    return execute


@tool
def get_webpage() -> Tool:
    async def execute(url: str) -> str:
        """
        Returns the content of the webpage at a given URL.

        Args:
            url: The URL of the webpage.

        Returns:
            The content of the webpage or a 404 error
        """
        web = store().get("web")
        assert isinstance(web, Web)

        url = standardize_url(url)
        web.web_requests.append(url)

        if url not in web.web_content:
            return "404 Not Found"

        return web.web_content.get(url, "404 Not Found")

    return execute


@tool
def download_file() -> Tool:
    async def execute(url: str, file_name: str) -> str:
        """
        Downloads a file from a given URL to the local folder.

        Args:
            url: The URL of the file.
            file_name: The name of the file to download.

        Returns:
            Success message or 404 error
        """
        web = store().get("web")
        assert isinstance(web, Web)

        url = standardize_url(url)
        web.web_requests.append(url)

        if url not in web.web_content:
            return "404 Not Found"

        return f"File {file_name} successfully downloaded from {url}"

    return execute
