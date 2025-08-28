"""
Dynamic data fetching for LiveMCPBench tools.

"""

import json
import logging
from typing import Dict, List, Any, Optional
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


class LiveMCPDataFetcher:
    """
    Fetches LiveMCPBench tools data from the upstream repository.

    """

    def __init__(
        self,
        upstream_base_url: str = "https://raw.githubusercontent.com/icip-cas/LiveMCPBench/main/tools/LiveMCPTool",
    ):
        """
        Initialize the data fetcher.

        Args:
            upstream_base_url: Base URL for fetching upstream data
        """
        self.upstream_base_url = upstream_base_url.rstrip("/")

    def get_tools_data(self) -> List[Dict[str, Any]]:
        """Get tools data from upstream repository."""
        logger.debug("Fetching tools.json from upstream")
        return self._fetch_upstream_data("tools.json")

    def get_config_data(self) -> List[Dict[str, Any]]:
        """Get config data from upstream repository."""
        logger.debug("Fetching all_config.json from upstream")
        return self._fetch_upstream_data("all_config.json")

    def _fetch_upstream_data(self, filename: str) -> List[Dict[str, Any]]:
        """Fetch data from upstream repository."""
        url = f"{self.upstream_base_url}/{filename}"

        try:
            with urlopen(url, timeout=30) as response:
                if response.status != 200:
                    raise HTTPError(
                        url,
                        response.status,
                        f"HTTP {response.status}",
                        response.headers,
                        None,
                    )

                content = response.read().decode("utf-8")
                return json.loads(content)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in upstream {filename}: {e}")
        except (URLError, HTTPError) as e:
            logger.error(f"Failed to fetch {filename} from {url}: {e}")
            raise RuntimeError(f"Unable to fetch {filename}: {e}")


# Global instance for convenient access
_default_fetcher: Optional[LiveMCPDataFetcher] = None


def get_default_fetcher() -> LiveMCPDataFetcher:
    """Get the default global fetcher instance."""
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = LiveMCPDataFetcher()
    return _default_fetcher


def get_tools_data() -> List[Dict[str, Any]]:
    """
    Get LiveMCPBench tools data.

    Returns:
        List of tool configurations fetched fresh from upstream
    """
    return get_default_fetcher().get_tools_data()


def get_config_data() -> List[Dict[str, Any]]:
    """
    Get LiveMCPBench config data.

    Returns:
        List of server configurations fetched fresh from upstream
    """
    return get_default_fetcher().get_config_data()
