from typing import Any

from openbench.evals.agentdojo.base_tasks import TaskEnvironment
from openbench.evals.agentdojo.tools.slack import Slack
from openbench.evals.agentdojo.tools.web import Web


class SlackEnvironment(TaskEnvironment):
    slack: Slack
    web: Web

    @classmethod
    def create(cls, dict: dict[str, Any]) -> "SlackEnvironment":
        return cls.model_validate(dict)
