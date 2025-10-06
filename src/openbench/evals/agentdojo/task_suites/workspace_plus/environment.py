from typing import Any

from openbench.evals.agentdojo.task_suites.workspace.environment import (
    WorkspaceEnvironment,
)
from openbench.evals.agentdojo.tools.calendar_client import Calendar
from openbench.evals.agentdojo.tools.cloud_drive_client import CloudDrive
from openbench.evals.agentdojo.tools.email_client import Inbox
from openbench.evals.agentdojo.tools.terminal import Terminal


class WorkspacePlusEnvironment(WorkspaceEnvironment):
    terminal: Terminal

    @classmethod
    def create(cls, dict: dict[str, Any]) -> "WorkspacePlusEnvironment":
        if set(dict.keys()) != {"terminal", "inbox", "calendar", "cloud_drive"}:
            raise ValueError(
                "WorkspacePlusEnvironment must have terminal, inbox, calendar, and cloud_drive keys"
            )
        return cls(
            terminal=Terminal.model_validate(dict["terminal"]),
            inbox=Inbox.create(dict["inbox"]),
            calendar=Calendar.create(dict["calendar"]),
            cloud_drive=CloudDrive.create(dict["cloud_drive"]),
        )
