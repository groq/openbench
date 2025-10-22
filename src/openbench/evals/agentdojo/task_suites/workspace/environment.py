from typing import TYPE_CHECKING, Any, Optional

from openbench.evals.agentdojo.base_tasks import TaskEnvironment
from openbench.evals.agentdojo.tools.calendar_client import Calendar
from openbench.evals.agentdojo.tools.cloud_drive_client import CloudDrive
from openbench.evals.agentdojo.tools.email_client import Inbox

if TYPE_CHECKING:
    from deepdiff.diff import DeepDiff as DeepDiffType  # type: ignore[import-not-found]


class WorkspaceEnvironment(TaskEnvironment):
    inbox: Inbox
    calendar: Calendar
    cloud_drive: CloudDrive

    @classmethod
    def create(cls, dict: dict[str, Any]) -> "WorkspaceEnvironment":
        if set(dict.keys()) != {"inbox", "calendar", "cloud_drive"}:
            raise ValueError(
                "WorkspaceEnvironment must have inbox, calendar, and cloud_drive keys"
            )
        return cls(
            inbox=Inbox.create(dict["inbox"]),
            calendar=Calendar.create(dict["calendar"]),
            cloud_drive=CloudDrive.create(dict["cloud_drive"]),
        )


class _WorkspaceDeepDiff:
    """Lazy-loading callable for DeepDiff with workspace-specific exclusions.

    This avoids importing DeepDiff at module level since it's an optional dependency.
    """

    _PATHS_TO_EXCLUDE = [
        "root.inbox.sent",
        "root.inbox.received",
        "root.inbox.drafts",
        "root.responses",
        "root.model_fields_set",
        "root.calendar.initial_events",
        "root.inbox.initial_emails",
        "root.cloud_drive.initial_files",
    ]

    _deepdiff_fn: Optional[type["DeepDiffType"]] = None

    def __init__(self) -> None:
        self._deepdiff_fn: Optional[type["DeepDiffType"]] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._deepdiff_fn is None:
            try:
                from deepdiff import DeepDiff  # type: ignore[import-not-found]

                self._deepdiff_fn = DeepDiff
            except ImportError:
                raise ImportError(
                    "DeepDiff is required for workspace environment comparison. "
                    "Install it with: pip install deepdiff"
                )
        kwargs["exclude_paths"] = self._PATHS_TO_EXCLUDE
        return self._deepdiff_fn(*args, **kwargs)


WorkspaceDeepDiff = _WorkspaceDeepDiff()
