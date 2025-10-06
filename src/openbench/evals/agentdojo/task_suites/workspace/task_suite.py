from openbench.evals.agentdojo.task_combinators import TaskCombinator
from openbench.evals.agentdojo.task_suite import BaseTaskSuite
from openbench.evals.agentdojo.task_suites.workspace.environment import (
    WorkspaceEnvironment,
)
from openbench.evals.agentdojo.task_suites.workspace.injection_tasks import (
    InjectionTask0,
    InjectionTask1,
    InjectionTask2,
    InjectionTask3,
    InjectionTask4,
    InjectionTask5,
    InjectionTask6,
    InjectionTask7,
    InjectionTask8,
    InjectionTask9,
    InjectionTask10,
    InjectionTask11,
    InjectionTask12,
    InjectionTask13,
)
from openbench.evals.agentdojo.task_suites.workspace.user_tasks import (
    UserTask0,
    UserTask1,
    UserTask2,
    UserTask3,
    UserTask5,
    UserTask6,
    UserTask7,
    UserTask8,
    UserTask9,
    UserTask10,
    UserTask11,
    UserTask12,
    UserTask13,
    UserTask14,
    UserTask15,
    UserTask16,
    UserTask17,
    UserTask18,
    UserTask20,
    UserTask21,
    UserTask22,
    UserTask24,
    UserTask25,
    UserTask26,
    UserTask27,
    UserTask28,
    UserTask29,
    UserTask30,
    UserTask31,
    UserTask32,
    UserTask33,
    UserTask34,
    UserTask35,
    WorkspaceUserTask,
)
from openbench.evals.agentdojo.tools.calendar_client import (
    add_calendar_event_participants,
    cancel_calendar_event,
    create_calendar_event,
    get_current_day,
    get_day_calendar_events,
    reschedule_calendar_event,
    search_calendar_events,
)
from openbench.evals.agentdojo.tools.cloud_drive_client import (
    append_to_file,
    create_file,
    delete_file,
    get_file_by_id,
    list_files,
    search_files,
    search_files_by_filename,
    share_file,
)
from openbench.evals.agentdojo.tools.email_client import (
    delete_email,
    get_draft_emails,
    get_received_emails,
    get_sent_emails,
    get_unread_emails,
    search_contacts_by_email,
    search_contacts_by_name,
    search_emails,
    send_email,
)

WorkspaceTaskSuite = BaseTaskSuite[WorkspaceEnvironment]

TOOLS = [
    # Email tools
    send_email(),
    delete_email(),
    get_unread_emails(),
    get_sent_emails(),
    get_received_emails(),
    get_draft_emails(),
    search_emails(),
    search_contacts_by_name(),
    search_contacts_by_email(),
    # Calendar tools
    get_current_day(),
    search_calendar_events(),
    get_day_calendar_events(),
    create_calendar_event(),
    cancel_calendar_event(),
    reschedule_calendar_event(),
    add_calendar_event_participants(),
    # Drive tools
    append_to_file(),
    search_files_by_filename(),
    create_file(),
    delete_file(),
    get_file_by_id(),
    list_files(),
    share_file(),
    search_files(),
]

task_combinator = TaskCombinator[WorkspaceEnvironment]()
UserTask23 = task_combinator.create_combined_task(
    "UserTask23", UserTask1(), UserTask22()
)

UserTask4 = task_combinator.create_combined_task("UserTask4", UserTask1(), UserTask6())

UserTask19 = task_combinator.create_combined_task(
    "UserTask19", UserTask1(), UserTask13()
)

UserTask36 = task_combinator.create_combined_task(
    "UserTask36", UserTask30(), UserTask31()
)

UserTask37 = task_combinator.create_combined_task(
    "UserTask37", UserTask30(), UserTask32()
)

UserTask38 = task_combinator.create_combined_task(
    "UserTask38", UserTask27(), UserTask35()
)

UserTask39 = task_combinator.create_combined_task(
    "UserTask39", UserTask16(), UserTask22()
)

USER_TASKS: list[type[WorkspaceUserTask]] = [
    UserTask0,
    UserTask1,
    UserTask2,
    UserTask3,
    UserTask4,
    UserTask5,
    UserTask6,
    UserTask7,
    UserTask8,
    UserTask9,
    UserTask10,
    UserTask11,
    UserTask12,
    UserTask13,
    UserTask14,
    UserTask15,
    UserTask16,
    UserTask17,
    UserTask18,
    UserTask19,
    UserTask20,
    UserTask21,
    UserTask22,
    UserTask23,
    UserTask24,
    UserTask25,
    UserTask26,
    UserTask27,
    UserTask28,
    UserTask29,
    UserTask30,
    UserTask31,
    UserTask32,
    UserTask33,
    UserTask34,
    UserTask35,
    UserTask36,
    UserTask37,
    UserTask38,
    UserTask39,
]

INJECTION_TASKS = [
    InjectionTask0,
    InjectionTask1,
    InjectionTask2,
    InjectionTask3,
    InjectionTask4,
    InjectionTask5,
    InjectionTask6,
    InjectionTask7,
    InjectionTask8,
    InjectionTask9,
    InjectionTask10,
    InjectionTask11,
    InjectionTask12,
    InjectionTask13,
]

WORKSPACE_TASK_SUITE = WorkspaceTaskSuite(
    name="workspace",
    environment_type=WorkspaceEnvironment,
    tools=TOOLS,
    user_tasks=USER_TASKS,
    injection_tasks=INJECTION_TASKS,
)
