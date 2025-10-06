from openbench.evals.agentdojo.task_suite import BaseTaskSuite
from openbench.evals.agentdojo.task_suites.banking.environment import BankingEnvironment
from openbench.evals.agentdojo.task_suites.banking.injection_tasks import (
    InjectionTask0,
    InjectionTask1,
    InjectionTask2,
    InjectionTask3,
    InjectionTask4,
    InjectionTask5,
    InjectionTask6,
    InjectionTask7,
    InjectionTask8,
)
from openbench.evals.agentdojo.task_suites.banking.user_tasks import (
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
)
from openbench.evals.agentdojo.tools.banking_client import (
    get_balance,
    get_iban,
    get_most_recent_transactions,
    get_scheduled_transactions,
    schedule_transaction,
    send_money,
    update_scheduled_transaction,
)
from openbench.evals.agentdojo.tools.file_reader import read_file
from openbench.evals.agentdojo.tools.user_account import (
    get_user_info,
    update_password,
    update_user_info,
)

BankingTaskSuite = BaseTaskSuite[BankingEnvironment]

TOOLS = [
    get_balance(),
    get_iban(),
    get_most_recent_transactions(),
    get_scheduled_transactions(),
    schedule_transaction(),
    send_money(),
    update_scheduled_transaction(),
    read_file(),
    get_user_info(),
    update_password(),
    update_user_info(),
]

USER_TASKS = [
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
]

BANKING_TASK_SUITE = BankingTaskSuite(
    name="banking",
    environment_type=BankingEnvironment,
    tools=TOOLS,
    user_tasks=USER_TASKS,
    injection_tasks=INJECTION_TASKS,
)
