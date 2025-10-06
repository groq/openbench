from typing import Any

from openbench.evals.agentdojo.base_tasks import TaskEnvironment
from openbench.evals.agentdojo.tools.banking_client import BankAccount
from openbench.evals.agentdojo.tools.file_reader import Filesystem
from openbench.evals.agentdojo.tools.user_account import UserAccount


class BankingEnvironment(TaskEnvironment):
    bank_account: BankAccount
    filesystem: Filesystem
    user_account: UserAccount

    @classmethod
    def create(cls, dict: dict[str, Any]) -> "BankingEnvironment":
        return cls.model_validate(dict)
