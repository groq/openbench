from typing import Optional

from inspect_ai.tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel, Field

from openbench.evals.agentdojo.utils import tool_result_to_str


class Transaction(BaseModel):
    id: int = Field(..., title="ID of the transaction")
    sender: str = Field(..., title="IBAN of the sender")
    recipient: str = Field(..., title="IBAN of the recipient")

    amount: float = Field(..., title="Amount of the transaction")
    subject: str = Field(..., title="Subject of the transaction")

    date: str = Field(..., title="Date of the transaction")
    recurring: bool = Field(..., title="Is the transaction recurring")


class BankAccount(BaseModel):
    balance: float
    iban: str
    transactions: list[Transaction]
    scheduled_transactions: list[Transaction]


def _calculate_next_id() -> int:
    account = store().get("bank_account")
    assert isinstance(account, BankAccount)
    return (
        max(
            [t.id for t in account.transactions]
            + [t.id for t in account.scheduled_transactions],
            default=0,
        )
        + 1
    )


@tool
def next_id() -> Tool:
    async def execute() -> int:
        """
        Get the next ID for a transaction.

        Args:
            None

        Returns:
            The next ID for a transaction
        """
        return _calculate_next_id()

    return execute


@tool
def get_iban() -> Tool:
    async def execute() -> str:
        """
        Get the IBAN of the current bank account.

        Args:
            None

        Returns:
            The IBAN of the current bank account
        """
        account = store().get("bank_account")
        assert isinstance(account, BankAccount)
        return account.iban

    return execute


@tool
def send_money() -> Tool:
    async def execute(recipient: str, amount: float, subject: str, date: str) -> str:
        """
        Sends a transaction to the recipient.

        Args:
            recipient: IBAN of the recipient
            amount: Amount of the transaction
            subject: Subject of the transaction
            date: Date of the transaction

        Returns:
            A dictionary with a confirmation message
        """
        account = store().get("bank_account")
        assert isinstance(account, BankAccount)

        transaction = Transaction(
            id=_calculate_next_id(),
            sender=account.iban,
            recipient=recipient,
            amount=amount,
            subject=subject,
            date=date,
            recurring=False,
        )
        account.transactions.append(transaction)

        return f"Transaction to {recipient} for {amount} sent."

    return execute


@tool
def schedule_transaction() -> Tool:
    async def execute(
        recipient: str, amount: float, subject: str, date: str, recurring: bool
    ) -> str:
        """
        Schedule a transaction.

        Args:
            recipient: IBAN of the recipient
            amount: Amount of the transaction
            subject: Subject of the transaction
            date: Next date of the transaction
            recurring: Is the transaction recurring

        Returns:
            A dictionary with a confirmation message
        """
        account = store().get("bank_account")
        assert isinstance(account, BankAccount)

        transaction = Transaction(
            id=_calculate_next_id(),
            sender=account.iban,
            recipient=recipient,
            amount=amount,
            subject=subject,
            date=date,
            recurring=recurring,
        )
        account.scheduled_transactions.append(transaction)

        return f"Transaction to {recipient} for {amount} scheduled."

    return execute


@tool
def update_scheduled_transaction() -> Tool:
    async def execute(
        id: int,
        recipient: Optional[str] = None,
        amount: Optional[float] = None,
        subject: Optional[str] = None,
        date: Optional[str] = None,
        recurring: Optional[bool] = None,
    ) -> str:
        """
        Update a scheduled transaction.

        Args:
            id: ID of the transaction (mandatory)
            recipient: IBAN of the recipient (optional)
            amount: Amount of the transaction (optional)
            subject: Subject of the transaction (optional)
            date: Next date of the transaction (optional)
            recurring: Is the transaction recurring (optional)

        Returns:
            A dictionary with a confirmation message
        """
        account = store().get("bank_account")
        assert isinstance(account, BankAccount)

        transaction = next(
            (t for t in account.scheduled_transactions if t.id == id), None
        )
        if transaction:
            if recipient:
                transaction.recipient = recipient
            if amount is not None:
                transaction.amount = amount
            if subject:
                transaction.subject = subject
            if date:
                transaction.date = date
            if recurring is not None:
                transaction.recurring = recurring
        else:
            return f"Transaction with ID {id} not found."

        return f"Transaction with ID {id} updated."

    return execute


@tool
def get_balance() -> Tool:
    async def execute() -> float:
        """
        Get the balance of the account.

        Args:
            None

        Returns:
            The current balance of the account
        """
        account = store().get("bank_account")
        assert isinstance(account, BankAccount)
        return account.balance

    return execute


@tool
def get_most_recent_transactions() -> Tool:
    async def execute(n: int = 100) -> str:
        """
        Get the list of the most recent transactions, e.g. to summarize the last n transactions.

        Args:
            n: Number of transactions to return (default: 100)

        Returns:
            List of the most recent transactions
        """
        account = store().get("bank_account")
        assert isinstance(account, BankAccount)
        return tool_result_to_str([t for t in account.transactions[-int(n) :]])

    return execute


@tool
def get_scheduled_transactions() -> Tool:
    async def execute() -> str:
        """
        Get the list of scheduled transactions.

        Args:
            None

        Returns:
            List of scheduled transactions
        """
        account = store().get("bank_account")
        assert isinstance(account, BankAccount)
        return tool_result_to_str([t for t in account.scheduled_transactions])

    return execute
