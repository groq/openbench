from inspect_ai.tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel, Field

from openbench.evals.agentdojo.utils import tool_result_to_str


class UserAccount(BaseModel):
    """Represents a simple user account."""

    first_name: str = Field(..., title="First name of the user")
    last_name: str = Field(..., title="Last name of the user")
    street: str = Field(..., title="Street of the user")
    city: str = Field(..., title="City of the user")
    password: str = Field(..., title="Password of the user")


@tool
def get_user_info() -> Tool:
    async def execute() -> str:
        """
        Get the user information.

        Args:
            None

        Returns:
            User information including first name, last name, street, and city
        """
        account = store().get("user_account")
        assert isinstance(account, UserAccount)

        info = {
            "first_name": account.first_name,
            "last_name": account.last_name,
            "street": account.street,
            "city": account.city,
        }

        return tool_result_to_str(info)

    return execute


@tool
def update_password() -> Tool:
    async def execute(password: str) -> str:
        """
        Update the user password.

        Args:
            password: New password for the user

        Returns:
            Confirmation message
        """
        account = store().get("user_account")
        assert isinstance(account, UserAccount)

        account.password = password

        return tool_result_to_str({"message": "Password updated."})

    return execute


@tool
def update_user_info() -> Tool:
    async def execute(
        first_name: str | None = None,
        last_name: str | None = None,
        street: str | None = None,
        city: str | None = None,
    ) -> str:
        """
        Update the user information.

        Args:
            first_name: First name of the user (optional)
            last_name: Last name of the user (optional)
            street: Street of the user (optional)
            city: City of the user (optional)

        Returns:
            Updated user information
        """
        account = store().get("user_account")
        assert isinstance(account, UserAccount)

        if first_name:
            account.first_name = first_name
        if last_name:
            account.last_name = last_name
        if street:
            account.street = street
        if city:
            account.city = city

        info = {
            "first_name": account.first_name,
            "last_name": account.last_name,
            "street": account.street,
            "city": account.city,
        }

        return tool_result_to_str(info)

    return execute
