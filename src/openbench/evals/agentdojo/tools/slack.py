from inspect_ai.tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel, Field

from openbench.evals.agentdojo.utils import tool_result_to_str


class Message(BaseModel):
    sender: str = Field(..., title="Sender of the message")
    recipient: str = Field(
        ..., title="Recipient of the message (either user or a channel)"
    )
    body: str = Field(..., title="Body of the message")


class Slack(BaseModel):
    users: list[str] = Field(..., title="List of users in the slack")
    channels: list[str] = Field(..., title="List of channels in the slack")
    user_channels: dict[str, list[str]] = Field(
        ..., title="Channels each user is a member of"
    )
    user_inbox: dict[str, list[Message]] = Field(..., title="Inbox of each user")
    channel_inbox: dict[str, list[Message]] = Field(..., title="Inbox of each channel")


@tool
def get_channels() -> Tool:
    async def execute() -> str:
        """
        Get the list of channels in the slack.

        Args:
            None

        Returns:
            List of channels in the slack
        """
        slack = store().get("slack")
        assert isinstance(slack, Slack)
        return tool_result_to_str(slack.channels)

    return execute


@tool
def add_user_to_channel() -> Tool:
    async def execute(user: str, channel: str) -> str:
        """
        Add a user to a given channel.

        Args:
            user: The user to add to the channel.
            channel: The channel to add the user to.

        Returns:
            Confirmation message
        """
        slack = store().get("slack")
        assert isinstance(slack, Slack)

        if user not in slack.users:
            return f"User {user} not found in the users list"
        if channel not in slack.channels:
            return f"Channel {channel} not found in the channels list"

        slack.user_channels[user].append(channel)
        return f"User {user} added to channel {channel}"

    return execute


@tool
def read_channel_messages() -> Tool:
    async def execute(channel: str) -> str:
        """
        Read the messages from the given channel.

        Args:
            channel: The channel to read the messages from.

        Returns:
            List of messages in the channel
        """
        slack = store().get("slack")
        assert isinstance(slack, Slack)

        if channel not in slack.channels:
            return "Channel does not exist!"

        return tool_result_to_str(slack.channel_inbox.get(channel, []))

    return execute


@tool
def read_inbox() -> Tool:
    async def execute(user: str) -> str:
        """
        Read the messages from the given user inbox.

        Args:
            user: The user whose inbox to read.

        Returns:
            List of messages in the user's inbox
        """
        slack = store().get("slack")
        assert isinstance(slack, Slack)

        if user not in slack.users:
            return f"User {user} not found in the users list"

        return tool_result_to_str(slack.user_inbox.get(user, []))

    return execute


@tool
def send_direct_message() -> Tool:
    async def execute(recipient: str, body: str) -> str:
        """
        Send a direct message from bot to recipient with the given body.

        Args:
            recipient: The recipient of the message.
            body: The body of the message.

        Returns:
            Confirmation message
        """
        slack = store().get("slack")
        assert isinstance(slack, Slack)

        sender = "bot"
        if recipient not in slack.users:
            return f"Recipient {recipient} not found in the users list"

        msg = Message(sender=sender, recipient=recipient, body=body)
        if recipient not in slack.user_inbox:
            slack.user_inbox[recipient] = []

        slack.user_inbox[recipient].append(msg)
        return f"Direct message sent to {recipient}"

    return execute


@tool
def send_channel_message() -> Tool:
    async def execute(channel: str, body: str) -> str:
        """
        Send a channel message from bot to channel with the given body.

        Args:
            channel: The channel to send the message to.
            body: The body of the message.

        Returns:
            Confirmation message
        """
        slack = store().get("slack")
        assert isinstance(slack, Slack)

        sender = "bot"
        if channel not in slack.channels:
            return f"Channel {channel} not found in the channels list"

        msg = Message(sender=sender, recipient=channel, body=body)
        if channel not in slack.channel_inbox:
            slack.channel_inbox[channel] = []

        slack.channel_inbox[channel].append(msg)
        return f"Channel message sent to {channel}"

    return execute


@tool
def invite_user_to_slack() -> Tool:
    async def execute(user: str, user_email: str) -> str:
        """
        Invites a user to the Slack workspace.

        Args:
            user: The user to invite.
            user_email: The user email where invite should be sent.

        Returns:
            Confirmation message
        """
        slack = store().get("slack")
        assert isinstance(slack, Slack)

        if user in slack.users:
            return f"User {user} already in the users list"

        slack.users.append(user)
        slack.user_inbox[user] = []
        slack.user_channels[user] = []

        return f"User {user} invited to Slack with email {user_email}"

    return execute


@tool
def remove_user_from_slack() -> Tool:
    async def execute(user: str) -> str:
        """
        Remove a user from the Slack workspace.

        Args:
            user: The user to remove.

        Returns:
            Confirmation message
        """
        slack = store().get("slack")
        assert isinstance(slack, Slack)

        if user not in slack.users:
            return f"User {user} not found in the users list"

        slack.users.remove(user)
        del slack.user_inbox[user]
        del slack.user_channels[user]

        return f"User {user} removed from Slack"

    return execute


@tool
def get_users_in_channel() -> Tool:
    async def execute(channel: str) -> str:
        """
        Get the list of users in the given channel.

        Args:
            channel: The channel to get the users from.

        Returns:
            List of users in the channel
        """
        slack = store().get("slack")
        assert isinstance(slack, Slack)

        if channel not in slack.channels:
            return f"Channel {channel} not found in the channels list"

        users = []
        for user, channels in slack.user_channels.items():
            if channel in channels:
                users.append(user)

        return tool_result_to_str(users)

    return execute
