import datetime
from typing import Any, Dict, List, Optional

from inspect_ai.tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel, EmailStr, computed_field

from openbench.evals.agentdojo.tools.types import (
    CalendarEvent,
    CloudDriveFileID,
    Email,
    EmailID,
    EmailStatus,
)
from openbench.evals.agentdojo.utils import tool_result_to_str


class EmailContact(BaseModel):
    email: EmailStr
    name: str


class InitialInbox(BaseModel):
    account_email: EmailStr
    initial_emails: List[Email]


class Inbox(BaseModel):
    account_email: EmailStr
    emails: dict[EmailID, Email]
    contact_list: list[EmailContact]
    trash: dict[EmailID, Email] = {}

    @classmethod
    def create(cls, dict: dict[str, Any]) -> "Inbox":
        initial_inbox = InitialInbox.model_validate(dict)
        initial_emails = initial_inbox.initial_emails
        account_email = initial_inbox.account_email

        all_ids = [email.id_ for email in initial_emails]
        if len(all_ids) != len(set(all_ids)):
            duplicate_ids = [id_ for id_ in all_ids if all_ids.count(id_) > 1]
            raise ValueError(f"Email IDs must be unique. Duplicates: {duplicate_ids}")
        emails = {email.id_: email for email in initial_emails}

        existing_contacts = set()
        contact_list: list[EmailContact] = []
        for email in initial_emails:
            email_contacts = [*email.recipients, email.sender, *email.cc, *email.bcc]
            for recipient in email_contacts:
                if recipient in existing_contacts:
                    continue
                if recipient not in [contact.email for contact in contact_list]:
                    name = recipient.split("@")[0]
                    name = name.replace(".", " ").title()
                    new_contact = EmailContact(email=recipient, name=name)
                    contact_list.append(new_contact)
                    existing_contacts.add(recipient)

        return cls(
            account_email=account_email,
            emails=emails,
            contact_list=contact_list,
        )

    def from_address(self, address: str) -> list[Email]:
        return [
            email
            for email in self.emails.values()
            if email.sender.lower() == address.lower()
        ]

    def _get_next_id(self) -> EmailID:
        largest_id = max((int(key) for key in self.emails.keys()), default=0)
        return EmailID(largest_id + 1)

    def _get_emails_by_status(self, status: EmailStatus) -> list[Email]:
        return [email for email in self.emails.values() if email.status == status]

    @computed_field  # type: ignore
    @property
    def received(self) -> list[Email]:
        return self._get_emails_by_status(EmailStatus.received)

    @computed_field  # type: ignore
    @property
    def sent(self) -> list[Email]:
        return self._get_emails_by_status(EmailStatus.sent)

    @computed_field  # type: ignore
    @property
    def drafts(self) -> list[Email]:
        return self._get_emails_by_status(EmailStatus.draft)

    def send_email(
        self,
        recipients: list[EmailStr],
        subject: str,
        body: str,
        attachments: list[CloudDriveFileID | CalendarEvent] | None = None,
        cc: list[EmailStr] | None = None,
        bcc: list[EmailStr] | None = None,
    ) -> Email:
        if cc is None:
            cc = []
        if bcc is None:
            bcc = []
        if attachments is None:
            attachments = []
        new_email = Email(
            id_=self._get_next_id(),
            sender=self.account_email,
            body=body,
            subject=subject,
            recipients=recipients,
            status=EmailStatus.sent,
            timestamp=datetime.datetime.now(),
            cc=cc,
            bcc=bcc,
            attachments=attachments,
            read=True,
        )
        self.emails[new_email.id_] = new_email
        return new_email

    def search_emails(self, query: str, sender: EmailStr | None = None) -> list[Email]:
        if sender is not None:
            emails = self.from_address(sender)
        else:
            emails = list(self.emails.values())
        return [
            email
            for email in emails
            if query.lower() in email.subject.lower()
            or query.lower() in email.body.lower()
        ]

    def delete_email(self, email_id: EmailID) -> Email:
        if email_id not in self.emails:
            raise ValueError(f"Email with ID '{email_id}' not found.")
        email = self.emails.pop(email_id)
        self.trash[email_id] = email
        return email

    def _list_unread(self) -> list[Email]:
        return [email for email in self.emails.values() if not email.read]

    def get_unread(self) -> list[Email]:
        unread_emails = self._list_unread()
        for email in unread_emails:
            email.read = True
        return unread_emails

    def find_contacts_by_name(self, name: str) -> list[EmailContact]:
        matches = []
        for contact in self.contact_list:
            if name.lower() in contact.name.lower():
                matches.append(contact)
        if len(matches) == 0:
            raise ValueError(f"Contact with name '{name}' not found.")
        return matches

    def find_contacts_by_email(self, email: EmailStr) -> list[EmailContact]:
        matches = []
        for contact in self.contact_list:
            if email.lower() in contact.email.lower():
                matches.append(contact)
        if not matches:
            raise ValueError(f"Contact with email '{email}' not found.")
        return matches

    def get_most_recently_received_email(self) -> Email:
        return max(self.received, key=lambda email: email.timestamp)


@tool
def get_unread_emails() -> Tool:
    async def execute() -> str:
        """
        Returns all the unread emails in the inbox. Each email has a sender, a subject, and a body.

        The emails are marked as read after this function is called.

        Args:
            None

        Returns:
            List of unread emails
        """
        inbox = store().get("inbox")
        assert isinstance(inbox, Inbox)
        return tool_result_to_str(inbox.get_unread())

    return execute


@tool
def send_email() -> Tool:
    async def execute(
        recipients: List[str],
        subject: str,
        body: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
    ) -> str:
        """
        Sends an email with the given body to the given address.

        Args:
            recipients: The list with the email addresses of the recipients.
            subject: The subject of the email.
            body: The body of the email.
            attachments: The list of attachments to include in the email. If None, no attachments are included.
                If the attachment has as "type" "file", then it is expected to have a field "file_id", with the ID of the file in the
                cloud drive. If the attachment has as "type" "event", then it is expected to be a calendar event in the field "event_details".
                A calendar event has the following fields: `title`, `description`, `start_time` (in ISO format), `end_time`
                (in ISO format), `location`, and participants (a list of emails).
            cc: The list of email addresses to include in the CC field. If None, no email addresses are included.
            bcc: The list of email addresses to include in the BCC field. If None, no email addresses are included.

        Returns:
            The details of the sent email
        """
        inbox = store().get("inbox")
        assert isinstance(inbox, Inbox)

        if attachments is not None:
            parsed_attachments: list[CloudDriveFileID | CalendarEvent] = []
            for attachment in attachments:
                if not isinstance(attachment, dict):
                    return "Attachments must be dictionaries."
                if "type" not in attachment:
                    return "Attachment must have a 'type' field."
                if attachment["type"] == "file" or "file_id" in attachment:
                    if "file_id" not in attachment:
                        return "Attachment of type 'file' must have a 'file_id' field."
                    parsed_attachments.append(CloudDriveFileID(attachment["file_id"]))
                else:
                    if "event_details" not in attachment:
                        return "Attachment of type 'event' must have an 'event_details' field."
                    parsed_attachments.append(
                        CalendarEvent(**attachment["event_details"])
                    )

        email_attachments = parsed_attachments if attachments is not None else None

        email = inbox.send_email(recipients, subject, body, email_attachments, cc, bcc)
        return tool_result_to_str(email)

    return execute


@tool
def search_emails() -> Tool:
    async def execute(
        query: str,
        sender: Optional[str] = None,
    ) -> str:
        """
        Searches for emails in the inbox that contain the given query in the subject or body.

        If sender is provided, only emails from that address are searched.

        Args:
            query: The query to search for in the email subject or body. If empty, all emails are returned.
            sender: The email address of the sender. If None, all emails are searched. Use search_contacts_by_name
                   if the email address is unknown.

        Returns:
            List of matching emails
        """
        inbox = store().get("inbox")
        assert isinstance(inbox, Inbox)

        matches = inbox.search_emails(query, sender)
        if len(matches) == 0:
            return "No emails found. Try with a different query."
        return tool_result_to_str(matches)

    return execute


@tool
def delete_email() -> Tool:
    async def execute(email_id: str) -> str:
        """
        Deletes the email with the given email_id from the inbox.

        Args:
            email_id: The id of the email to delete.

        Returns:
            A confirmation message
        """
        inbox = store().get("inbox")
        assert isinstance(inbox, Inbox)

        try:
            inbox.delete_email(email_id)
        except ValueError as e:
            return str(e)
        return f"Email with id {email_id} deleted successfully."

    return execute


@tool
def get_sent_emails() -> Tool:
    async def execute() -> str:
        """
        Returns all the sent emails in the inbox. Each email has a recipient, a subject, and a body.

        Args:
            None

        Returns:
            List of sent emails
        """
        inbox = store().get("inbox")
        assert isinstance(inbox, Inbox)
        return tool_result_to_str(inbox.sent)

    return execute


@tool
def get_received_emails() -> Tool:
    async def execute() -> str:
        """
        Returns all the received emails in the inbox. Each email has a sender, a subject, and a body.

        Args:
            None

        Returns:
            List of received emails
        """
        inbox = store().get("inbox")
        assert isinstance(inbox, Inbox)
        return tool_result_to_str(inbox.received)

    return execute


@tool
def get_draft_emails() -> Tool:
    async def execute() -> str:
        """
        Returns all the draft emails in the inbox. Each email has a recipient, a subject, and a body.

        Args:
            None

        Returns:
            List of draft emails
        """
        inbox = store().get("inbox")
        assert isinstance(inbox, Inbox)
        return tool_result_to_str(inbox.drafts)

    return execute


@tool
def search_contacts_by_name() -> Tool:
    async def execute(query: str) -> str:
        """
        Finds contacts in the inbox's contact list by name.

        It returns a list of contacts that match the given name.

        Args:
            query: The name of the contacts to search for.

        Returns:
            List of matching contacts
        """
        inbox = store().get("inbox")
        assert isinstance(inbox, Inbox)
        try:
            contact_list = inbox.find_contacts_by_name(query)
        except ValueError as e:
            return str(e)
        return tool_result_to_str(contact_list)

    return execute


@tool
def search_contacts_by_email() -> Tool:
    async def execute(query: str) -> str:
        """
        Finds contacts in the inbox's contact list by email.

        It returns a list of contacts that match the given email.

        Args:
            query: The email of the contacts to search for.

        Returns:
            List of matching contacts
        """
        inbox = store().get("inbox")
        assert isinstance(inbox, Inbox)
        try:
            contact_list = inbox.find_contacts_by_email(query)
        except ValueError as e:
            return str(e)
        return tool_result_to_str(contact_list)

    return execute
