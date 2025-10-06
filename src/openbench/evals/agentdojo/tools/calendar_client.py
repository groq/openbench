import datetime
from typing import Any, List, Optional

from inspect_ai.tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel, EmailStr

from openbench.evals.agentdojo.tools.email_client import Inbox
from openbench.evals.agentdojo.tools.types import (
    CalendarEvent,
    CalendarEventID,
    EventStatus,
)
from openbench.evals.agentdojo.utils import tool_result_to_str


class InitialCalendar(BaseModel):
    current_day: datetime.date
    account_email: EmailStr
    initial_events: list[CalendarEvent]


class Calendar(BaseModel):
    current_day: datetime.date
    account_email: EmailStr
    events: dict[CalendarEventID, CalendarEvent]

    @classmethod
    def create(cls, dict: dict[str, Any]) -> "Calendar":
        initial_calendar = InitialCalendar.model_validate(dict)
        current_day = initial_calendar.current_day
        account_email = initial_calendar.account_email
        initial_events = initial_calendar.initial_events

        all_ids = [event.id_ for event in initial_events]
        if len(all_ids) != len(set(all_ids)):
            duplicate_ids = [id_ for id_ in all_ids if all_ids.count(id_) > 1]
            raise ValueError(f"Event IDs must be unique. Duplicates: {duplicate_ids}")
        events = {event.id_: event for event in initial_events}
        return cls(
            current_day=current_day,
            account_email=account_email,
            events=events,
        )

    def _get_next_id(self) -> CalendarEventID:
        largest_id = max((int(key) for key in self.events.keys()), default=0)
        return CalendarEventID(largest_id + 1)

    def get_by_day(self, day: datetime.date) -> list[CalendarEvent]:
        return [
            event for event in self.events.values() if event.start_time.date() == day
        ]

    def create_event(
        self,
        title: str,
        description: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        location: str | None,
        participants: list[EmailStr],
    ) -> CalendarEvent:
        id_ = self._get_next_id()
        participants.append(self.account_email)
        participants = list(set(participants))
        event = CalendarEvent(
            title=title,
            description=description,
            start_time=start_time,
            end_time=end_time,
            location=location,
            participants=participants,
            id_=id_,
        )
        self.events[id_] = event
        return event

    def cancel_event(self, inbox: Inbox, event_id: CalendarEventID) -> None:
        if event_id not in self.events:
            raise ValueError(f"Event with ID '{event_id}' not found.")
        self.events[event_id].status = EventStatus.canceled
        inbox.send_email(
            self.events[event_id].participants,
            f"Canceled: '{self.events[event_id].title}'",
            "The event has been canceled.",
            attachments=[self.events[event_id]],
        )

    def search_events(
        self, query: str, date: datetime.date | None = None
    ) -> list[CalendarEvent]:
        if date is not None:
            events = self.get_by_day(date)
        else:
            events = list(self.events.values())
        return [
            event
            for event in events
            if query.lower() in event.title.lower()
            or query.lower() in event.description.lower()
        ]

    def day_has_conflicts(self, day: datetime.date) -> bool:
        events = self.get_by_day(day)
        for i, event in enumerate(events):
            for other_event in events[i + 1 :]:
                if (
                    event.start_time <= other_event.start_time < event.end_time
                    or event.start_time < other_event.end_time <= event.end_time
                ):
                    return True
        return False

    def reschedule_event(
        self,
        inbox: Inbox,
        event_id: CalendarEventID,
        new_start_time: datetime.datetime,
        new_end_time: datetime.datetime | None = None,
    ) -> CalendarEvent:
        if event_id not in self.events:
            raise ValueError(f"Event with ID '{event_id}' not found.")
        event = self.events[event_id]
        old_start_time = event.start_time
        event.start_time = new_start_time
        if new_end_time is not None:
            event.end_time = new_end_time
        else:
            event.end_time = new_start_time + (event.end_time - old_start_time)
        start_date = event.start_time.date().isoformat()
        start_time = event.start_time.time().isoformat()
        end_date = event.end_time.date().isoformat()
        end_time = event.end_time.time().isoformat()
        body = f"The event has been rescheduled. It will start on  {start_date} at {start_time} and end on {end_date} at {end_time}."
        inbox.send_email(
            event.participants,
            subject="Rescheduled: '{event.title}'",
            body=body,
            attachments=[event],
        )
        return event

    def add_participants(
        self, event_id: CalendarEventID, participants: list[EmailStr]
    ) -> CalendarEvent:
        if event_id not in self.events:
            raise ValueError(f"Event with ID '{event_id}' not found.")
        event = self.events[event_id]
        event.participants.extend(participants)
        return event


@tool
def get_day_calendar_events() -> Tool:
    async def execute(day: str) -> str:
        """
        Returns the appointments for the given day.

        Args:
            day: The day for which to return the appointments. Must be in format YYYY-MM-DD.

        Returns:
            List of the appointments for the given day
        """
        calendar = store().get("calendar")
        assert isinstance(calendar, Calendar)
        try:
            date = datetime.datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            return f"Error: Invalid date format '{day}'. Please use YYYY-MM-DD format."
        events = calendar.get_by_day(date.date())
        return tool_result_to_str(events)

    return execute


@tool
def create_calendar_event() -> Tool:
    async def execute(
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
        participants: Optional[List[str]] = None,
        location: Optional[str] = None,
    ) -> str:
        """
        Creates a new calendar event with the given details and adds it to the calendar.

        It also sends an email to the participants with the event details.

        Args:
            title: The title of the event.
            start_time: The start time of the event. Must be in format YYYY-MM-DD HH:MM.
            end_time: The end time of the event. Must be in format YYYY-MM-DD HH:MM.
            description: The description of the event. (default: "")
            participants: The list of participants' email addresses. If None, no participants are set. The calendar owner's email address is always included.
            location: The location of the event. If None, no location is set.

        Returns:
            The newly created calendar event
        """
        calendar = store().get("calendar")
        inbox = store().get("inbox")
        assert isinstance(calendar, Calendar)
        assert isinstance(inbox, Inbox)

        try:
            parsed_start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        except ValueError:
            return f"Error: Invalid start_time format '{start_time}'. Please use YYYY-MM-DD HH:MM format."

        try:
            parsed_end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M")
        except ValueError:
            return f"Error: Invalid end_time format '{end_time}'. Please use YYYY-MM-DD HH:MM format."

        if participants is None:
            participants = []

        new_event = calendar.create_event(
            title,
            description,
            parsed_start_time,
            parsed_end_time,
            location,
            participants,
        )
        inbox.send_email(
            recipients=participants,
            subject=f"Invitation: {title}",
            body=description,
            attachments=[new_event],
        )

        return tool_result_to_str(new_event)

    return execute


@tool
def search_calendar_events() -> Tool:
    async def execute(query: str, date: Optional[str] = None) -> str:
        """
        Searches calendar events that match the given query in the title or the description. If provided, filters events by date.

        Args:
            query: The query string to search for in event titles and descriptions.
            date: The date for which to search events. Must be in format YYYY-MM-DD. If None, searches all events.

        Returns:
            List of calendar events matching the search criteria
        """
        calendar = store().get("calendar")
        assert isinstance(calendar, Calendar)

        if date is not None and date != "None":
            try:
                date_ = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                return f"Error: Invalid date format '{date}'. Please use YYYY-MM-DD format."
        else:
            date_ = None

        matches = calendar.search_events(query, date_)
        if len(matches) == 0:
            return "No events found. Try with a different query."

        return tool_result_to_str(matches)

    return execute


@tool
def get_current_day() -> Tool:
    async def execute() -> str:
        """
        Returns the current day in ISO format, e.g. '2022-01-01'.

        It is useful to know what the current day, year, or month is, as the assistant
        should not assume what the current date is.

        Args:
            None

        Returns:
            The current day in ISO format
        """
        calendar = store().get("calendar")
        assert isinstance(calendar, Calendar)
        return calendar.current_day.isoformat()

    return execute


@tool
def cancel_calendar_event() -> Tool:
    async def execute(event_id: str) -> str:
        """
        Cancels the event with the given event_id. The event will be marked as canceled and no longer appear in the calendar.

        It will also send an email to the participants notifying them of the cancellation.

        Args:
            event_id: The ID of the event to cancel.

        Returns:
            A confirmation message
        """
        calendar = store().get("calendar")
        inbox = store().get("inbox")
        assert isinstance(calendar, Calendar)
        assert isinstance(inbox, Inbox)

        try:
            calendar.cancel_event(inbox, CalendarEventID(event_id))
        except ValueError as e:
            return str(e)
        return f"Event with ID {event_id} has been canceled and participants have been notified."

    return execute


@tool
def reschedule_calendar_event() -> Tool:
    async def execute(
        event_id: str, new_start_time: str, new_end_time: Optional[str] = None
    ) -> str:
        """
        Reschedules the event with the given event_id to the new start and end times.

        It will also send an email to the participants notifying them of the rescheduling.

        Args:
            event_id: The ID of the event to reschedule.
            new_start_time: The new start time of the event. Must be in format YYYY-MM-DD HH:MM.
            new_end_time: The new end time of the event. Must be in format YYYY-MM-DD HH:MM.
                          If None, the end time will be computed based on the new start time to keep the event duration the same.

        Returns:
            The rescheduled calendar event
        """
        calendar = store().get("calendar")
        inbox = store().get("inbox")
        assert isinstance(calendar, Calendar)
        assert isinstance(inbox, Inbox)

        try:
            parsed_new_start_time = datetime.datetime.strptime(
                new_start_time, "%Y-%m-%d %H:%M"
            )
        except ValueError:
            return f"Error: Invalid new_start_time format '{new_start_time}'. Please use YYYY-MM-DD HH:MM format."

        if new_end_time is not None:
            try:
                parsed_new_end_time = datetime.datetime.strptime(
                    new_end_time, "%Y-%m-%d %H:%M"
                )
            except ValueError:
                return f"Error: Invalid new_end_time format '{new_end_time}'. Please use YYYY-MM-DD HH:MM format."
        else:
            parsed_new_end_time = None

        try:
            calendar_event = calendar.reschedule_event(
                inbox,
                CalendarEventID(event_id),
                parsed_new_start_time,
                parsed_new_end_time,
            )
        except ValueError as e:
            return str(e)

        return tool_result_to_str(calendar_event)

    return execute


@tool
def add_calendar_event_participants() -> Tool:
    async def execute(event_id: str, participants: List[str]) -> str:
        """
        Adds the given participants to the event with the given event_id.

        It will also email the new participants notifying them of the event.

        Args:
            event_id: The ID of the event to add participants to.
            participants: The list of participants' email addresses to add to the event.

        Returns:
            The updated calendar event
        """
        calendar = store().get("calendar")
        assert isinstance(calendar, Calendar)

        try:
            calendar_event = calendar.add_participants(
                CalendarEventID(event_id), participants
            )
        except ValueError as e:
            return str(e)
        return tool_result_to_str(calendar_event)

    return execute
