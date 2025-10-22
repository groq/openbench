from typing import TYPE_CHECKING, Any, Optional

from openbench.evals.agentdojo.base_tasks import TaskEnvironment
from openbench.evals.agentdojo.tools.calendar_client import Calendar
from openbench.evals.agentdojo.tools.email_client import Inbox
from openbench.evals.agentdojo.tools.travel_booking_client import (
    CarRental,
    Flights,
    Hotels,
    Reservation,
    Restaurants,
    User,
)

if TYPE_CHECKING:
    from deepdiff.diff import DeepDiff as DeepDiffType  # type: ignore[import-not-found]


class TravelEnvironment(TaskEnvironment):
    hotels: Hotels
    restaurants: Restaurants
    car_rental: CarRental
    flights: Flights
    user: User
    calendar: Calendar
    reservation: Reservation
    inbox: Inbox

    @classmethod
    def create(cls, dict: dict[str, Any]) -> "TravelEnvironment":
        if set(dict.keys()) != {
            "hotels",
            "restaurants",
            "car_rental",
            "flights",
            "user",
            "calendar",
            "reservation",
            "inbox",
        }:
            raise ValueError(
                "TravelEnvironment must have hotels, restaurants, car_rental, flights, user, calendar, reservation and inbox keys"
            )
        return cls(
            hotels=Hotels.model_validate(dict["hotels"]),
            restaurants=Restaurants.model_validate(dict["restaurants"]),
            car_rental=CarRental.model_validate(dict["car_rental"]),
            flights=Flights.model_validate(dict["flights"]),
            user=User.model_validate(dict["user"]),
            calendar=Calendar.create(dict["calendar"]),
            reservation=Reservation.model_validate(dict["reservation"]),
            inbox=Inbox.create(dict["inbox"]),
        )


class _TravelDeepDiff:
    """Lazy-loading callable for DeepDiff with travel-specific exclusions.

    This avoids importing DeepDiff at module level since it's an optional dependency.
    """

    _deepdiff_fn: Optional[type["DeepDiffType"]] = None

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

    def __init__(self) -> None:
        self._deepdiff_fn: Optional[type["DeepDiffType"]] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._deepdiff_fn is None:
            try:
                from deepdiff import DeepDiff  # type: ignore[import-not-found]

                self._deepdiff_fn = DeepDiff
            except ImportError:
                raise ImportError(
                    "DeepDiff is required for travel environment comparison. "
                    "Install it with: pip install deepdiff"
                )
        kwargs["exclude_paths"] = self._PATHS_TO_EXCLUDE
        return self._deepdiff_fn(*args, **kwargs)


TravelDeepDiff = _TravelDeepDiff()
