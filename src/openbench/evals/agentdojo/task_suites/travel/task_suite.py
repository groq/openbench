from openbench.evals.agentdojo.task_suite import BaseTaskSuite
from openbench.evals.agentdojo.task_suites.travel.environment import TravelEnvironment
from openbench.evals.agentdojo.task_suites.travel.injection_tasks import (
    InjectionTask0,
    InjectionTask1,
    InjectionTask2,
    InjectionTask3,
    InjectionTask4,
    InjectionTask5,
    InjectionTask6,
)
from openbench.evals.agentdojo.task_suites.travel.user_tasks import (
    TravelUserTask,
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
)
from openbench.evals.agentdojo.tools.calendar_client import (
    cancel_calendar_event,
    create_calendar_event,
    get_day_calendar_events,
    search_calendar_events,
)
from openbench.evals.agentdojo.tools.email_client import send_email
from openbench.evals.agentdojo.tools.travel_booking_client import (
    check_restaurant_opening_hours,
    get_all_car_rental_companies_in_city,
    get_all_hotels_in_city,
    get_all_restaurants_in_city,
    get_car_fuel_options,
    get_car_price_per_day,
    get_car_rental_address,
    get_car_types_available,
    get_contact_information_for_restaurants,
    get_cuisine_type_for_restaurants,
    get_dietary_restrictions_for_all_restaurants,
    get_flight_information,
    get_hotels_address,
    get_hotels_prices,
    get_price_for_restaurants,
    get_rating_reviews_for_car_rental,
    get_rating_reviews_for_hotels,
    get_rating_reviews_for_restaurants,
    get_restaurants_address,
    get_user_information,
    reserve_car_rental,
    reserve_hotel,
    reserve_restaurant,
)

TravelTaskSuite = BaseTaskSuite[TravelEnvironment]

TOOLS = [
    # User functions
    get_user_information(),
    # Hotel functions
    get_all_hotels_in_city(),
    get_hotels_prices(),
    get_rating_reviews_for_hotels(),
    get_hotels_address(),
    # Restaurant functions
    get_all_restaurants_in_city(),
    get_cuisine_type_for_restaurants(),
    get_restaurants_address(),
    get_rating_reviews_for_restaurants(),  # get_rating_reviews_for_restaurants
    get_dietary_restrictions_for_all_restaurants(),
    get_contact_information_for_restaurants(),
    get_price_for_restaurants(),
    check_restaurant_opening_hours(),
    # Car rental functions
    get_all_car_rental_companies_in_city(),
    get_car_types_available(),
    get_rating_reviews_for_car_rental(),
    get_car_fuel_options(),
    get_car_rental_address(),
    get_car_price_per_day(),
    # Calendar functions
    create_calendar_event(),
    search_calendar_events(),
    get_day_calendar_events(),
    cancel_calendar_event(),
    # For making a reservation
    reserve_hotel(),
    reserve_car_rental(),
    reserve_restaurant(),
    # For getting flights
    get_flight_information(),
    # For sending emails
    send_email(),
]

USER_TASKS: list[type[TravelUserTask]] = [
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
]

INJECTION_TASKS = [
    InjectionTask0,
    InjectionTask1,
    InjectionTask2,
    InjectionTask3,
    InjectionTask4,
    InjectionTask5,
    InjectionTask6,
]

TRAVEL_TASK_SUITE = TravelTaskSuite(
    name="travel",
    environment_type=TravelEnvironment,
    tools=TOOLS,
    user_tasks=USER_TASKS,
    injection_tasks=INJECTION_TASKS,
)
