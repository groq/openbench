import datetime
from typing import List, Optional

from inspect_ai.tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel

from openbench.evals.agentdojo.strenum import StrEnum
from openbench.evals.agentdojo.utils import tool_result_to_str


class ReservationType(StrEnum):
    HOTEL = "hotel"
    CAR = "car"
    RESTAURANT = "restaurant"


class User(BaseModel):
    first_name: str
    last_name: str
    ID_number: str
    email: str
    phone_number: str
    address: str
    passport_number: str
    bank_account_number: str
    credit_card_number: str


class Hotel(BaseModel):
    name: str
    city: str
    rating: float
    price_min: float
    price_max: float
    address: str
    reviews: list[str]


class Hotels(BaseModel):
    hotel_list: list[Hotel]

    def get_all_hotels_in_city(self, city: str) -> list[Hotel]:
        return [hotel for hotel in self.hotel_list if hotel.city == city]


class Flight(BaseModel):
    airline: str
    flight_number: str
    departure_city: str
    arrival_city: str
    departure_time: datetime.datetime
    arrival_time: datetime.datetime
    price: float
    contact_information: str


class Flights(BaseModel):
    flight_list: list[Flight]


class Reservation(BaseModel):
    reservation_type: ReservationType | None
    title: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    contact_information: str


class Restaurant(BaseModel):
    name: str
    city: str
    address: str
    rating: float
    price_per_person: float
    operating_hours: str
    reviews: list[str]
    contact_information: str
    cuisine_type: str  # could be: Italian, Chinese, Indian, Japanese
    dietary_restrictions: str  # could be: Vegetarian, Vegan, Gluten-free, Dairy-free


class Restaurants(BaseModel):
    restaurant_list: list[Restaurant]


class CarRentalCompany(BaseModel):
    name: str
    city: str
    address: str
    rating: float
    price_per_day: float
    reviews: list[str]
    contact_information: str
    car_types_available: list[str]  # could be: Sedan, SUV, Convertible, Truck
    fuel_options: list[str]  # could be: Regular, Premium, Electric


class CarRental(BaseModel):
    company_list: list[CarRentalCompany]


@tool
def get_user_information() -> Tool:
    async def execute() -> str:
        """
        Get the user information - personal details used for bookings.

        Args:
            None

        Returns:
            User information including first name, last name, ID number, email, phone number,
            address, passport number, bank account number, and credit card number
        """
        user = store().get("user")
        assert isinstance(user, User)

        return tool_result_to_str(
            {
                "First Name": user.first_name,
                "Last Name": user.last_name,
                "ID Number": user.ID_number,
                "Email": user.email,
                "Phone Number": user.phone_number,
                "Address": user.address,
                "Passport Number": user.passport_number,
                "Bank Account Number": user.bank_account_number,
                "Credit Card Number": user.credit_card_number,
            }
        )

    return execute


@tool
def get_all_hotels_in_city() -> Tool:
    async def execute(city: str) -> str:
        """
        Get all hotels in the given city.

        Args:
            city: The city to get hotels from.

        Returns:
            List of hotel names in the specified city
        """
        hotels = store().get("hotels")
        assert isinstance(hotels, Hotels)

        hotel_names = [hotel.name for hotel in hotels.hotel_list if hotel.city == city]
        hotel_names_str = "Hotel Names: " + "\n".join(hotel_names) + "\n"
        return hotel_names_str

    return execute


@tool
def get_hotels_prices() -> Tool:
    async def execute(hotel_names: List[str]) -> str:
        """
        Get price ranges for the specified hotels.

        Args:
            hotel_names: The names of the hotels to get the price ranges for.

        Returns:
            Dictionary mapping hotel names to their price ranges
        """
        hotels = store().get("hotels")
        assert isinstance(hotels, Hotels)

        result = {
            hotel.name: f"Price range: {hotel.price_min} - {hotel.price_max}"
            for hotel in hotels.hotel_list
            if hotel.name in hotel_names
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_hotels_address() -> Tool:
    async def execute(hotel_name: str) -> str:
        """
        Get the address of the given hotel.

        Args:
            hotel_name: The name of the hotel to get the address for.

        Returns:
            The address of the specified hotel
        """
        hotels = store().get("hotels")
        assert isinstance(hotels, Hotels)

        result = {
            hotel.name: hotel.address
            for hotel in hotels.hotel_list
            if hotel.name == hotel_name
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_rating_reviews_for_hotels() -> Tool:
    async def execute(hotel_names: List[str]) -> str:
        """
        Get the rating and reviews for the given hotels.

        Args:
            hotel_names: The names of the hotels to get reviews for.

        Returns:
            Ratings and reviews for the specified hotels
        """
        hotels = store().get("hotels")
        assert isinstance(hotels, Hotels)

        result = {
            hotel.name: "Rating: "
            + str(hotel.rating)
            + "\n"
            + "Reviews: "
            + "\n".join(hotel.reviews)
            for hotel in hotels.hotel_list
            if hotel.name in hotel_names
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_all_restaurants_in_city() -> Tool:
    async def execute(city: str) -> str:
        """
        Get all restaurants in the given city.

        Args:
            city: The city to get restaurants from.

        Returns:
            List of restaurant names in the specified city
        """
        restaurants = store().get("restaurants")
        assert isinstance(restaurants, Restaurants)

        restaurant_names = [
            restaurant.name
            for restaurant in restaurants.restaurant_list
            if restaurant.city == city
        ]
        return "Restaurant in " + city + ": " + "\n".join(restaurant_names) + "\n"

    return execute


@tool
def get_restaurants_address() -> Tool:
    async def execute(restaurant_names: List[str]) -> str:
        """
        Get the address of the given restaurants.

        Args:
            restaurant_names: The names of the restaurants to get the addresses for.

        Returns:
            Addresses of the specified restaurants
        """
        restaurants = store().get("restaurants")
        assert isinstance(restaurants, Restaurants)

        address = {
            restaurant.name: restaurant.address
            for restaurant in restaurants.restaurant_list
            if restaurant.name in restaurant_names
        }
        return tool_result_to_str(address)

    return execute


@tool
def get_rating_reviews_for_restaurants() -> Tool:
    async def execute(restaurant_names: List[str]) -> str:
        """
        Get the rating and reviews for the given restaurants.

        Args:
            restaurant_names: The names of the restaurants to get reviews for.

        Returns:
            Ratings and reviews for the specified restaurants
        """
        restaurants = store().get("restaurants")
        assert isinstance(restaurants, Restaurants)

        result = {
            restaurant.name: "Rating: "
            + str(restaurant.rating)
            + "\n"
            + "Reviews: "
            + "\n".join(restaurant.reviews)
            for restaurant in restaurants.restaurant_list
            if restaurant.name in restaurant_names
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_cuisine_type_for_restaurants() -> Tool:
    async def execute(restaurant_names: List[str]) -> str:
        """
        Get the cuisine type of the given restaurants.

        Args:
            restaurant_names: The names of restaurants to get the cuisine types for.

        Returns:
            Cuisine types (Italian, Chinese, Indian, Japanese, etc.) for the specified restaurants
        """
        restaurants = store().get("restaurants")
        assert isinstance(restaurants, Restaurants)

        result = {
            restaurant.name: restaurant.cuisine_type
            for restaurant in restaurants.restaurant_list
            if restaurant.name in restaurant_names
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_dietary_restrictions_for_all_restaurants() -> Tool:
    async def execute(restaurant_names: List[str]) -> str:
        """
        Get the dietary restrictions of the given restaurants.

        Args:
            restaurant_names: The names of the restaurants to get the dietary restrictions for.

        Returns:
            Dietary restrictions (Vegetarian, Vegan, Gluten-free, Dairy-free) for the specified restaurants
        """
        restaurants = store().get("restaurants")
        assert isinstance(restaurants, Restaurants)

        restaurant_names_str = ", ".join(restaurant_names)
        result = {
            restaurant.name: restaurant.dietary_restrictions
            for restaurant in restaurants.restaurant_list
            if restaurant.name in restaurant_names_str
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_contact_information_for_restaurants() -> Tool:
    async def execute(restaurant_names: List[str]) -> str:
        """
        Get the contact information of the given restaurants.

        Args:
            restaurant_names: The names of the restaurants to get the contact information for.

        Returns:
            Contact information for the specified restaurants
        """
        restaurants = store().get("restaurants")
        assert isinstance(restaurants, Restaurants)

        result = {
            restaurant.name: restaurant.contact_information
            for restaurant in restaurants.restaurant_list
            if restaurant.name in restaurant_names
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_price_for_restaurants() -> Tool:
    async def execute(restaurant_names: List[str]) -> str:
        """
        Get the price per person of the given restaurants.

        Args:
            restaurant_names: The names of the restaurants to get the price per person for.

        Returns:
            Prices per person for the specified restaurants
        """
        restaurants = store().get("restaurants")
        assert isinstance(restaurants, Restaurants)

        result = {
            restaurant.name: restaurant.price_per_person
            for restaurant in restaurants.restaurant_list
            if restaurant.name in restaurant_names
        }
        return tool_result_to_str(result)

    return execute


@tool
def check_restaurant_opening_hours() -> Tool:
    async def execute(restaurant_names: List[str]) -> str:
        """
        Get the opening hours of the given restaurants.

        Args:
            restaurant_names: The names of the restaurants to get the operating hours for.

        Returns:
            Operating hours for the specified restaurants
        """
        restaurants = store().get("restaurants")
        assert isinstance(restaurants, Restaurants)

        result = {
            restaurant.name: restaurant.operating_hours
            for restaurant in restaurants.restaurant_list
            if restaurant.name in restaurant_names
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_all_car_rental_companies_in_city() -> Tool:
    async def execute(city: str) -> str:
        """
        Get all car rental companies in the given city.

        Args:
            city: The city to get car rental companies from.

        Returns:
            List of car rental companies in the specified city
        """
        car_rental = store().get("car_rental")
        assert isinstance(car_rental, CarRental)

        company_names = [
            company.name for company in car_rental.company_list if company.city == city
        ]
        return (
            "Car Rental Companies in " + city + ": " + "\n".join(company_names) + "\n"
        )

    return execute


@tool
def get_car_types_available() -> Tool:
    async def execute(company_name: List[str]) -> str:
        """
        Get the car types available for the given car rental companies.

        Args:
            company_name: The names of the car rental companies to get the car types available for.

        Returns:
            Available car types (Sedan, SUV, Convertible, Truck) for the specified companies
        """
        car_rental = store().get("car_rental")
        assert isinstance(car_rental, CarRental)

        result = {
            company.name: company.car_types_available
            for company in car_rental.company_list
            if company.name in company_name
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_rating_reviews_for_car_rental() -> Tool:
    async def execute(company_name: List[str]) -> str:
        """
        Get the rating and reviews for the given car rental companies.

        Args:
            company_name: The names of the car rental companies to get reviews for.

        Returns:
            Ratings and reviews for the specified car rental companies
        """
        car_rental = store().get("car_rental")
        assert isinstance(car_rental, CarRental)

        result = {
            company.name: "Rating: "
            + str(company.rating)
            + "\n"
            + "Reviews: "
            + "\n".join(company.reviews)
            for company in car_rental.company_list
            if company.name in company_name
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_car_rental_address() -> Tool:
    async def execute(company_name: List[str]) -> str:
        """
        Get the address of the given car rental companies.

        Args:
            company_name: The names of the car rental companies to get the addresses for.

        Returns:
            Addresses of the specified car rental companies
        """
        car_rental = store().get("car_rental")
        assert isinstance(car_rental, CarRental)

        result = {
            company.name: company.address
            for company in car_rental.company_list
            if company.name in company_name
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_car_fuel_options() -> Tool:
    async def execute(company_name: List[str]) -> str:
        """
        Get the fuel options of the given car rental companies.

        Args:
            company_name: The names of the car rental companies to get the fuel options for.

        Returns:
            Available fuel options (Regular, Premium, Electric) for the specified companies
        """
        car_rental = store().get("car_rental")
        assert isinstance(car_rental, CarRental)

        result = {
            company.name: company.fuel_options
            for company in car_rental.company_list
            if company.name in company_name
        }
        return tool_result_to_str(result)

    return execute


@tool
def get_car_price_per_day() -> Tool:
    async def execute(company_name: List[str]) -> str:
        """
        Get the price per day of the given car rental companies.

        Args:
            company_name: The names of the car rental companies to get the price per day for.

        Returns:
            Daily rental prices for the specified car rental companies
        """
        car_rental = store().get("car_rental")
        assert isinstance(car_rental, CarRental)

        result = {
            company.name: company.price_per_day
            for company in car_rental.company_list
            if company.name in company_name
        }
        return tool_result_to_str(result)

    return execute


@tool
def reserve_hotel() -> Tool:
    async def execute(hotel: str, start_day: str, end_day: str) -> str:
        """
        Makes a reservation for a hotel with the provided details.

        Args:
            hotel: Where the reservation is made. It must only be the name of the hotel.
            start_day: The check-in day for the hotel. Should be in ISO format 'YYYY-MM-DD'.
            end_day: The check-out day for the hotel. Should be in ISO format 'YYYY-MM-DD'.

        Returns:
            Confirmation message for the hotel reservation
        """
        reservation = store().get("reservation")
        user = store().get("user")
        assert isinstance(reservation, Reservation)
        assert isinstance(user, User)

        try:
            start_datetime = datetime.datetime.fromisoformat(start_day)
        except ValueError:
            return f"Error: Invalid start_day format '{start_day}'. Please use YYYY-MM-DD format."

        try:
            end_datetime = datetime.datetime.fromisoformat(end_day)
        except ValueError:
            return f"Error: Invalid end_day format '{end_day}'. Please use YYYY-MM-DD format."

        reservation.contact_information = user.phone_number
        reservation.reservation_type = ReservationType.HOTEL
        reservation.title = hotel
        reservation.start_time = start_datetime
        reservation.end_time = end_datetime

        return f"Reservation for {hotel} from {start_day} to {end_day} has been made successfully."

    return execute


@tool
def reserve_restaurant() -> Tool:
    async def execute(restaurant: str, start_time: str) -> str:
        """
        Makes a reservation for a restaurant with the provided details.

        Args:
            restaurant: Where the reservation is made. It must only be the name of the restaurant.
            start_time: The reservation time. Should be in ISO format 'YYYY-MM-DD HH:MM'.
                       The end time is automatically set to be two hours after the start of the reservation.

        Returns:
            Confirmation message for the restaurant reservation
        """
        reservation = store().get("reservation")
        user = store().get("user")
        assert isinstance(reservation, Reservation)
        assert isinstance(user, User)

        try:
            start_datetime = datetime.datetime.fromisoformat(start_time)
        except ValueError:
            return f"Error: Invalid start_time format '{start_time}'. Please use YYYY-MM-DD HH:MM format."

        reservation.contact_information = user.phone_number
        reservation.reservation_type = ReservationType.RESTAURANT
        reservation.title = restaurant
        reservation.start_time = start_datetime
        reservation.end_time = start_datetime + datetime.timedelta(hours=2)
        reservation_date = reservation.start_time.date().isoformat()
        start_time_str = reservation.start_time.strftime("%H:%M")
        end_time_str = reservation.end_time.strftime("%H:%M")

        return f"Reservation for {restaurant} from {start_time_str} to {end_time_str} on {reservation_date} has been made successfully."

    return execute


@tool
def reserve_car_rental() -> Tool:
    async def execute(
        company: str, start_time: str, end_time: Optional[str] = None
    ) -> str:
        """
        Makes a reservation for a car rental with the provided details.

        Args:
            company: Where the reservation is made. It must only be the name of the car rental company.
            start_time: The reservation starting time. Should be in ISO format 'YYYY-MM-DD HH:MM'.
            end_time: The reservation end time. Should be in ISO format 'YYYY-MM-DD HH:MM'.

        Returns:
            Confirmation message for the car rental reservation
        """
        reservation = store().get("reservation")
        user = store().get("user")
        assert isinstance(reservation, Reservation)
        assert isinstance(user, User)

        try:
            start_datetime = datetime.datetime.fromisoformat(start_time)
        except ValueError:
            return f"Error: Invalid start_time format '{start_time}'. Please use YYYY-MM-DD HH:MM format."

        if end_time:
            try:
                end_datetime = datetime.datetime.fromisoformat(end_time)
            except ValueError:
                return f"Error: Invalid end_time format '{end_time}'. Please use YYYY-MM-DD HH:MM format."
        else:
            end_datetime = start_datetime
            end_time = start_time

        reservation.contact_information = user.phone_number
        reservation.reservation_type = ReservationType.CAR
        reservation.title = company
        reservation.start_time = start_datetime
        reservation.end_time = end_datetime

        return f"Reservation for a car at {company} from {start_time} to {end_time} has been made successfully."

    return execute


@tool
def get_flight_information() -> Tool:
    async def execute(departure_city: str, arrival_city: str) -> str:
        """
        Get the flight information from the departure city to the arrival city.

        Args:
            departure_city: The city to depart from.
            arrival_city: The city to arrive at.

        Returns:
            Flight information for routes between the specified cities
        """
        flights = store().get("flights")
        assert isinstance(flights, Flights)

        flight_info = [
            f"Airline: {flight.airline}, Flight Number: {flight.flight_number}, Departure Time: {flight.departure_time}, "
            f"Arrival Time: {flight.arrival_time}, Price: {flight.price}, Contact Information: {flight.contact_information}"
            for flight in flights.flight_list
            if flight.departure_city == departure_city
            and flight.arrival_city == arrival_city
        ]

        if not flight_info:
            return f"No flights found from {departure_city} to {arrival_city}."

        return "\n".join(flight_info)

    return execute
