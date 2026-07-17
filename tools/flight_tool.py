"""
Flight Search Module

This module provides functionality to search for real-time flight information
using the AviationStack API. It retrieves flight data including airline details,
departure/arrival airports, and flight status.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
API_KEY = os.getenv("AVIATIONSTACK_API_KEY")


def search_flights(query):
    """
    Search for flight information using the AviationStack API.
    
    This function queries the AviationStack API for real-time flight data
    and returns formatted flight details including airline, departure/arrival
    airports, and flight status.
    
    Args:
        query (str): Search query parameter (currently not used but kept
                    for potential future filtering functionality).
    
    Returns:
        str: A formatted string containing flight information for up to
             5 flights. Each flight includes airline name, departure airport,
             arrival airport, and current status. If no flights are found,
             returns an empty string.
    
    Example:
        >>> flight_info = search_flights("")
        >>> print(flight_info)
        Airline: American Airlines
        Departure: John F Kennedy International Airport
        Arrival: Los Angeles International Airport
        Status: active
    """
    
    # API endpoint for AviationStack flights
    url = "http://api.aviationstack.com/v1/flights"
    
    # Query parameters for the API request
    params = {
        "access_key": API_KEY,      # API authentication key
        "limit": 5                  # Maximum number of results to return
    }
    
    # Make GET request to the AviationStack API
    response = requests.get(url, params=params)
    
    # Parse JSON response
    data = response.json()
    
    # Initialize list to store flight information
    flights = []
    
    # Check if response contains flight data
    if "data" in data:
        
        # Process up to first 5 flights from the response
        for flight in data["data"][:5]:
            
            # Extract airline name with fallback to "Unknown"
            airline = flight.get("airline", {}).get("name", "Unknown")
            
            # Extract departure airport with fallback to "Unknown"
            departure = flight.get(
                "departure", {}
            ).get("airport", "Unknown")
            
            # Extract arrival airport with fallback to "Unknown"
            arrival = flight.get(
                "arrival", {}
            ).get("airport", "Unknown")
            
            # Extract flight status with fallback to "Unknown"
            status = flight.get("flight_status", "Unknown")
            
            # Create formatted flight information string
            flights.append(
                f"""
                Airline: {airline}
                Departure: {departure}
                Arrival: {arrival}
                Status: {status}
                """
            )
    
    # Return all flight information as a single formatted string
    return "\n".join(flights)