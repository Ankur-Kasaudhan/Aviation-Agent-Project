"""
Travel Planning Agent System

This module implements a multi-agent workflow for travel planning using LangGraph.
It orchestrates specialized agents for flight search, hotel search, itinerary
creation, and final response generation, with persistent state management using PostgreSQL.
"""

import os
from typing import TypedDict, Annotated
import operator

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection URL for PostgreSQL checkpoint persistence
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize LLM with Groq's Llama model
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


# State Definition
class TravelState(TypedDict):
    """
    State schema for the travel planning workflow.
    
    Attributes:
        messages: List of conversation messages (accumulated across agents)
        user_query: Original user travel request
        flight_results: Flight search results from flight_agent
        hotel_results: Hotel search results from hotel_agent
        itinerary: Generated travel itinerary from itinerary_agent
        llm_calls: Counter tracking total LLM invocations
    """
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# Flight Agent
def flight_agent(state: TravelState):
    """
    Agent responsible for searching flight information.
    
    Args:
        state (TravelState): Current workflow state containing user query
        
    Returns:
        dict: Updated state with flight_results and incremented llm_calls
    """
    query = state["user_query"]
    flight_data = search_flights(query)
    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content=f"Flight results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# Hotel Agent
def hotel_agent(state: TravelState):
    """
    Agent responsible for searching hotel information using Tavily.
    
    Args:
        state (TravelState): Current workflow state containing user query
        
    Returns:
        dict: Updated state with hotel_results and incremented llm_calls
    """
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# Itinerary Agent
def itinerary_agent(state: TravelState):
    """
    Agent responsible for creating a travel itinerary based on flight and hotel data.
    
    This agent combines flight results, hotel results, and the user's original
    query to generate a comprehensive travel plan using the LLM.
    
    Args:
        state (TravelState): Current workflow state containing flight and hotel results
        
    Returns:
        dict: Updated state with generated itinerary and incremented llm_calls
    """
    # Construct prompt with all available travel information
    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """

    # Invoke LLM with system and human messages
    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# Final Response Agent
def final_agent(state: TravelState):
    """
    Agent responsible for generating the final consolidated travel response.
    
    This agent combines all previous results (flights, hotels, itinerary)
    into a polished final response for the user.
    
    Args:
        state (TravelState): Current workflow state with all travel information
        
    Returns:
        dict: Updated state with final response message and incremented llm_calls
    """
    # Construct final prompt with all accumulated data
    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    # Invoke LLM to generate final response
    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# Build LangGraph Workflow
graph = StateGraph(TravelState)

# Add all agent nodes to the graph
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

# Define execution flow (sequential pipeline)
graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# Persistent connection so both CLI and Streamlit can share the compiled app
_conn = psycopg.connect(DATABASE_URL, autocommit=True)

# Initialize PostgreSQL checkpointer for state persistence
checkpointer = PostgresSaver(_conn)
checkpointer.setup()  # Set up database tables if not exist

# Compile the graph with checkpointer for persistent memory
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    """
    CLI entry point for testing the travel planning system.
    
    Runs an interactive session where users can input travel requests
    and receive comprehensive travel plans including flights, hotels,
    and itineraries.
    """
    
    # Configuration for thread-based state persistence
    config = {
        "configurable": {
            "thread_id": "user_aarohi"  # Unique thread identifier for this session
        }
    }

    # Get user input
    user_input = input("Enter travel request: ")

    # Invoke the workflow with initial state
    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    # Display final response
    print("\nFINAL RESPONSE:\n")

    # Print all messages from the workflow
    for msg in result["messages"]:
        print(msg.content)