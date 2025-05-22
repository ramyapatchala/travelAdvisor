import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, Tool, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime, timedelta
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize Streamlit
st.set_page_config(page_title="Multi-Agent Travel Advisor", layout="wide")
st.title("🌍 Multi-Agent Travel Advisor")

# Configuration
MODEL_NAME = "gpt-4-1106-preview"
TEMPERATURE = 0.3
MEMORY_WINDOW = 10  # Number of messages to retain in memory

# Initialize LLM
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# --------------------------
# Core Tools and Components
# --------------------------

class TravelTools:
    @staticmethod
    def get_search_tools():
        """Return search-related tools"""
        search = DuckDuckGoSearchRun()
        google_search = GoogleSearchAPIWrapper()
        
        return [
            Tool(
                name="WebSearch",
                func=search.run,
                description="Useful for general web searches about travel destinations, attractions, etc."
            ),
            Tool(
                name="GoogleSearch",
                func=google_search.run,
                description="Useful for more precise web searches using Google"
            )
        ]

    @staticmethod
    def get_weather_tool():
        """Return weather checking tool"""
        # This would connect to a weather API in production
        def get_weather(location: str, date: Optional[str] = None):
            # Mock implementation - replace with actual API call
            return f"Weather in {location} on {date or 'today'}: Sunny, 25°C"
        
        return Tool(
            name="CheckWeather",
            func=get_weather,
            description="Useful for checking weather conditions at a travel destination"
        )

    @staticmethod
    def get_translation_tool():
        """Return language translation tool"""
        def translate_text(text: str, target_language: str):
            # In production, connect to a translation API
            return f"[Translated to {target_language}]: {text}"
        
        return Tool(
            name="TranslateText",
            func=translate_text,
            description="Useful for translating text between languages"
        )

# --------------------------
# Specialized Agents
# --------------------------

class TravelAgentManager:
    def __init__(self):
        self.agents = {}
        self.conversation_history = []
        self.initialize_agents()
    
    def initialize_agents(self):
        """Create all specialized agents"""
        self.agents = {
            "destination_expert": self.create_destination_expert(),
            "itinerary_planner": self.create_itinerary_planner(),
            "local_guide": self.create_local_guide(),
            "travel_logistics": self.create_travel_logistics(),
            "cultural_adviser": self.create_cultural_adviser()
        }
    
    def create_base_agent(self, tools: List[Tool], system_prompt: str) -> AgentExecutor:
        """Create a base agent with given tools and prompt"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def create_destination_expert(self) -> AgentExecutor:
        """Agent specialized in destination recommendations"""
        tools = TravelTools.get_search_tools() + [
            Tool(
                name="CompareDestinations",
                func=self.compare_destinations,
                description="Useful for comparing different travel destinations"
            )
        ]
        
        system_prompt = """You are a Destination Expert specializing in recommending travel locations based on user preferences.
        Consider budget, interests, travel dates, and other preferences when making recommendations.
        Be specific about why each destination might suit the traveler."""
        
        return self.create_base_agent(tools, system_prompt)
    
    def create_itinerary_planner(self) -> AgentExecutor:
        """Agent specialized in creating detailed itineraries"""
        tools = TravelTools.get_search_tools() + [
            TravelTools.get_weather_tool(),
            Tool(
                name="CalculateTravelTime",
                func=self.calculate_travel_time,
                description="Useful for estimating travel time between locations"
            )
        ]
        
        system_prompt = """You are an Itinerary Planner that creates detailed day-by-day travel plans.
        Consider opening hours, travel time between locations, and realistic time allocations.
        Always include free time for exploration and rest."""
        
        return self.create_base_agent(tools, system_prompt)
    
    def create_local_guide(self) -> AgentExecutor:
        """Agent specialized in local knowledge"""
        tools = TravelTools.get_search_tools() + [
            Tool(
                name="FindLocalEvents",
                func=self.find_local_events,
                description="Useful for finding local events and activities"
            ),
            TravelTools.get_translation_tool()
        ]
        
        system_prompt = """You are a Local Guide with deep knowledge about specific destinations.
        Provide insights about local customs, hidden gems, and authentic experiences.
        Offer practical advice about transportation, safety, and etiquette."""
        
        return self.create_base_agent(tools, system_prompt)
    
    def create_travel_logistics(self) -> AgentExecutor:
        """Agent specialized in travel logistics"""
        tools = TravelTools.get_search_tools() + [
            Tool(
                name="CheckVisaRequirements",
                func=self.check_visa_requirements,
                description="Useful for checking visa and entry requirements"
            ),
            Tool(
                name="FindTransportOptions",
                func=self.find_transport_options,
                description="Useful for finding transportation options between locations"
            )
        ]
        
        system_prompt = """You are a Travel Logistics expert focusing on practical travel arrangements.
        Handle questions about visas, transportation, packing lists, and other practical matters.
        Provide clear, accurate information and verify details when needed."""
        
        return self.create_base_agent(tools, system_prompt)
    
    def create_cultural_adviser(self) -> AgentExecutor:
        """Agent specialized in cultural information"""
        tools = [
            TravelTools.get_translation_tool(),
            Tool(
                name="ExplainCulturalNorms",
                func=self.explain_cultural_norms,
                description="Useful for explaining cultural norms and etiquette"
            )
        ]
        
        system_prompt = """You are a Cultural Adviser specializing in helping travelers understand local customs.
        Explain cultural norms, appropriate behavior, and potential cultural misunderstandings.
        Be sensitive and provide context for your advice."""
        
        return self.create_base_agent(tools, system_prompt)
    
    # --------------------------
    # Custom Tool Implementations
    # --------------------------
    
    def compare_destinations(self, destinations: str) -> str:
        """Compare multiple travel destinations"""
        destinations = destinations.split(",")
        # In production, this would use actual comparison logic
        return f"Comparison of {len(destinations)} destinations: " + "\n".join(
            f"- {d.strip()}: Great for [specific reason]" for d in destinations
        )
    
    def calculate_travel_time(self, route: str) -> str:
        """Calculate travel time between locations"""
        # In production, connect to maps API
        return f"Estimated travel time for {route}: 2-3 hours by car"
    
    def find_local_events(self, location_and_date: str) -> str:
        """Find local events at a destination"""
        parts = location_and_date.split(",")
        location = parts[0].strip()
        date = parts[1].strip() if len(parts) > 1 else "next week"
        return f"Events in {location} around {date}: [list of events]"
    
    def check_visa_requirements(self, country_pair: str) -> str:
        """Check visa requirements between countries"""
        parts = country_pair.split(",")
        from_country = parts[0].strip()
        to_country = parts[1].strip()
        return f"Visa requirements for {from_country} citizens traveling to {to_country}: [details]"
    
    def find_transport_options(self, route: str) -> str:
        """Find transportation options between locations"""
        return f"Transport options for {route}: [list of options]"
    
    def explain_cultural_norms(self, country: str) -> str:
        """Explain cultural norms for a country"""
        return f"Cultural norms in {country}: [detailed explanation]"
    
    # --------------------------
    # Agent Coordination
    # --------------------------
    
    def route_question(self, user_input: str) -> str:
        """Determine which agent(s) should handle the question"""
        # Simple routing logic - in production could use LLM to determine routing
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["recommend", "destination", "where should", "best place"]):
            return ["destination_expert"]
        elif any(word in input_lower for word in ["itinerary", "schedule", "plan", "day by day"]):
            return ["itinerary_planner"]
        elif any(word in input_lower for word in ["local", "hidden gem", "off the beaten path"]):
            return ["local_guide"]
        elif any(word in input_lower for word in ["visa", "pack", "logistics", "transport"]):
            return ["travel_logistics"]
        elif any(word in input_lower for word in ["culture", "custom", "etiquette", "norm"]):
            return ["cultural_adviser"]
        else:
            # Default to all agents for complex questions
            return list(self.agents.keys())
    
    def get_agent_response(self, agent_name: str, user_input: str) -> str:
        """Get response from a specific agent"""
        agent = self.agents[agent_name]
        response = agent.invoke({
            "input": user_input,
            "chat_history": self.conversation_history[-MEMORY_WINDOW:]
        })
        return response["output"]
    
    def get_combined_response(self, user_input: str) -> str:
        """Get coordinated response from multiple agents"""
        agents_to_ask = self.route_question(user_input)
        
        if len(agents_to_ask) == 1:
            return self.get_agent_response(agents_to_ask[0], user_input)
        
        # For multiple agents, get responses and synthesize
        responses = {}
        for agent_name in agents_to_ask:
            responses[agent_name] = self.get_agent_response(agent_name, user_input)
        
        # Synthesize responses
        synthesis_prompt = f"""Combine these expert responses to answer the user's travel question:
        User Question: {user_input}
        
        Expert Responses:
        {responses}
        
        Provide a comprehensive, well-organized answer that addresses all aspects of the question.
        """
        
        synthesizer = llm.invoke(synthesis_prompt)
        return synthesizer.content

# --------------------------
# Streamlit UI
# --------------------------

def main():
    st.sidebar.title("Agent Selection")
    agent_manager = TravelAgentManager()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask your travel question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.spinner("Consulting travel experts..."):
            response = agent_manager.get_combined_response(prompt)
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()