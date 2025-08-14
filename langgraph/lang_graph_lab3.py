from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import requests
import gradio as gr
import os
import asyncio
import nest_asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import Tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.checkpoint.memory import MemorySaver
# Load environment variables
load_dotenv(override=True)

# Apply nest_asyncio for environments with existing event loops (e.g. Jupyter)
nest_asyncio.apply()

# Define LangGraph state
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Set up Pushover notification tool
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(text: str):
    """Send a push notification to the user"""
    requests.post(pushover_url, data={
        "token": pushover_token,
        "user": pushover_user,
        "message": text
    })

tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="Useful for when you want to send a push notification"
)

async_browser = create_async_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
# Set up Playwright browser tools
async def setup_browser_tools():    
    for tool in tools:
        print(f"{tool.name} = {tool}")
    
    return {tool.name: tool for tool in tools}

# Async function to navigate and extract text
async def navigate_and_extract():
    tool_dict = await setup_browser_tools()
    navigate_tool = tool_dict.get("navigate_browser")
    extract_text_tool = tool_dict.get("extract_text")

    await navigate_tool.arun({"url": "https://www.cnn.com"})
    text = await extract_text_tool.arun({})
    return text

# Entry point

all_tools = tools + [tool_push]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",          # Or "gemini-1.5-pro", "gemini-1.5-flash", etc.
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
llm_with_tools = llm.bind_tools(all_tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=all_tools))
graph_builder.add_conditional_edges( "chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    text = asyncio.run(navigate_and_extract())
    print("\nExtracted Text:\n")
    print(text)

config = {"configurable": {"thread_id": "10"}}

async def chat(user_input: str, history):
    result = await graph.ainvoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
