import os
import asyncio
from dotenv import load_dotenv
from litellm import completion
from crewai import Agent

load_dotenv()


class GeminiLLMWrapper:
    async def run_streaming(self, prompt):
        yield "Hello "
        yield "world!"

llm = GeminiLLMWrapper()

agent = Agent(role="Test", goal="Goal", llm=llm,backstory="test this")

async def main():
    async for c in agent.llm.run_streaming("test"):
        print(c, end="")

import asyncio
asyncio.run(main())
