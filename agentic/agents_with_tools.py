import os
import asyncio
from dotenv import load_dotenv
from litellm import completion
from crewai import Agent,Task, Crew
from langchain.tools import StructuredTool,Tool

load_dotenv()

class GeminiLLMWrapper:
    def __init__(self, model="gemini/gemini-1.5-flash"):
        self.model = model
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

    async def run_streaming(self, prompt: str):
        # Call LiteLLM completion with stream=True, async generator
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            stream=True,
        )
        async for chunk in response:
            if chunk and "choices" in chunk:
                delta = chunk["choices"][0]["delta"].get("content", "")
                yield delta

    def __call__(self, prompt, **kwargs):
        # Simple blocking call for full completion
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
        )
        return response["choices"][0]["message"]["content"]

# Create your LLM wrapper instance
llm = GeminiLLMWrapper()

def professional_email_generator(_: str) -> str:
    """Generate a formal cold email for ComplAI."""
    return (
        "Hi,\n\nAs a company preparing for SOC2, you know the importance of compliance. "
        "ComplAI offers a reliable AI-driven platform to streamline audits. Let's connect.\n\n- ComplAI Team"
    )

def engaging_email_generator(_: str) -> str:
    """Generate a witty and engaging cold email for ComplAI."""
    return (
        "Hey there!\n\nSOC2 audits don’t have to suck. ComplAI is the AI-powered compliance wingman you didn’t know you needed. "
        "Wanna see how fun compliance can be?\n\nCheers,\nComplAI"
    )

def concise_email_generator(_: str) -> str:
    """Generate a short and concise cold email for ComplAI."""
    return (
        "Hi,\n\nGet SOC2 ready—fast. ComplAI automates compliance so you don't have to. "
        "One demo, zero stress.\n\n- ComplAI"
    )

tools = [
    Tool.from_function(
        func=professional_email_generator,
        name="professional_email_generator",
        description="Generates a professional cold email for ComplAI"
    ),
    Tool.from_function(
        func=engaging_email_generator,
        name="engaging_email_generator",
        description="Generates an engaging, witty cold email for ComplAI"
    ),
    Tool.from_function(
        func=concise_email_generator,
        name="concise_email_generator",
        description="Generates a short, to-the-point cold email for ComplAI"
    ),
]
# ✅ Define the reviewing agent (Sales Manager)

manager_agent = Agent(
    role="Sales Manager",
    goal="Pick the best cold email for ComplAI that is most likely to convert",
    backstory=(
        "You're a sales expert who knows exactly what kind of messaging converts. "
        "Use the tools available to collect email drafts and pick the most effective one."
    ),
    tools=tools,
    verbose=True,
    llm=llm
)

# ✅ Define the evaluation task

task = Task(
    description="""
Call the following tools to generate 3 styles of cold emails:
- professional_email_generator
- engaging_email_generator
- concise_email_generator

Compare the emails for effectiveness, tone, clarity, and likelihood to convert.
Pick the best one and explain why.
""",
    expected_output="The best email along with a short explanation.",
    agent=manager_agent
)

# ✅ Run the Crew

crew = Crew(
    agents=[manager_agent],
    tasks=[task],
    verbose=True
)

result = crew.kickoff()

print("\n🏆 Best Email Chosen:\n", result)