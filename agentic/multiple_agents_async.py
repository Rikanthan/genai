import asyncio
import os
from dotenv import load_dotenv
from litellm import completion

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

# 📦 Shared LLM wrapper
def call_gemini(prompt: str, model="gemini/gemini-1.5-flash"):
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key
    )
    return response["choices"][0]["message"]["content"]

# 🤖 AsyncAgent definition
class AsyncAgent:
    def __init__(self, name, role, goal, backstory):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory

    async def run(self, input_prompt):
        full_prompt = (
            f"You are {self.role} named {self.name}. "
            f"Your goal is: {self.goal}. "
            f"Background: {self.backstory}.\n\n"
            f"Now respond to this:\n{input_prompt}"
        )
        return await asyncio.to_thread(call_gemini, full_prompt)

# 🎭 Define multiple agents
comedian = AsyncAgent(
    name="JesterBot",
    role="Comedian",
    goal="Make people laugh with AI-themed jokes",
    backstory="A witty AI with a sense of humor about machine learning and agents"
)

critic = AsyncAgent(
    name="CritiqueBot",
    role="Comedy Critic",
    goal="Evaluate and critique AI-generated humor",
    backstory="A sarcastic AI comedy judge from the future"
)
instructions1 = (
    "You are a sales agent working for ComplAI, a company providing "
    "a SaaS tool for SOC2 compliance and audit preparation powered by AI. "
    "Write professional, serious cold emails."
)
instructions2 = (
    "You are a humorous, engaging sales agent working for ComplAI, a company providing "
    "a SaaS tool for SOC2 compliance and audit preparation powered by AI. "
    "Write witty, engaging cold emails."
)
instructions3 = (
    "You are a busy sales agent working for ComplAI, a company providing "
    "a SaaS tool for SOC2 compliance and audit preparation powered by AI. "
    "Write concise, to-the-point cold emails."
)

sales_agent1 = AsyncAgent(name="Agent1", role="Professional Sales Agent", goal=instructions1, backstory=instructions1)
sales_agent2 = AsyncAgent(name="Agent2",role="Engaging Sales Agent", goal=instructions2, backstory=instructions2)
sales_agent3 = AsyncAgent(name="Agent3",role="Busy Sales Agent", goal=instructions3, backstory=instructions3)
prompt = "Write a cold email for ComplAI, an AI-powered SOC2 compliance platform."
# 🧪 Main async function
async def main():
    print("Professional Sales Agent...")
    sales_agent1_task = asyncio.create_task(sales_agent1.run(prompt))
    result1 = await sales_agent1_task
    print("Professional Sales Agent email", result1)

    print("Engaging Sales Agent...")
    sales_agent2_task = asyncio.create_task(sales_agent2.run(prompt))
    result2 = await sales_agent2_task
    print("Professional Sales Agent email", result2)

    print("Busy Sales Agent...")
    sales_agent3_task = asyncio.create_task(sales_agent3.run(prompt))
    result3 = await sales_agent3_task
    print("Busy Sales Agent email", result3)

if __name__ == "__main__":
    asyncio.run(main())
