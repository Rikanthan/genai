import asyncio
import os
from dotenv import load_dotenv
from litellm import completion

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment")

# Async agent runner
class AsyncGeminiAgent:
    def __init__(self, role, goal, backstory, model="gemini/gemini-1.5-flash"):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.model = model
        self.api_key = api_key

    async def run(self, prompt: str):
        return await asyncio.to_thread(self._call_model, prompt)

    def _call_model(self, prompt: str):
        full_prompt = (
            f"You are {self.role}. "
            f"Your goal is: {self.goal}. "
            f"Background: {self.backstory}. "
            f"Now answer this prompt:\n\n{prompt}"
        )
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            api_key=self.api_key
        )
        return response["choices"][0]["message"]["content"]

# Create an agent
agent = AsyncGeminiAgent(
    role="Comedian AI",
    goal="Tell smart jokes about AI",
    backstory="An AI trained on thousands of tech stand-up shows"
)

# Async run function
async def main():
    print("🎤 Asking the agent to tell a joke about autonomous AI agents...")
    result = await agent.run("Tell a joke about autonomous AI agents")
    print("\n🤣 Final Output:\n", result)

if __name__ == "__main__":
    asyncio.run(main())