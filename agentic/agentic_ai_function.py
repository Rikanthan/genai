from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv
from litellm import completion

# Load environment variables
load_dotenv()

# Use LiteLLM manually inside an LLM wrapper
class GeminiLLMWrapper:
    def __init__(self, model="gemini/gemini-1.5-flash"):
        self.model = model
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

    def __call__(self, prompt, **kwargs):
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,   # <-- Pass API key here explicitly
        )
        return response["choices"][0]["message"]["content"]

# Initialize LLM wrapper
llm = GeminiLLMWrapper()

# Define the agent
agent = Agent(
    role="AI Trend Researcher",
    goal="Identify key trends in artificial intelligence for 2025",
    backstory="A leading expert in AI trend forecasting and emerging technologies.",
    verbose=True,
    llm=llm
)

# Define the task
task = Task(
    description="Summarize the top 3 emerging AI trends expected to dominate in 2025.",
    expected_output="A bullet-point summary of 3 major AI trends.",
    agent=agent
)

# Build and run the crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n✅ Final Output:\n", result)


