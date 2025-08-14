import os
import asyncio
from dotenv import load_dotenv
from litellm import completion
from crewai import Agent,Task, Crew

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

# Instructions for each sales agent
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

# Create CrewAI Agents using the same llm wrapper
sales_agent1 = Agent(role="Professional Sales Agent", goal=instructions1, backstory=instructions1, verbose=True, llm=llm)
sales_agent2 = Agent(role="Engaging Sales Agent", goal=instructions2, backstory=instructions2, verbose=True, llm=llm)
sales_agent3 = Agent(role="Busy Sales Agent", goal=instructions3, backstory=instructions3, verbose=True, llm=llm)

# Async function to stream response from a given agent
task1 = Task(
    description= "Write a cold email for ComplAI, an AI-powered SOC2 compliance platform.",
    expected_output="A short, persuasive cold email",
    agent=sales_agent1
)

task2 = Task(
    description= "Write a cold email for ComplAI, an AI-powered SOC2 compliance platform.",
    expected_output="A short, persuasive cold email",
    agent=sales_agent1
)

task3 = Task(
    description= "Write a cold email for ComplAI, an AI-powered SOC2 compliance platform.",
    expected_output="A short, persuasive cold email",
    agent=sales_agent1
)

initial_crew = Crew(agents=[sales_agent1, sales_agent2, sales_agent3], tasks=[task1, task2, task3], verbose=True)
initial_crew.kickoff()

# STEP 3: Collect outputs
agent_outputs = {
    task1.agent.role: task1.output,
    task2.agent.role: task2.output,
    task3.agent.role: task3.output,
}

# Combine for review
email_review_text = "\n\n".join(
    [f"{agent}:\n{output}" for agent, output in agent_outputs.items()]
)

# STEP 4: Define a reviewer agent
reviewer_agent = Agent(
    role="Sales Manager",
    goal="Review cold emails and select the best one",
    backstory="An experienced sales manager who knows what converts",
    verbose=True,
    llm=llm
)

# STEP 5: Define task to evaluate
evaluation_task = Task(
    description=f"""You're given 3 cold emails written by different agents. 
Evaluate all based on clarity, persuasiveness, and likelihood to get a response.
Pick the best one and explain why.

Emails:
{email_review_text}
""",
    expected_output="The best email and a short explanation.",
    agent=reviewer_agent
)

# STEP 6: Run final review
review_crew = Crew(agents=[reviewer_agent], tasks=[evaluation_task], verbose=True)
final_result = review_crew.kickoff()

# ✅ Print all results
print("\n📨 Initial Agent Outputs:")
for role, output in agent_outputs.items():
    print(f"\n🔹 {role}:\n{output}")

print("\n🏆 Final Evaluation by Sales Manager:\n", final_result)