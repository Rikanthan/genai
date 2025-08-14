import os
import asyncio
import re
from dotenv import load_dotenv
from litellm import completion
from crewai import Agent, Task, Crew
from sib_api_v3_sdk import ApiClient, Configuration
from sib_api_v3_sdk.api.transactional_emails_api import TransactionalEmailsApi
from sib_api_v3_sdk.models import SendSmtpEmail
from sib_api_v3_sdk.rest import ApiException
from html import escape
# Load env vars
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")

# ---------------- Gemini Wrapper ---------------- #
class GeminiLLMWrapper:
    def __init__(self, model="gemini/gemini-1.5-flash"):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in .env")
        self.model = model
        self.api_key = GOOGLE_API_KEY

    async def run_streaming(self, prompt: str):
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
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
        )
        return response["choices"][0]["message"]["content"]

llm = GeminiLLMWrapper()

# ---------------- Agent Factory ---------------- #
def create_sales_agent(name, tone, style):
    prompt = (
        f"You are a {tone.lower()} sales agent working for ComplAI, a company providing "
        f"a SaaS tool for SOC2 compliance and audit preparation powered by AI. "
        f"Write {style} cold emails."
    )
    return Agent(role=f"{tone} Sales Agent", goal=prompt, backstory=prompt, verbose=True, llm=llm)

# Create agents
agents = [
    create_sales_agent("Agent1", "Professional", "professional, serious"),
    create_sales_agent("Agent2", "Engaging", "witty, engaging"),
    create_sales_agent("Agent3", "Busy", "concise, to-the-point"),
]

# ---------------- Task Setup ---------------- #
email_prompt = "Write a cold email for ComplAI, an AI-powered SOC2 compliance platform."
tasks = [
    Task(description=email_prompt, expected_output="A persuasive cold email", agent=agents[0]),
    Task(description=email_prompt, expected_output="A persuasive cold email", agent=agents[1]),
    Task(description=email_prompt, expected_output="A persuasive cold email", agent=agents[2]),
]

# ---------------- Run Initial Crew ---------------- #
initial_crew = Crew(agents=agents, tasks=tasks, verbose=True)
initial_crew.kickoff()

# Collect outputs
agent_outputs = {task.agent.role: task.output for task in tasks}

# ---------------- Evaluation Agent ---------------- #
review_input = "\n\n".join([f"{agent}:\n{output}" for agent, output in agent_outputs.items()])
reviewer_agent = Agent(
    role="Sales Manager",
    goal="Review cold emails and select the best one",
    backstory="An experienced sales manager who knows what converts",
    verbose=True,
    llm=llm
)

evaluation_task = Task(
    description=f"""You're given 3 cold emails written by different agents. 
Evaluate them based on clarity, persuasiveness, and likelihood to get a response.
Pick the best one and explain why.

Emails:
{review_input}
""",
    expected_output="The best email and a short explanation.",
    agent=reviewer_agent
)

review_crew = Crew(agents=[reviewer_agent], tasks=[evaluation_task], verbose=True)
final_result = review_crew.kickoff()

# ---------------- Extract Best Email ---------------- #
def extract_best_email(result_text, agent_outputs: dict) -> tuple[str, str]:
    result_text = str(result_text)  # Convert CrewOutput to string
    for agent_role in agent_outputs:
        if agent_role.lower() in result_text.lower():
            return agent_role, agent_outputs[agent_role]
    return "Unknown", "No matching agent found."

best_agent, best_email = extract_best_email(final_result, agent_outputs)
# ---------------- Send Email via Brevo ---------------- #
def send_email_with_brevo(subject, html_content, recipient_name, recipient_email):
    if not BREVO_API_KEY:
        raise ValueError("BREVO_API_KEY not found in .env")

    configuration = Configuration()
    configuration.api_key['api-key'] = BREVO_API_KEY
    api_instance = TransactionalEmailsApi(ApiClient(configuration))

    email = SendSmtpEmail(
        to=[{"email": recipient_email, "name": recipient_name}],
        sender={"name": "ComplAI Sales Bot", "email": "rikanthanricky@gmail.com"},
        subject=subject,
        html_content=f"<h2>{subject}</h2><p>{html_content}</p>"
    )

    try:
        response = api_instance.send_transac_email(email)
        print("📤 Email sent successfully. Message ID:", response.message_id)
    except ApiException as e:
        print("❌ Failed to send email:", e)

def format_email_html(email_output) -> str:
    # Convert TaskOutput to string safely
    text = str(email_output).strip()
    safe_text = escape(text)
    return "<br>".join(safe_text.splitlines())
# Final send
formatted_html = format_email_html(best_email)

send_email_with_brevo(
    subject=f"Best Cold Email by {best_agent}",
    html_content=f"<div style='font-family: Arial, sans-serif; font-size: 16px;'>{formatted_html}</div>",
    recipient_name="Rikanthan",
    recipient_email="kingnotfound1@gmail.com"
)
