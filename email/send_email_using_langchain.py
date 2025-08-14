from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import get_gmail_credentials, build_resource_service
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
gemini_api_key = os.getenv('GOOGLE_API_KEY')
# 1. Authenticate Gmail
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],  # full Gmail access
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)

# 2. Load Gmail tools
gmail_tools = GmailToolkit(api_resource=api_resource).get_tools()

# 3. Groq LLM
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key)
llm2 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or gemini-1.5-pro
    google_api_key=gemini_api_key,
    temperature=0
)
# 4. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can send or draft emails using Gmail."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 5. Agent
agent = create_openai_functions_agent(
    llm=llm2,
    tools=gmail_tools,
    prompt=prompt
)

# 6. Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=gmail_tools,
    verbose=True
)

# 7. Send or draft an email
agent_executor.invoke({
    "input": "send an email to rikanthanricky@gmail.com with subject 'Hello' and body 'This is an testing email.'"
})
