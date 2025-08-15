import os
from dotenv import load_dotenv

from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import get_gmail_credentials, build_resource_service
from langchain_googledrive.tools.google_drive.tool import GoogleDriveSearchTool
from langchain_googledrive.utilities.google_drive import GoogleDriveAPIWrapper

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# -------------------- Load ENV --------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# -------------------- Gmail Setup --------------------
# Make sure credentials.json exists from Google Cloud OAuth setup
# And token.json will be generated after first run
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=[
        "https://mail.google.com/",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.readonly"
    ],
    client_secrets_file="credentials.json",
)

api_resource = build_resource_service(credentials=credentials)
gmail_toolkit = GmailToolkit(api_resource=api_resource)
gmail_tools = gmail_toolkit.get_tools()

# -------------------- Google Drive Setup --------------------
os.environ["GOOGLE_ACCOUNT_FILE"] = "credentials.json"

google_drive_tool = GoogleDriveSearchTool(
    api_wrapper=GoogleDriveAPIWrapper(
        folder_id="root",       # search entire drive
        num_results=2,
        template="gdrive-query-in-folder"
    )
)

# -------------------- LLM Setup --------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key,
    temperature=0
)

# -------------------- Prompt --------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can send emails via Gmail and search files in Google Drive."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# -------------------- Agent --------------------
all_tools = gmail_tools + [google_drive_tool]

agent = create_tool_calling_agent(
    llm=llm,
    tools=all_tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True
)

# -------------------- Run Agent --------------------
response = agent_executor.invoke({
    "input": "Draft an email to renujah2020@gmail.com with subject 'Hello' and body 'This is a test email from Gemini agent.' and also search for a file called 'project_proposal'."
})

print(response)
