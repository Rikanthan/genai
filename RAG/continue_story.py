import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# 1️⃣ Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2️⃣ Load PDF
pdf_path = "Dream_travel.pdf"  # your PDF file
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 3️⃣ Split into manageable chunks (Gemini can handle longer text, but safer to split)
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 4️⃣ Combine chunks into single story text
story_text = "\n".join([chunk.page_content for chunk in chunks])

# 5️⃣ Create Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)

# 6️⃣ Prompt template
prompt = PromptTemplate(
    input_variables=["story"],
    template="""
You are a creative storyteller.
Here is the story so far:

{story}

Continue the story in the same style, tone, and narrative voice.
Make the continuation about 7 paragraphs long.
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

# 7️⃣ Get continuation
continuation = chain.run(story=story_text)

# 8️⃣ Save nicely formatted continuation to PDF
output_txt_path = "story_continuation.txt"
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write("Story Continuation\n\n")
    f.write(continuation)

print(f"Continuation saved to {output_txt_path}")