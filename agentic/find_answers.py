import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool

# For PDF text extraction
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
import pytesseract

# For saving results
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# 1️⃣ Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2️⃣ OCR fallback function
def extract_text_with_ocr(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path)
    text = ""
    for i, page in enumerate(pages):
        text += pytesseract.image_to_string(page, lang="eng+tam")  # Tamil + English OCR
        text += "\n\n"
    return text.strip()

# 3️⃣ Load PDF tool
def load_pdf_text_tool(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text = "\n".join([d.page_content for d in docs])
    if not text.strip():  # fallback if no text found
        text = extract_text_with_ocr(pdf_path)
    return text

# 4️⃣ Gemini LLM + Chain for solving questions
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)

prompt = PromptTemplate(
    input_variables=["question_paper"],
    template="""
You are a Maths problem solver.
Here is the exam paper:

{question_paper}

Analyze each question and provide the answer within 5 steps.
Format strictly as:
Q1: answer
Q2: answer
Q3: answer
...
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def solve_math_tool(question_paper: str) -> str:
    return chain.run(question_paper=question_paper)

# 5️⃣ Save results to PDF tool
def save_to_pdf_tool(answers: str, output_path="answers.pdf") -> str:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path)
    story = []

    story.append(Paragraph("Maths Exam Answers", styles['Title']))
    story.append(Spacer(1, 20))

    for line in answers.split("\n"):
        story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 10))

    doc.build(story)
    return f"Answers saved to {output_path}"

# 6️⃣ Define tools for the agent
tools = [
    Tool(name="Load PDF", func=load_pdf_text_tool, description="Extract text from a PDF, with OCR fallback"),
    Tool(name="Solve Maths", func=solve_math_tool, description="Solve maths problems from exam paper"),
    Tool(name="Save PDF", func=save_to_pdf_tool, description="Save answers to a nicely formatted PDF file"),
]

# 7️⃣ Initialize agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# 8️⃣ Run agent
pdf_path = "MathsTM.pdf"
result = agent.run(f"""
1. Load the exam from {pdf_path}.
2. Solve all the questions step by step.
3. Save the answers to a PDF file.
""")

print(result)
