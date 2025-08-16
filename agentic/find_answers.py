import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.agents import initialize_agent, Tool

# For PDF text extraction
from langchain_community.document_loaders import PyPDFLoader

# OCR deps (optional)
try:
    from pdf2image import convert_from_path
    import pytesseract
    from pytesseract import TesseractNotFoundError
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# =========================
# CONFIG
# =========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
FORCE_OUTPUT_LANG = os.getenv("OUTPUT_LANGUAGE")  # optional: "ta" or "en"

# =========================
# OCR fallback
# =========================
def extract_text_with_ocr(pdf_path: str) -> str:
    if not OCR_AVAILABLE:
        print("⚠️ OCR not available (Poppler or Tesseract not installed).")
        return ""
    try:
        pages = convert_from_path(pdf_path)
        text = ""
        for page in pages:
            try:
                text += pytesseract.image_to_string(page, lang="eng+tam")
            except TesseractNotFoundError:
                print("⚠️ Tesseract not installed. Install it for OCR support.")
                return ""
            text += "\n\n"
        return text.strip()
    except Exception as e:
        print(f"⚠️ OCR failed: {e}")
        return ""

# =========================
# Language detection (script-based)
# =========================
def detect_output_language(text: str) -> str:
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
    latin_chars = len(re.findall(r'[A-Za-z]', text))

    if tamil_chars > 50 and tamil_chars >= latin_chars * 1.2:
        return "ta"
    if latin_chars > 50 and latin_chars >= tamil_chars * 1.2:
        return "en"

    # fallback
    try:
        from langdetect import detect
        code = detect(text)
        if code.startswith("ta"):
            return "ta"
        if code.startswith("en"):
            return "en"
    except Exception:
        pass

    return "en"

def lang_code_to_name(code: str) -> str:
    return {"ta": "Tamil", "en": "English"}.get(code, "English")

# =========================
# Tools
# =========================
def load_pdf_text_tool(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text = "\n".join(d.page_content for d in docs)
    if not text.strip():
        text = extract_text_with_ocr(pdf_path)
    return text if text.strip() else "⚠️ No text could be extracted from this PDF."

# LLM + parser
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["question_paper", "output_language"],
    template="""
You are a Maths problem solver. 
Write ALL answers in {output_language} ONLY.

Exam paper:
{question_paper}

For each question:
- Provide EXACTLY 5 steps labeled: Step 1, Step 2, Step 3, Step 4, Step 5.
- Do NOT translate the question, just answer in {output_language}.
- Output strictly in this format:

Q1:
Step 1: ...
Step 2: ...
Step 3: ...
Step 4: ...
Step 5: ...
Final Answer: ...

Q2:
Step 1: ...
...
Final Answer: ...
"""
)

# Chain with safe parser
chain = prompt | llm | StrOutputParser()

def solve_math_tool(question_paper: str) -> str:
    if "No text could be extracted" in question_paper:
        return "⚠️ Cannot solve because no text was extracted."
    code = FORCE_OUTPUT_LANG if FORCE_OUTPUT_LANG in {"ta", "en"} else detect_output_language(question_paper)
    lang_name = lang_code_to_name(code)
    return chain.invoke({"question_paper": question_paper, "output_language": lang_name})

def save_to_txt_tool(answers: str) -> str:
    output_path = "answers.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(answers.strip() + "\n")
    return f"✅ Answers saved to {output_path}"

# =========================
# Agent
# =========================
tools = [
    Tool(name="Load PDF", func=load_pdf_text_tool,
         description="Extract text from a PDF, with OCR fallback"),
    Tool(name="Solve Maths", func=solve_math_tool,
         description="Solve maths problems with exactly 5 steps per question, in detected language",
         return_direct=True),   # ensure no parsing issues
    Tool(name="Save TXT", func=save_to_txt_tool,
         description="Save answers to answers.txt"),
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# =========================
# Run
# =========================
pdf_path = "MathsTM.pdf"
result = agent.run(f"""
1) Load the exam from {pdf_path}.
2) Solve all questions with exactly 5 steps each in the detected language.
3) Save the answers to a TXT file.
""")

print(result)
