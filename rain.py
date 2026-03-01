import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()

# Enable CORS (required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Convenience CLI entrypoint.  When you run this module directly the
# port can be controlled via the ``PORT`` environment variable or by
# passing ``--port`` to ``uvicorn``; this mirrors common PaaS
# conventions.  The examiner tool used earlier tries to boot the server
# by importing the module, so providing a ``main`` guard here avoids the
# application starting twice when the import happens.
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    # you can also override on the command line: ``python rain.py`` will
    # respect PORT, ``uvicorn rain:app --port 1234`` works too.
    uvicorn.run("rain:app", host="0.0.0.0", port=port, reload=True)

# -----------------------------
# Request / Response Models
# -----------------------------
class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# -----------------------------
# Tool Function
# -----------------------------
def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


# -----------------------------
# AI Error Analyzer
# -----------------------------
def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    """
    Use Gemini structured output to extract error line numbers.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
You are a Python debugging assistant.

Identify the exact line number(s) in the user's code where the error occurred.
Return only the line numbers.

CODE:
{code}

TRACEBACK:
{tb}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER),
                    )
                },
                required=["error_lines"],
            ),
        ),
    )

    result = ErrorAnalysis.model_validate_json(response.text)
    return result.error_lines


# -----------------------------
# Endpoint
# -----------------------------
@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):

    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    execution = execute_python_code(request.code)

    # If success → no AI call
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    # If error → invoke AI
    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"],
        )

        return {
            "error": error_lines,
            "result": execution["output"],  # Exact traceback
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI error analysis failed: {str(e)}",
        )