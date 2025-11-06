import io
import os
import json
import base64
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ---- Load .env early ----
load_dotenv()

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "index.html"

# ---- Config / Pricing ----
INPUT_PRICE_PER_1K = float(os.getenv("INPUT_PRICE_PER_1K", "0.003"))
OUTPUT_PRICE_PER_1K = float(os.getenv("OUTPUT_PRICE_PER_1K", "0.015"))
TOKENS_PER_IMAGE = int(os.getenv("TOKENS_PER_IMAGE", "1600"))
TOKENS_PER_PAGE = int(os.getenv("TOKENS_PER_PAGE", "5000"))
DEFAULT_MODEL = os.getenv("MODEL_ID", "us.anthropic.claude-3-sonnet-20240229-v1:0")

ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Models from .env
MODELS_JSON = os.getenv("MODELS_JSON")
MODEL_IDS = os.getenv("MODEL_IDS") or os.getenv("MODELS") or os.getenv("BEDROCK_MODELS")

# ---- App ----
app = FastAPI(title="Bedrock Price Calculator API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

@app.get("/", response_class=HTMLResponse)
def root_page():
    if INDEX_PATH.exists():
        return FileResponse(str(INDEX_PATH))
    return HTMLResponse("<p>Server is running. Place an index.html next to server.py or open /static/index.html.</p>")

# ---- Helpers ----
def parse_models() -> List[Dict[str, str]]:
    if MODELS_JSON:
        try:
            arr = json.loads(MODELS_JSON)
            return [{"id": str(x.get("id") or x.get("model_id")), "label": str(x.get("label") or x.get("name") or x.get("id"))} for x in arr if x]
        except Exception:
            pass
    if MODEL_IDS:
        return [{"id": mid.strip(), "label": mid.strip()} for mid in MODEL_IDS.split(",") if mid.strip()]
    return [{"id": DEFAULT_MODEL, "label": DEFAULT_MODEL}]

def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    text = " ".join(text.split())
    words = len(text.split())
    chars = len(text)
    word_based = int(words * 1.3)
    char_based = int(chars / 4)
    if words < 50:
        return max(1, word_based)
    return max(1, int((word_based + char_based) / 2))

def estimate_output_tokens_from_input_count(input_tokens: int) -> int:
    ratio = 0.6
    est = int(input_tokens * ratio)
    return max(50, min(est, 4000))

def price_for_tokens(input_tokens: int, output_tokens: int):
    input_cost = (input_tokens / 1000.0) * INPUT_PRICE_PER_1K
    output_cost = (output_tokens / 1000.0) * OUTPUT_PRICE_PER_1K
    total = input_cost + output_cost
    return round(input_cost, 6), round(output_cost, 6), round(total, 6)

def _is_image_name(name: str) -> bool:
    name = name.lower()
    return name.endswith((".png",".jpg",".jpeg",".gif",".webp"))

def _mimetype_for_image(name: str) -> str:
    name = name.lower()
    if name.endswith(".png"): return "image/png"
    if name.endswith(".gif"): return "image/gif"
    if name.endswith(".webp"): return "image/webp"
    if name.endswith(".jpg") or name.endswith(".jpeg"): return "image/jpeg"
    return "image/png"

def _extract_pdf_pages(file_bytes: bytes) -> int:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return len(reader.pages)
    except Exception:
        return 1

def _extract_pdf_text(file_bytes: bytes, max_pages: int = 5, max_chars: int = 10000) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        parts = []
        for i, page in enumerate(reader.pages[:max_pages]):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                parts.append(t)
            if sum(len(p) for p in parts) >= max_chars:
                break
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception:
        return ""

def _extract_txt(file_bytes: bytes, max_chars: int = 20000) -> str:
    try:
        t = file_bytes.decode("utf-8", errors="ignore")
        if len(t) > max_chars:
            t = t[:max_chars]
        return t
    except Exception:
        return ""

def _build_bedrock_messages(prompt: str, uploads_data):
    """Use Claude 3 Messages API (Bedrock) with text + input_image blocks."""
    content = []
    base_text = (prompt or "").strip()
    
    # Count what we have
    has_images = any(_is_image_name(item.get("name", "").lower()) for item in (uploads_data or []))
    has_pdfs = any(item.get("name", "").lower().endswith(".pdf") for item in (uploads_data or []))
    has_txt = any(item.get("name", "").lower().endswith(".txt") for item in (uploads_data or []))
    
    # If no prompt but files uploaded, add a default prompt
    if not base_text and uploads_data:
        if has_images:
            base_text = "Please describe what you see in the uploaded image(s) and provide relevant insights."
        elif has_pdfs:
            base_text = "Please analyze and summarize the content of the uploaded PDF document(s)."
        elif has_txt:
            base_text = "Please analyze and summarize the content of the uploaded text file(s)."
        else:
            base_text = "Please analyze the uploaded file(s) and provide a summary of the content."
    
    # Add the prompt first
    if base_text:
        content.append({ "type": "text", "text": base_text })

    # Then add all files
    for item in (uploads_data or []):
        name = item.get("name","")
        data = item.get("bytes", b"")
        if not data:
            continue
        lower = name.lower()
        if _is_image_name(lower):
            b64 = base64.b64encode(data).decode("utf-8")
            content.append({
                "type": "image",
                "source": { 
                    "type": "base64", 
                    "media_type": _mimetype_for_image(lower), 
                    "data": b64 
                }
            })
        elif lower.endswith(".pdf"):
            txt = _extract_pdf_text(data)
            if txt:
                content.append({ "type": "text", "text": f"\n\n=== Content from PDF '{name}' ===\n{txt}\n=== End of PDF content ===" })
            else:
                pages = _extract_pdf_pages(data)
                content.append({ "type": "text", "text": f"\n\nNote: PDF '{name}' has {pages} page(s) but text extraction failed. It may be a scanned/image-based PDF." })
        elif lower.endswith(".txt"):
            txt = _extract_txt(data)
            if txt:
                content.append({ "type": "text", "text": f"\n\n=== Content from '{name}' ===\n{txt}\n=== End of file content ===" })
        else:
            content.append({ "type": "text", "text": f"\n\n[Unsupported file type attached: {name}]" })
    
    return [{ "role": "user", "content": content }]

def bedrock_generate(model_id: str, prompt: str, uploads_data, max_tokens: int = 512, temperature: float = 0.2):
    import boto3
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    )

    messages = _build_bedrock_messages(prompt, uploads_data)
    
    # Enhanced system prompt to handle files better
    system_prompt = """You are a helpful assistant analyzing user inputs and uploaded files.

When responding:
- If PDFs are uploaded, analyze the extracted text content provided
- If images are uploaded, describe what you see in the images
- If TXT files are uploaded, reference their content
- Always provide meaningful insights based on the provided content
- If you see extracted text from files, use that information to answer the user's question"""

    body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "system": system_prompt,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    
    # Debug logging - print what we're sending
    print("=" * 80)
    print("DEBUG: Sending to Bedrock API:")
    print(f"Model: {model_id}")
    print(f"Messages structure: {len(messages)} message(s)")
    for i, msg in enumerate(messages):
        print(f"  Message {i}: role={msg.get('role')}, content blocks={len(msg.get('content', []))}")
        for j, block in enumerate(msg.get('content', [])):
            block_type = block.get('type', 'unknown')
            print(f"    Block {j}: type={block_type}", end='')
            if block_type == 'text':
                text_preview = block.get('text', '')[:100]
                print(f", text_preview='{text_preview}...'")
            elif block_type == 'image':
                has_data = 'data' in block.get('source', {})
                media_type = block.get('source', {}).get('media_type', 'unknown')
                print(f", media_type={media_type}, has_data={has_data}")
            else:
                print()
    print("=" * 80)

    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())

    # Bedrock response with Messages API: payload['content'] -> list of {type:'text', text:'...'}
    output_text = ""
    for block in payload.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            output_text += block.get("text","")

    usage = payload.get("usage", {})
    
    print(f"DEBUG: Response usage: {usage}")
    print(f"DEBUG: Response length: {len(output_text)} chars")
    print("=" * 80)
    
    return output_text.strip(), usage

# ---- API ----
@app.get("/api/models")
def get_models():
    models = parse_models()
    return {"models": models}

def _compute_from_payload(model_id: str, prompt: str, uploads_data: List[Dict[str, Any]]):
    details = {"text_tokens": 0, "image_tokens": 0, "document_tokens": 0}
    input_tokens = 0

    # Combine prompt + extracted text to estimate tokens
    combined_text = (prompt or "")

    pdf_pages_total = 0
    image_count = 0
    pdf_text_found = False
    txt_file_found = False
    
    for item in uploads_data:
        name = item["name"].lower()
        data = item["bytes"]
        if not data:
            continue
        if _is_image_name(name):
            image_count += 1
        elif name.endswith(".pdf"):
            pages = _extract_pdf_pages(data)
            pdf_pages_total += pages
            extracted = _extract_pdf_text(data)
            if extracted:
                combined_text += "\n" + extracted
                pdf_text_found = True
            # Even if no text extracted, add token estimate for visual PDF content
            if not extracted:
                # Estimate tokens for visual PDF pages
                details["document_tokens"] += pages * TOKENS_PER_PAGE
        elif name.endswith(".txt"):
            extracted_txt = _extract_txt(data)
            if extracted_txt:
                combined_text += "\n" + extracted_txt
                txt_file_found = True
        else:
            # Unknown file type - treat as one page document
            pdf_pages_total += 1

    # Calculate text tokens from all combined text (prompt + extracted content)
    if combined_text.strip():
        details["text_tokens"] = estimate_tokens_from_text(combined_text)

    # Add image tokens
    details["image_tokens"] += image_count * TOKENS_PER_IMAGE
    
    # Calculate total estimated input tokens
    input_tokens = details["text_tokens"] + details["image_tokens"] + details["document_tokens"]

    # Ensure minimum token count if we have uploads but low estimate
    if uploads_data and input_tokens < 100:
        input_tokens = 100

    # Always call Bedrock API
    output_text, usage = bedrock_generate(
        model_id,
        prompt or "",
        uploads_data,
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "512")),
        temperature=float(os.getenv("TEMPERATURE", "0.2"))
    )

    # ALWAYS use Bedrock's actual usage tokens (they are most accurate)
    usage_in = int(usage.get("input_tokens", 0)) if isinstance(usage, dict) else 0
    usage_out = int(usage.get("output_tokens", 0)) if isinstance(usage, dict) else 0
    
    # Use actual usage from Bedrock API response
    if usage_in > 0:
        input_tok = usage_in
        # Update breakdown with actual tokens
        details["actual_input_tokens"] = usage_in
    else:
        input_tok = input_tokens
    
    if usage_out > 0:
        output_tok = usage_out
    else:
        output_tok = estimate_tokens_from_text(output_text) if output_text else estimate_output_tokens_from_input_count(input_tokens or 100)

    in_cost, out_cost, total = price_for_tokens(input_tok, output_tok)

    return {
        "success": True,
        "model_id": model_id,
        "output": output_text,
        "tokens": {
            "input": input_tok,
            "output": output_tok,
            "breakdown": details
        },
        "cost": {
            "input": in_cost,
            "output": out_cost,
            "total_per_request": total,
            "weekly_estimate": round(total * 700, 4),
            "monthly_estimate": round(total * 3000, 4)
        },
        "pricing_per_1k": {"input": INPUT_PRICE_PER_1K, "output": OUTPUT_PRICE_PER_1K}
    }

@app.post("/api/calc")
async def calc_endpoint(request: Request):
    ctype = request.headers.get("content-type","")
    try:
        if "application/json" in ctype:
            body = await request.json()
            prompt = body.get("text_input","") or ""
            model_id = body.get("model_id") or DEFAULT_MODEL
            uploads_data: List[Dict[str, Any]] = []
        else:
            form = await request.form()
            prompt = str(form.get("text_input") or "")
            model_id = str(form.get("model_id") or DEFAULT_MODEL)
            uploads_data = []
            files = form.getlist("files")
            for f in files:
                if isinstance(f, UploadFile):
                    data = await f.read()
                    uploads_data.append({"name": f.filename, "bytes": data})

        if not prompt and not uploads_data:
            raise HTTPException(status_code=400, detail="Provide a prompt or upload at least one file.")
        
        # Debug log
        print(f"DEBUG calc_endpoint: prompt_length={len(prompt)}, uploads_count={len(uploads_data)}")
        if uploads_data:
            for idx, item in enumerate(uploads_data):
                print(f"  Upload {idx}: name={item.get('name')}, size={len(item.get('bytes', b''))} bytes")

        result = _compute_from_payload(model_id, prompt, uploads_data)
        return JSONResponse(status_code=200, content=result)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"success": False, "error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8000")))