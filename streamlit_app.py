# streamlit_app.py
# Sonnet 4 Bedrock price calculator + LIVE model call that works for text, images, and PDFs
# Key fix: read uploaded files ONCE with getvalue() and reuse bytes (no empty payloads)

import io
import os
import json
import base64
import traceback
import streamlit as st

# ---- Pricing & simple heuristics ----
INPUT_PRICE_PER_1K = 0.003
OUTPUT_PRICE_PER_1K = 0.015
TOKENS_PER_IMAGE = 1600
TOKENS_PER_PAGE = 500

MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
ANTHROPIC_VERSION = "bedrock-2023-05-31"  # Anthropic Messages API version for Bedrock

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
    return "image/jpeg"

def _extract_pdf_pages(file_bytes: bytes) -> int:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return len(reader.pages)
    except Exception:
        return 1

def _extract_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _build_bedrock_messages(prompt: str, uploads_data):
    """
    Build Anthropic Messages API payload: one user message with text + image blocks.
    PDFs/TXTs are appended as text context (simple approach).
    uploads_data: list of {"name": str, "bytes": bytes}
    """
    content = []
    if prompt.strip():
        content.append({"type": "text", "text": prompt.strip()})

    doc_text_chunks = []
    image_blocks = 0

    for item in (uploads_data or []):
        name = item["name"]
        data = item["bytes"]
        if not data:
            continue

        if _is_image_name(name):
            b64 = base64.b64encode(data).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": _mimetype_for_image(name), "data": b64}
            })
            image_blocks += 1
        elif name.lower().endswith(".pdf"):
            pages = _extract_pdf_pages(data)
            doc_text_chunks.append(f"[PDF attached: {pages} page(s). Summarize/analyze as requested.]")
        elif name.lower().endswith(".txt"):
            txt = _extract_txt(data)
            if txt:
                doc_text_chunks.append(f"[TXT attached snippet]\n{txt[:8000]}")
        else:
            doc_text_chunks.append(f"[File attached: {name}]")

    if doc_text_chunks:
        content.append({"type": "text", "text": "\n\n".join(doc_text_chunks)})

    return [{"role": "user", "content": content}], image_blocks

def _bedrock_generate(model_id: str, prompt: str, uploads_data, max_tokens: int = 512, temperature: float = 0.2):
    import boto3
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    )
    messages, image_blocks = _build_bedrock_messages(prompt, uploads_data)

    body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }

    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    payload = json.loads(resp["body"].read())
    output_text = ""
    for block in payload.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            output_text += block.get("text", "")
    usage = payload.get("usage", {})  # may include 'input_tokens' and 'output_tokens'
    return output_text.strip(), usage, image_blocks

# ---- UI ----
st.set_page_config(page_title="Sonnet 4 — Cost Calculator (Bedrock-ready)", page_icon="💵", layout="centered")
st.title("Sonnet 4 — Bedrock Cost Calculator")

with st.expander("Settings", expanded=False):
    model_id_ui = st.text_input("Model ID", value=MODEL_ID)
    max_tokens = st.number_input("Max output tokens", min_value=50, max_value=8192, value=512, step=50)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    call_bedrock = st.checkbox("Call Bedrock to generate output", value=False, help="Requires AWS credentials + bedrock:InvokeModel")

prompt = st.text_area("Prompt", height=160, placeholder="Ask about the uploaded image/doc, or enter any prompt…")

uploads = st.file_uploader(
    "Upload (images / .pdf / .txt)",
    accept_multiple_files=True,
    type=["png","jpg","jpeg","gif","webp","pdf","txt"]
)

error_box = st.empty()
run = st.button("Calculate", use_container_width=True)

if run:
    try:
        # -------- Read uploads ONCE and reuse bytes --------
        uploads_data = []
        if uploads:
            for f in uploads:
                uploads_data.append({"name": f.name, "bytes": f.getvalue()})  # getvalue() works even if read before

        # ---- Token estimation from uploads ----
        details = {"text_tokens": 0, "image_tokens": 0, "document_tokens": 0}
        input_tokens = 0

        pdf_pages_total = 0
        image_count = 0
        for item in uploads_data:
            name = item["name"].lower()
            data = item["bytes"]
            if not data:
                continue
            if _is_image_name(name):
                image_count += 1
            elif name.endswith(".pdf"):
                pdf_pages_total += _extract_pdf_pages(data)
            elif name.endswith(".txt"):
                txt = _extract_txt(data)
                details["text_tokens"] += estimate_tokens_from_text(txt)
            else:
                pdf_pages_total += 1

        details["image_tokens"] += image_count * TOKENS_PER_IMAGE
        details["document_tokens"] += pdf_pages_total * TOKENS_PER_PAGE
        input_tokens = details["text_tokens"] + details["image_tokens"] + details["document_tokens"]

        # Prompt-only case
        if prompt and not uploads_data:
            details["text_tokens"] = estimate_tokens_from_text(prompt)
            input_tokens = details["text_tokens"]

        # ---- Produce / choose output ----
        output_text = ""
        usage = {}
        model_id = model_id_ui or MODEL_ID

        if call_bedrock:
            output_text, usage, img_blocks = _bedrock_generate(model_id, prompt, uploads_data, max_tokens, temperature)

            st.subheader("Model Output")
            if img_blocks == 0 and any(_is_image_name(x["name"]) for x in uploads_data):
                st.warning("No image blocks were attached to the request. (This should not happen now. If it does, check file types.)")
            st.write(output_text if output_text else "_(No text content returned)_")
        else:
            # Manual mode
            output_text = output_text_manual.strip()
            if output_text:
                st.subheader("Output response (from your pasted text)")
                st.write(output_text)
            else:
                st.subheader("Output response (estimated length; no real generation)")
                st.info("No actual model call here; output tokens were estimated from the input size. "
                        "Enable 'Call Bedrock to generate output' or paste a real output above to use its exact count.")

        # ---- Token counts (prefer Bedrock usage if present) ----
        if usage and "input_tokens" in usage and "output_tokens" in usage:
            input_tok = int(usage.get("input_tokens", input_tokens))
            output_tok = int(usage.get("output_tokens", estimate_output_tokens_from_input_count(input_tokens or 100)))
        else:
            input_tok = input_tokens
            if output_text:
                output_tok = estimate_tokens_from_text(output_text)
            else:
                output_tok = estimate_output_tokens_from_input_count(input_tokens or 100)

        in_cost, out_cost, total = price_for_tokens(input_tok, output_tok)

        st.subheader("Tokens")
        col1, col2, col3 = st.columns(3)
        col1.metric("Input tokens", f"{input_tok}")
        col2.metric("Output tokens", f"{output_tok}")
        col3.metric("Total tokens", f"{input_tok + output_tok}")

        st.subheader("Cost")
        col4, col5, col6 = st.columns(3)
        col4.metric("Input cost", f"${in_cost:.6f}")
        col5.metric("Output cost", f"${out_cost:.6f}")
        col6.metric("Total cost", f"${total:.6f}")

        st.subheader("Projections (100 requests/day)")
        col7, col8, col9 = st.columns(3)
        col7.metric("Daily", f"${total*100:.4f}")
        col8.metric("Weekly", f"${total*700:.2f}")
        col9.metric("Monthly", f"${total*3000:.2f}")

        with st.expander("Details"):
            st.json({
                "model_id": model_id,
                "pricing_per_1k": {"input": INPUT_PRICE_PER_1K, "output": OUTPUT_PRICE_PER_1K},
                "usage_from_bedrock": usage or "(not available; used estimations)",
                "breakdown_estimated": details
            })

    except Exception as e:
        tb = traceback.format_exc()
        error_box.error(f"Error: {e}")
        with st.expander("Traceback (for debugging)"):
            st.code(tb, language="python")
