from importlib.resources import files
import os

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"

from fastapi import FastAPI, UploadFile, File
import fitz
import io
import numpy as np
import asyncio
import uuid
import json

from fastapi.responses import HTMLResponse
from PIL import Image
from paddleocr import PaddleOCR

from resume_parser import parse_resume
from parser_engine import parse_resume_with_fallback, parse_resumes_with_fallback_batch
from ollama_parser import parse_resume_with_ollama

# Initialize PaddleOCR once
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
 
app = FastAPI(title="CV OCR API")

cv_queue = asyncio.Queue()

queue_results = {}
def save_all_results():
    os.makedirs("results", exist_ok=True)

    with open(
        "results/all_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            list(queue_results.values()),
            f,
            ensure_ascii=False,
            indent=2
        )          
async def cv_worker():

    while True:

        # =========================
        # GET FIRST CV
        # =========================

        first_job = await cv_queue.get()

        batch = [first_job]

        # =========================
        # COLLECT UP TO 4 MORE
        # =========================

        for _ in range(4):

            try:
                job = cv_queue.get_nowait()
                batch.append(job)

            except asyncio.QueueEmpty:
                break

        print(f"\n========== BATCH SIZE: {len(batch)} ==========")

        processed_batch = []

        # =========================
        # OCR EACH CV
        # =========================

        for job_id, filename, file_bytes in batch:

            try:

                print(f"OCR: {filename}")

                extracted_text = extract_text_from_pdf(file_bytes)

                processed_batch.append({
                    "job_id": job_id,
                    "filename": filename,
                    "text": extracted_text
                })

            except Exception as e:

                print(f"OCR failed for {filename}: {e}")

                queue_results[job_id] = {
                    "job_id": job_id,
                    "status": "failed",
                    "filename": filename,
                    "error": str(e)
                }

        if processed_batch:

            try:

                print(
                f"Parsing {len(processed_batch)} CVs "
             "with deterministic local resume parser..."
            )

                batch_results, engine_used = parse_resumes_with_fallback_batch(
                    processed_batch
                )

                for cv, candidate in zip(
                    processed_batch,
                    batch_results
                ):

                    job_id = cv["job_id"]
                    filename = cv["filename"]

                    queue_results[job_id] = {
                        "job_id": job_id,
                        "status": "completed",
                        "filename": filename,
                        "parser": engine_used,
                        "candidate": candidate
                    }

                    print(
                        f"Completed by {engine_used}: "
                        f"{filename} | Job ID: {job_id}"
                    )

            except Exception as e:

                # This only triggers if even the regex fallback blew up
                # (e.g. malformed input), since parse_resumes_with_fallback_batch
                # already tries Groq -> Ollama -> regex internally.
                print(f"All parser tiers failed for this batch: {e}")

                for cv in processed_batch:

                    job_id = cv["job_id"]
                    filename = cv["filename"]

                    try:

                        candidate = parse_resume(
                            cv["text"]
                        )

                        queue_results[job_id] = {
                            "job_id": job_id,
                            "status": "completed",
                            "filename": filename,
                            "parser": "regex_fallback",
                            "candidate": candidate
                        }

                        print(
                            f"Completed by LOCAL: "
                            f"{filename} | Job ID: {job_id}"
                        )

                    except Exception as local_error:

                        queue_results[job_id] = {
                            "job_id": job_id,
                            "status": "failed",
                            "filename": filename,
                            "error": str(local_error)
                        }

        save_all_results()
        
        for _ in batch:
            cv_queue.task_done()
  
def pil_to_paddle_array(image: Image.Image):
    """
    Convert PIL RGB image to BGR numpy array
    for PaddleOCR.
    """
    return np.array(image.convert("RGB"))[:, :, ::-1]


def run_paddle_ocr(image: Image.Image) -> str:

    image_array = pil_to_paddle_array(image)

    result = ocr.predict(image_array)

    texts = []

    for res in result:

        rec_texts = res.get("rec_texts", [])

        if rec_texts:
            texts.extend(rec_texts)

    return "\n".join(texts)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF.

    First tries normal PDF text extraction.
    If a page has no text, it uses PaddleOCR.
    """

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    extracted_text = []

    for page in document:

        # Try normal PDF text extraction first
        page_text = page.get_text()

        if page_text.strip():

            extracted_text.append(page_text)

        else:

            # No selectable text → OCR
            pix = page.get_pixmap(dpi=200)

            image_bytes = pix.tobytes("png")

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            ocr_text = run_paddle_ocr(image)

            extracted_text.append(ocr_text)

    document.close()

    return "\n".join(extracted_text).strip()


def extract_ocr_from_pdf(pdf_bytes: bytes) -> str:
    """
    Force PaddleOCR on every PDF page.
    Used only for testing OCR.
    """

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    all_text = []

    for page in document:

        pix = page.get_pixmap(dpi=200)

        image_bytes = pix.tobytes("png")

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        text = run_paddle_ocr(image)

        all_text.append(text)

    document.close()

    return "\n".join(all_text).strip()


@app.get("/")
def home():

    return {
        "message": "CV OCR API is running"
    }


# =========================================================
# FINAL PIPELINE
# PDF → Text/OCR → Local Resume Parser → JSON
# =========================================================

@app.post("/parse-cv")
async def parse_cv(file: UploadFile = File(...)):

    pdf_bytes = await file.read()

    extracted_text = extract_text_from_pdf(pdf_bytes)

    candidate, engine_used = parse_resume_with_fallback(extracted_text)

    return {
        "filename": file.filename,
        "parser": engine_used,
        "candidate": candidate
    }

@app.post("/local-parse-test")
async def local_parse_test(file: UploadFile = File(...)):

    # Read CV
    file_bytes = await file.read()

    # CV → Text
    extracted_text = extract_text_from_pdf(file_bytes)

    # Local parser only
    candidate = parse_resume(extracted_text)

    return {
        "filename": file.filename,
        "parser": "local",
        "candidate": candidate
    }


@app.post("/ollama-parse-test")
async def ollama_parse_test(file: UploadFile = File(...)):
    """Hits ONLY the local Ollama model, skipping Groq entirely.
    Use this to check the local fallback is set up correctly on its own,
    without needing to exhaust/disable Groq first."""

    file_bytes = await file.read()

    extracted_text = extract_text_from_pdf(file_bytes)

    candidate = parse_resume_with_ollama(extracted_text)

    return {
        "filename": file.filename,
        "parser": "ollama",
        "candidate": candidate
    }
# =========================================================
# OCR TEST
# Supports PDF + JPG + JPEG + PNG
# OCR ONLY — NO RESUME PARSER
# =========================================================

@app.post("/ocr-test")
async def ocr_test(file: UploadFile = File(...)):

    file_bytes = await file.read()

    filename = file.filename.lower()

    # -------------------------
    # PDF
    # -------------------------

    if filename.endswith(".pdf"):

        extracted_text = extract_ocr_from_pdf(file_bytes)

    # -------------------------
    # Image
    # -------------------------

    elif filename.endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

        extracted_text = run_paddle_ocr(image)

    else:

        return {
            "error": "Unsupported file type. Use PDF, JPG, JPEG, PNG or WEBP."
        }

    return {
        "filename": file.filename,
        "ocr_text": extracted_text
    }
   
@app.on_event("startup")
async def startup_event():

    asyncio.create_task(cv_worker())

    print("CV Queue Worker Started") 

from typing import Annotated
@app.post("/queue-cv")
async def queue_cv(file: UploadFile = File(...)):

    file_bytes = await file.read()

    job_id = str(uuid.uuid4())

    queue_results[job_id] = {
        "status": "queued",
        "filename": file.filename
    }

    await cv_queue.put(
        (job_id, file.filename, file_bytes)
    )

    return {
        "message": "CV added to queue",
        "job_id": job_id,
        "filename": file.filename
    }
@app.post("/queue-cvs")
async def queue_cvs(files: list[UploadFile] = File(...)):

    job_ids = []

    for file in files:

        file_bytes = await file.read()

        job_id = str(uuid.uuid4())

        queue_results[job_id] = {
            "status": "queued",
            "filename": file.filename
        }
        save_all_results()

        print("ADDED:", job_id, file.filename)
        print("CURRENT RESULTS:", len(queue_results))

        await cv_queue.put(
            (job_id, file.filename, file_bytes)
        )

        job_ids.append(job_id)

    return {
        "message": f"{len(job_ids)} CVs added to queue",
        "queue_size": cv_queue.qsize(),
        "job_ids": job_ids
    }
@app.get("/queue-results")
async def queue_results_all():

    print("RESULTS REQUESTED:", len(queue_results))

    return {
        "total": len(queue_results),
        "results": list(queue_results.values())
    }