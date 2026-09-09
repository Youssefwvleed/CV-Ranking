"""
Full end-to-end test for the whole project.

Unlike the other test files (which each check ONE piece in isolation),
this one drives the live server through every endpoint, using a real
CV file, and prints a PASS/FAIL summary at the end.

Requirements before running:
    1. The server must already be running:
           uvicorn main:app --reload
    2. A sample CV must exist at the project root, named exactly:
           test_cv.pdf
       (same file test_paddle.py expects)
    3. Ollama should be running with your model pulled, otherwise
       TEST 4 will legitimately FAIL — that's expected if you haven't
       set it up yet, not a bug in the code.

Run:
    python test_full_pipeline.py
"""

import time
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
CV_FILE = Path("test_cv.pdf")

results_summary = []


def record(name, passed, detail=""):
    results_summary.append((name, passed, detail))
    status = "PASSED" if passed else "FAILED"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def read_cv_bytes():
    if not CV_FILE.exists():
        raise FileNotFoundError(
            f"Couldn't find '{CV_FILE}'. Put a sample CV named exactly "
            "'test_cv.pdf' in the project folder before running this test."
        )
    return CV_FILE.read_bytes()


def test_server_up():

    print("\n========== TEST 1: Server is up (/) ==========")

    try:
        response = requests.get(BASE_URL + "/", timeout=5)
        ok = response.status_code == 200
        record("Server health check", ok, f"status={response.status_code}")

    except Exception as e:
        record("Server health check", False, str(e))


def test_ocr():

    print("\n========== TEST 2: OCR only (/ocr-test) ==========")

    try:
        files = {"file": (CV_FILE.name, read_cv_bytes(), "application/pdf")}
        response = requests.post(BASE_URL + "/ocr-test", files=files, timeout=120)

        data = response.json()
        text = data.get("ocr_text", "")

        ok = response.status_code == 200 and len(text.strip()) > 0
        record("OCR extraction", ok, f"{len(text)} characters extracted")

    except Exception as e:
        record("OCR extraction", False, str(e))


def test_local_parser():

    print("\n========== TEST 3: Regex parser only (/local-parse-test) ==========")

    try:
        files = {"file": (CV_FILE.name, read_cv_bytes(), "application/pdf")}
        response = requests.post(BASE_URL + "/local-parse-test", files=files, timeout=120)

        data = response.json()
        candidate = data.get("candidate", {})

        ok = response.status_code == 200 and isinstance(candidate, dict) and bool(candidate)
        record("Regex-only parser", ok, f"parser={data.get('parser')}")

    except Exception as e:
        record("Regex-only parser", False, str(e))


def test_ollama_parser():

    print("\n========== TEST 4: Ollama only (/ollama-parse-test) ==========")

    try:
        files = {"file": (CV_FILE.name, read_cv_bytes(), "application/pdf")}
        response = requests.post(BASE_URL + "/ollama-parse-test", files=files, timeout=180)

        data = response.json()
        candidate = data.get("candidate", {})

        ok = response.status_code == 200 and isinstance(candidate, dict) and bool(candidate)
        record("Ollama-only parser", ok, f"parser={data.get('parser')}")

    except Exception as e:
        record(
            "Ollama-only parser",
            False,
            f"{e} (expected to fail if Ollama isn't running/model not pulled yet)"
        )


def test_full_parse_cv():

    print("\n========== TEST 5: Full pipeline (/parse-cv) ==========")

    try:
        files = {"file": (CV_FILE.name, read_cv_bytes(), "application/pdf")}
        response = requests.post(BASE_URL + "/parse-cv", files=files, timeout=180)

        data = response.json()
        candidate = data.get("candidate", {})
        engine_used = data.get("parser")

        ok = response.status_code == 200 and isinstance(candidate, dict) and bool(candidate)
        record("Full pipeline (/parse-cv)", ok, f"served by: {engine_used}")

    except Exception as e:
        record("Full pipeline (/parse-cv)", False, str(e))


def wait_for_job(job_id, timeout=180):

    start = time.time()

    while time.time() - start < timeout:

        response = requests.get(BASE_URL + "/queue-results", timeout=10)
        entries = response.json().get("results", [])

        for entry in entries:
            if entry.get("job_id") == job_id and entry.get("status") in ("completed", "failed"):
                return entry

        time.sleep(2)

    return None


def test_single_queue():

    print("\n========== TEST 6: Single-file queue (/queue-cv) ==========")

    try:
        files = {"file": (CV_FILE.name, read_cv_bytes(), "application/pdf")}
        response = requests.post(BASE_URL + "/queue-cv", files=files, timeout=30)

        job_id = response.json().get("job_id")
        print(f"Queued job_id={job_id}, waiting for it to finish processing...")

        entry = wait_for_job(job_id)

        ok = entry is not None and entry.get("status") == "completed"
        final_status = entry.get("status") if entry else "TIMED OUT waiting"

        record("Single-file queue", ok, f"final status={final_status}")

    except Exception as e:
        record("Single-file queue", False, str(e))


def test_batch_queue(batch_size=3):

    print(f"\n========== TEST 7: Batch queue of {batch_size} CVs (/queue-cvs) ==========")

    try:
        multipart_files = [
            ("files", (CV_FILE.name, read_cv_bytes(), "application/pdf"))
            for _ in range(batch_size)
        ]

        response = requests.post(BASE_URL + "/queue-cvs", files=multipart_files, timeout=30)
        job_ids = response.json().get("job_ids", [])

        print(f"Queued {len(job_ids)} jobs, waiting for all of them to finish...")

        all_completed = True

        for job_id in job_ids:
            entry = wait_for_job(job_id)
            if not entry or entry.get("status") != "completed":
                all_completed = False

        record("Batch queue", all_completed, f"{len(job_ids)} CVs submitted together")

    except Exception as e:
        record("Batch queue", False, str(e))


def print_summary():

    print("\n\n================= SUMMARY =================")

    for name, passed, detail in results_summary:
        status = "PASSED" if passed else "FAILED"
        print(f"{status:8} | {name}")

    total = len(results_summary)
    passed_count = sum(1 for _, passed, _ in results_summary if passed)

    print(f"\n{passed_count}/{total} tests passed.")

    if passed_count < total:
        print(
            "\nA FAILED test isn't necessarily a bug — e.g. TEST 4 fails "
            "on its own if Ollama isn't running yet, which is fine as long "
            "as TEST 5 still passes (it'll just report a different engine)."
        )


if __name__ == "__main__":

    test_server_up()
    test_ocr()
    test_local_parser()
    test_ollama_parser()
    test_full_parse_cv()
    test_single_queue()
    test_batch_queue()

    print_summary()