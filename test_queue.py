import requests
from pathlib import Path

SERVER_URL = "http://127.0.0.1:8000/queue-cvs"

CV_FOLDER = Path(r"C:\Users\DELL\Downloads\archive (3)\data\data\TEACHER")

# Get only the first 20 CV files
cv_files = [
    p for p in CV_FOLDER.iterdir()
    if p.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"]
][:20]

print(f"Found {len(cv_files)} CVs")

multipart_files = []

for file_path in cv_files:
    multipart_files.append(
        (
            "files",
            (
                file_path.name,
                open(file_path, "rb"),
                "application/octet-stream"
            )
        )
    )

try:

    response = requests.post(
        SERVER_URL,
        files=multipart_files
    )

    print("\nStatus:", response.status_code)
    print("Response:")

    try:
        print(response.json())
    except Exception:
        print(response.text)

finally:

    # Close all opened files
    for _, (_, file_object, _) in multipart_files:
        file_object.close()