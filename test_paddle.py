import fitz
import io

from PIL import Image
from paddleocr import PaddleOCR


ocr = PaddleOCR(lang="en")


def extract_ocr_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    all_text = ""

    for page_number, page in enumerate(document):

        print(f"\n--- Processing page {page_number + 1} ---")

        # Render PDF page as image
        pix = page.get_pixmap(dpi=200)

        image_bytes = pix.tobytes("png")

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # PaddleOCR
        result = ocr.predict(image)

        page_text = []

        for res in result:

            texts = res.get("rec_texts", [])

            page_text.extend(texts)

        page_text = "\n".join(page_text)

        print(page_text)

        all_text += page_text + "\n"

    document.close()

    return all_text


if __name__ == "__main__":

    text = extract_ocr_from_pdf("test_cv.pdf")

    print("\n\n==============================")
    print("FINAL OCR OUTPUT")
    print("==============================\n")

    print(text)