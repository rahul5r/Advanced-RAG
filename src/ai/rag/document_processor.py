import logging
from pathlib import Path
import io
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        """
        Initializes a simple PDF text extractor using PyMuPDF.
        """
        logger.info("PyMuPDF DocumentProcessor initialised.")

    def process(self, source: str | io.BytesIO, filename: str = "document.pdf") -> str:
        """
        Extracts text from a PDF document.
        Args:
            source: Path to the local file OR an in-memory BytesIO stream.
            filename: Original filename (used for logging).
        Returns:
            str: The extracted plain text.
        """
        try:
            if isinstance(source, io.BytesIO):
                doc = fitz.open(stream=source.read(), filetype="pdf")
            else:
                doc = fitz.open(str(source))

            pages = []

            for i, page in enumerate(doc):
                try:
                    text = page.get_text("text")
                    if text.strip():
                        pages.append({"text": text, "page_no": i + 1})
                except Exception as page_error:
                    logger.warning("Failed to extract page %d: %s", i + 1, page_error)

            doc.close()

            display_name = filename if isinstance(source, io.BytesIO) else source
            logger.info("Successfully extracted %d pages from %s.", len(pages), display_name)
            return pages

        except Exception as e:
            display_name = filename if isinstance(source, io.BytesIO) else source
            logger.error("Failed to process document %s: %s", display_name, e)

            raise


if __name__ == "__main__":
    # uv run -m src.ai.rag.document_processor

    test_file = "data/pdfs/Company-Policy.pdf"
    print("STARTED...")
    processor = DocumentProcessor()
    print("LOADED...")

    try:
        print(f"\n--- Processing: {test_file} ---")
        md_output = processor.process(test_file)

        print("\n--- Text Preview (First 500 chars) ---")
        print(md_output[:500] + "...")
        print("\n--- End of Preview ---")

        output_name = "test_output.txt"
        with open(output_name, "w", encoding="utf-8") as f:
            f.write(md_output)
        print(f"\nFull output saved to {output_name}")

    except Exception as e:
        print(f"Error during test: {e}")