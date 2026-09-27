from __future__ import annotations

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    """
    Smaller, factoid-friendly chunking.

    - Chunk size: 450 chars, overlap: 90 chars
    - Rich separators cover headings, lists, equations, and page text
    - Light text normalisation before splitting
    - Drops chunks shorter than 80 chars (too small to be meaningful)
    """

    def __init__(self, chunk_size: int = 450, overlap: int = 90):
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n• ",
                "\n- ",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                ", ",
                " ",
                "",
            ],
        )

    def _normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_text(self, text: str) -> list[str]:
        text = self._normalize(text)
        if not text:
            return []

        chunks = self.splitter.split_text(text)

        cleaned: list[str] = []
        for chunk in chunks:
            chunk = self._normalize(chunk)
            if len(chunk) < 80:
                continue
            cleaned.append(chunk)

        return cleaned

    def split_documents(self, docs: list[dict]) -> list[dict]:
        """
        Splits a list of documents while preserving their metadata.
        Each doc should have 'text' and 'page_no'.
        Returns a list of dicts: {'text': chunk_text, 'page_no': original_page_no}
        """
        all_chunks = []
        for doc in docs:
            text = doc.get("text", "")
            page_no = doc.get("page_no")
            
            chunks = self.split_text(text)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "page_no": page_no
                })
        return all_chunks