import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "sandbox"
DEFAULT_VECTOR_STORE_PATH = DEFAULT_CORPUS_PATH / "vector_store"
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".rst"}
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
VECTOR_DIMENSIONS = 512
VECTOR_STORE_FILENAME = "index.json"


@dataclass(frozen=True)
class RAGResult:
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    text: str
    tokens: Counter[str]
    vector: dict[str, float]


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def iter_text_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []

    return (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and DEFAULT_VECTOR_STORE_PATH.name not in path.relative_to(root).parts
    )


def chunk_text(text: str, *, max_words: int = 180, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    step = max(max_words - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def hashed_vector(tokens: Counter[str], dimensions: int = VECTOR_DIMENSIONS) -> dict[str, float]:
    if not tokens:
        return {}

    vector: Counter[int] = Counter()
    for token, count in tokens.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % dimensions
        vector[index] += count

    norm = math.sqrt(sum(value * value for value in vector.values()))
    if not norm:
        return {}
    return {str(index): value / norm for index, value in sorted(vector.items())}


def dot_product(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


class LocalRAGIndex:
    def __init__(self) -> None:
        self.corpus_path = DEFAULT_CORPUS_PATH
        self.vector_store_path = DEFAULT_VECTOR_STORE_PATH
        self.chunks = self._load_chunks()
        self._persist_vector_store()

    def _load_chunks(self) -> list[DocumentChunk]:
        loaded: list[DocumentChunk] = []
        for path in iter_text_files(self.corpus_path):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")

            for chunk in chunk_text(text):
                tokens = Counter(tokenize(chunk))
                if tokens:
                    loaded.append(
                        DocumentChunk(
                            source=str(path.relative_to(self.corpus_path)),
                            text=chunk,
                            tokens=tokens,
                            vector=hashed_vector(tokens),
                        )
                    )
        return loaded

    def search(self, query: str, max_results: int = 5) -> list[RAGResult]:
        query_terms = Counter(tokenize(query))
        if not query_terms or not self.chunks:
            return []

        query_vector = hashed_vector(query_terms)
        scored: list[RAGResult] = []
        for chunk in self.chunks:
            score = dot_product(query_vector, chunk.vector)

            if score > 0:
                scored.append(
                    RAGResult(
                        source=chunk.source,
                        text=chunk.text,
                        score=score,
                    )
                )

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:max_results]

    def _persist_vector_store(self) -> None:
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        index_path = self.vector_store_path / VECTOR_STORE_FILENAME
        payload = {
            "corpus_path": str(self.corpus_path),
            "dimensions": VECTOR_DIMENSIONS,
            "chunks": [
                {
                    "source": chunk.source,
                    "text": chunk.text,
                    "tokens": dict(chunk.tokens),
                    "vector": chunk.vector,
                }
                for chunk in self.chunks
            ],
        }
        index_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def format_rag_results(results: list[RAGResult]) -> str:
    if not results:
        return "No relevant passages found in the sandbox vector index."

    lines = ["Sandbox vector index results:"]
    for index, result in enumerate(results, start=1):
        lines.append(f"\n[{index}] Source: {result.source}")
        lines.append(f"Score: {result.score:.3f}")
        lines.append(result.text)
    return "\n".join(lines)


def rag_search(
    query: str,
    max_results: int = 5,
) -> str:
    """Vectorize sandbox documents, persist the index, and return cited passages."""
    index = LocalRAGIndex()
    if not index.chunks:
        return (
            f"No documents found in {index.corpus_path}. "
            "Add .md, .txt, .markdown, or .rst files to /sandbox to use local RAG."
        )

    return format_rag_results(index.search(query, max_results=max_results))


def safe_markdown_filename(filename: str | None, title: str) -> str:
    source = filename or title
    name = Path(source).name.strip().lower()
    name = SAFE_FILENAME_RE.sub("-", name).strip(".-")
    if not name:
        name = "sandbox-note"
    if not name.endswith(".md"):
        name = f"{name}.md"
    return name


def save_to_knowledge(
    title: str,
    content: str,
    filename: str | None = None,
) -> str:
    """Save reusable research notes into the sandbox RAG corpus."""
    if not title.strip():
        return "Could not save sandbox document: title is required."
    if not content.strip():
        return "Could not save sandbox document: content is required."

    root = DEFAULT_CORPUS_PATH
    root.mkdir(parents=True, exist_ok=True)
    safe_name = safe_markdown_filename(filename, title)
    path = root / safe_name
    body = f"# {title.strip()}\n\n{content.strip()}\n"
    path.write_text(body, encoding="utf-8")
    return f"Saved sandbox document to {path}."
