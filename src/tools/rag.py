import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "knowledge"
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".rst"}
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


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


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def iter_text_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []

    return (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
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


class LocalRAGIndex:
    def __init__(self, corpus_path: str | Path = DEFAULT_CORPUS_PATH) -> None:
        self.corpus_path = Path(corpus_path).expanduser().resolve()
        self.chunks = self._load_chunks()
        self.document_frequency = self._build_document_frequency()

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
                        )
                    )
        return loaded

    def _build_document_frequency(self) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            frequency.update(chunk.tokens.keys())
        return frequency

    def search(self, query: str, max_results: int = 5) -> list[RAGResult]:
        query_terms = Counter(tokenize(query))
        if not query_terms or not self.chunks:
            return []

        total_chunks = len(self.chunks)
        scored: list[RAGResult] = []
        for chunk in self.chunks:
            score = 0.0
            for term, query_count in query_terms.items():
                term_frequency = chunk.tokens.get(term, 0)
                if not term_frequency:
                    continue
                idf = math.log((1 + total_chunks) / (1 + self.document_frequency[term])) + 1
                score += query_count * term_frequency * idf

            if score > 0:
                length_penalty = math.sqrt(sum(chunk.tokens.values()))
                scored.append(
                    RAGResult(
                        source=chunk.source,
                        text=chunk.text,
                        score=score / max(length_penalty, 1.0),
                    )
                )

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:max_results]


def format_rag_results(results: list[RAGResult]) -> str:
    if not results:
        return "No relevant passages found in the local knowledge base."

    lines = ["Local knowledge base results:"]
    for index, result in enumerate(results, start=1):
        lines.append(f"\n[{index}] Source: {result.source}")
        lines.append(f"Score: {result.score:.3f}")
        lines.append(result.text)
    return "\n".join(lines)


def rag_search(
    query: str,
    corpus_path: str = str(DEFAULT_CORPUS_PATH),
    max_results: int = 5,
) -> str:
    """Search local knowledge-base documents and return cited relevant passages."""
    index = LocalRAGIndex(corpus_path)
    if not index.chunks:
        return (
            f"No documents found in {index.corpus_path}. "
            "Add .md, .txt, .markdown, or .rst files to use local RAG."
        )

    return format_rag_results(index.search(query, max_results=max_results))


def safe_markdown_filename(filename: str | None, title: str) -> str:
    source = filename or title
    name = Path(source).name.strip().lower()
    name = SAFE_FILENAME_RE.sub("-", name).strip(".-")
    if not name:
        name = "knowledge-note"
    if not name.endswith(".md"):
        name = f"{name}.md"
    return name


def save_to_knowledge(
    title: str,
    content: str,
    filename: str | None = None,
    corpus_path: str = str(DEFAULT_CORPUS_PATH),
) -> str:
    """Save reusable research notes into the local RAG knowledge base."""
    if not title.strip():
        return "Could not save knowledge: title is required."
    if not content.strip():
        return "Could not save knowledge: content is required."

    root = Path(corpus_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_name = safe_markdown_filename(filename, title)
    path = root / safe_name
    body = f"# {title.strip()}\n\n{content.strip()}\n"
    path.write_text(body, encoding="utf-8")
    return f"Saved knowledge document to {path}."
