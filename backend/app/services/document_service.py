"""
PDF ingestion for RAG-grounded question generation.

Upload flow: extract text + chunk synchronously (CPU-only, fast, per page so
citations stay exact) so the document and its chunk rows exist immediately
with status='processing'. Embedding each chunk — the slow, Gemini-network-
bound part — runs in the background, same pattern as
embedding_service.index_questions_background.

The raw PDF bytes are never stored, only the extracted/chunked text: this
project has no object storage configured (kept entirely on free tiers), so
re-uploading is the only way to reprocess a document.
"""
import io
import logging
from concurrent.futures import ThreadPoolExecutor

from pypdf import PdfReader

from app.extensions import db
from app.models.document import SourceDocument, DocumentChunk
from app.services import embedding_service

logger = logging.getLogger(__name__)

# Separate small pool from embedding_service's — document embedding batches
# (many chunks per document) shouldn't starve the per-question indexing pool.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='doc-embed')

MAX_CHUNK_CHARS = 3200   # ~800 tokens at ~4 chars/token, Gemini's usual estimate
OVERLAP_CHARS = 500      # ~125 tokens of overlap between sub-chunks split from one long page


class DocumentProcessingError(Exception):
    """Raised synchronously, e.g. for a PDF with no extractable text — before any row is created."""


def extract_pages(file_bytes):
    """Returns one text string per PDF page, in order (empty string for a blank/image-only page)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    return [(page.extract_text() or '').strip() for page in reader.pages]


def chunk_page(text, max_chars=MAX_CHUNK_CHARS, overlap_chars=OVERLAP_CHARS):
    """
    Splits one page's text into overlapping chunks if it's long; a short page
    is a single chunk. Chunks never cross a page boundary — page citations
    must be exact, and joining across pages risks stitching two unrelated
    sections together right at the cut.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_chars
    return chunks


def create_document(subject_id, uploaded_by, filename, file_bytes, title=None):
    """
    Extracts + chunks synchronously and saves the document with
    status='processing'. Caller schedules embed_document_background()
    afterwards. Raises DocumentProcessingError if nothing extractable was
    found (e.g. a scanned PDF with no text layer) — no row is created.
    """
    try:
        pages = extract_pages(file_bytes)
    except Exception as e:
        raise DocumentProcessingError(f'Could not read this PDF: {e}')

    if not any(p.strip() for p in pages):
        raise DocumentProcessingError(
            'No extractable text found in this PDF — it may be a scanned image with no text layer.'
        )

    document = SourceDocument(
        subject_id=subject_id, uploaded_by=uploaded_by, filename=filename,
        title=title, page_count=len(pages), status='processing',
    )
    db.session.add(document)
    db.session.flush()  # assigns document.id for the chunks' FK

    chunk_index = 0
    for page_number, page_text in enumerate(pages, start=1):
        for piece in chunk_page(page_text):
            db.session.add(DocumentChunk(
                document_id=document.id, chunk_index=chunk_index,
                page_number=page_number, text=piece,
            ))
            chunk_index += 1

    db.session.commit()
    return document


def embed_document(document_id):
    """
    Embeds every not-yet-embedded chunk of a document and flips its status.
    Synchronous — call via embed_document_background() from a route.
    """
    document = db.session.get(SourceDocument, document_id)
    if document is None:
        return

    chunks = (DocumentChunk.query.filter_by(document_id=document_id)
              .order_by(DocumentChunk.chunk_index).all())
    if not chunks:
        document.status = 'failed'
        document.error_message = 'No text chunks were extracted from this file.'
        db.session.commit()
        return

    try:
        # RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY (vs. the SEMANTIC_SIMILARITY pair
        # embedding_service uses for question-to-question dedup) is Gemini's
        # documented pairing for asymmetric search — a short topic string
        # retrieving long passage chunks — which is the shape of this lookup.
        vectors = embedding_service.get_embedder().embed(
            [c.text for c in chunks], task_type='RETRIEVAL_DOCUMENT'
        )
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        document.status = 'ready'
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        document = db.session.get(SourceDocument, document_id)
        document.status = 'failed'
        document.error_message = str(e)[:500]
        db.session.commit()
        logger.error('Embedding failed for document %s: %s', document_id, e, exc_info=True)


def embed_document_background(app, document_id):
    """
    Fire-and-forget: schedules embed_document on a background thread so the
    upload request returns as soon as chunking is done, without waiting on
    however many Gemini embed calls the document needs.

    Runs inline under TESTING for the same reason as
    embedding_service.index_questions_background: TestingConfig's
    sqlite:///:memory: gives a separate thread its own empty database.
    """
    def _run():
        with app.app_context():
            embed_document(document_id)

    if app.config.get('TESTING'):
        _run()
    else:
        _executor.submit(_run)
