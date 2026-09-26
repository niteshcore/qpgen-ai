from datetime import datetime, timezone

from app.extensions import db
from app.models.question_embedding import EmbeddingVector

CHUNK_EMBEDDING_DIM = 768  # same Gemini model/dim as question_embeddings


class SourceDocument(db.Model):
    """
    A teacher-uploaded PDF (syllabus, notes, textbook chapter) that grounds
    RAG question generation. Chunking + embedding happens in the background
    after upload (see app/services/document_service.py) — `status` tracks
    that so the UI can poll rather than assume it's instantly ready.
    """
    __tablename__ = 'source_documents'

    STATUSES = ['processing', 'ready', 'failed']

    id = db.Column(db.Integer, primary_key=True)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id', ondelete='CASCADE'), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    filename = db.Column(db.String(255), nullable=False)
    title = db.Column(db.String(200), nullable=True)  # teacher-given label, e.g. "Unit 3 — Scheduling"
    page_count = db.Column(db.Integer, nullable=False, default=0)

    status = db.Column(db.String(20), nullable=False, default='processing')
    error_message = db.Column(db.Text, nullable=True)  # populated when status == 'failed'

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    subject = db.relationship('Subject', backref=db.backref('source_documents', lazy=True, cascade='all, delete-orphan'))
    uploader = db.relationship('User', backref=db.backref('uploaded_documents', lazy=True))
    chunks = db.relationship('DocumentChunk', backref='document', lazy=True,
                             cascade='all, delete-orphan', order_by='DocumentChunk.chunk_index')

    def to_dict(self):
        return {
            'id': self.id,
            'subject_id': self.subject_id,
            'subject_name': self.subject.name if self.subject else None,
            'filename': self.filename,
            'title': self.title or self.filename,
            'page_count': self.page_count,
            'chunk_count': len(self.chunks),
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
        }

    def __repr__(self):
        return f'<SourceDocument {self.id}: {self.filename} ({self.status})>'


class DocumentChunk(db.Model):
    """One page-scoped slice of a SourceDocument's extracted text, embedded for retrieval."""
    __tablename__ = 'document_chunks'

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('source_documents.id', ondelete='CASCADE'), nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)  # stable order within the document, 0-based
    page_number = db.Column(db.Integer, nullable=False)  # 1-indexed — chunks never span pages, so this is exact
    text = db.Column(db.Text, nullable=False)

    # Nullable: the row is created with its text first; embedding fills in
    # once the background embed call succeeds (see document_service.process_document).
    embedding = db.Column(EmbeddingVector(CHUNK_EMBEDDING_DIM), nullable=True)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<DocumentChunk doc={self.document_id} #{self.chunk_index} p.{self.page_number}>'
