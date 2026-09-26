"""
Retrieval-augmented question generation: given a topic, fetch the most
relevant chunks from uploaded course material and require Gemini to write
questions strictly from those passages, citing which one grounds each
question.

The citation is never taken on the model's word — it names one of the
passage numbers we ourselves assigned in the prompt, and we look up that
passage's real page number ourselves. A hallucinated or out-of-range
citation is dropped (question kept, source left null) rather than trusted.
"""
import json
import logging
import re

import numpy as np
import sqlalchemy as sa

from app.extensions import db
from app.models.document import SourceDocument, DocumentChunk
from app.models.question_embedding import EmbeddingVector
from app.services import embedding_service

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHUNKS = 8


class RagUnavailable(Exception):
    """No ready, embedded chunks to ground generation in."""


def retrieve_chunks(subject_id, query_text, document_id=None, top_k=MAX_CONTEXT_CHUNKS):
    """
    Returns the top_k DocumentChunk rows most relevant to query_text, most
    relevant first. Restricted to `status='ready'` documents (still-
    processing or failed documents have no, or incomplete, embeddings) in
    `subject_id`, optionally narrowed to one `document_id`.
    """
    query_vector = embedding_service.get_embedder().embed([query_text], task_type='RETRIEVAL_QUERY')[0]

    base_query = (db.session.query(DocumentChunk)
                  .join(SourceDocument, SourceDocument.id == DocumentChunk.document_id)
                  .filter(SourceDocument.subject_id == subject_id, SourceDocument.status == 'ready'))
    if document_id is not None:
        base_query = base_query.filter(DocumentChunk.document_id == document_id)

    if db.engine.dialect.name == 'postgresql':
        qvec = sa.bindparam('qvec', value=list(map(float, query_vector)),
                            type_=EmbeddingVector(embedding_service.EMBEDDING_DIM))
        distance = DocumentChunk.embedding.op('<=>', return_type=sa.Float)(qvec)
        chunks = base_query.order_by(distance).limit(top_k).all()
    else:
        rows = base_query.filter(DocumentChunk.embedding.isnot(None)).all()
        if not rows:
            return []
        matrix = np.stack([r.embedding for r in rows])
        scores = matrix @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        chunks = [rows[i] for i in order]

    return chunks


_JSON_ARRAY_RE = re.compile(r'```(?:json)?\s*(\[.*?\])\s*```|(\[.*\])', re.DOTALL)


def _extract_json_array(content):
    match = _JSON_ARRAY_RE.search(content)
    if not match:
        raise ValueError('No JSON array found in AI response')
    return json.loads(match.group(1) or match.group(2))


def _call_gemini(prompt, difficulty='medium'):
    """
    Thin, deliberately separate seam around AIService's text generation:
    real code goes through here (reusing its retry/temperature logic rather
    than duplicating it), and tests monkeypatch this one function instead
    of needing a live Gemini call or a parallel fake-model class.
    """
    from app.services.ai_service import AIService
    ai_service = AIService()
    if not ai_service.model:
        raise ValueError('Gemini API Key is not configured.')
    response = ai_service._call_with_retry(prompt, difficulty=difficulty)
    if not response or not response.text:
        raise ValueError('Empty response from Gemini')
    return response.text


def generate_grounded_questions(subject_name, chunks, question_type='short', difficulty='medium',
                                marks=3, count=3, focus=None):
    """
    Generates `count` questions grounded only in `chunks`. Returns a list of
    dicts shaped like the rest of the AI-generation pipeline (text,
    question_type, blooms_level, difficulty, marks, options/correct_answer
    for MCQ) plus `source_page` and `_source_chunk_id` (internal — the route
    uses it to set Question.source_document_id/source_page; never persisted
    itself).
    """
    if not chunks:
        raise RagUnavailable('No embedded course material available for this subject/document yet.')

    passages = '\n\n'.join(
        f'[PASSAGE {i + 1} — page {c.page_number}]\n{c.text}' for i, c in enumerate(chunks)
    )
    focus_line = f"Focus especially on: {focus}\n" if focus else ''

    prompt = f"""
You are an expert professor writing exam questions for '{subject_name}'.

You are given numbered passages from the course's own material below. Every
question you write MUST be answerable using ONLY these passages — do not
use outside knowledge, and do not invent facts, numbers, or terms that
aren't in the text.

{passages}

{focus_line}
Write {count} question(s) of type "{question_type}", difficulty "{difficulty}",
worth {marks} marks each.

For each question, include "source_passage": <the integer number of the ONE
passage above it is most directly based on>. This must be one of the passage
numbers shown (1 to {len(chunks)}).

Format the output as a JSON array. Each object:
{{
    "text": "the question",
    "question_type": "{question_type}",
    "blooms_level": "one of: remember, understand, apply, analyze, evaluate, create",
    "difficulty": "{difficulty}",
    "marks": {marks},
    "option_a": "only if MCQ, else null",
    "option_b": "only if MCQ, else null",
    "option_c": "only if MCQ, else null",
    "option_d": "only if MCQ, else null",
    "correct_answer": "correct option or model answer",
    "source_passage": 1
}}

Return ONLY the JSON array. No extra text or markdown.
"""

    content = _call_gemini(prompt, difficulty=difficulty)
    raw_questions = _extract_json_array(content)
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError('AI did not return a JSON array of questions')

    results = []
    for q in raw_questions:
        if 'text' not in q:
            continue
        passage_num = q.get('source_passage')
        chunk = None
        if isinstance(passage_num, int) and 1 <= passage_num <= len(chunks):
            chunk = chunks[passage_num - 1]
        else:
            logger.warning('Grounded question missing/invalid source_passage (%r); saving without a citation', passage_num)

        results.append({
            'text': q['text'],
            'question_type': q.get('question_type', question_type),
            'blooms_level': q.get('blooms_level', 'understand'),
            'difficulty': q.get('difficulty', difficulty),
            'marks': q.get('marks', marks),
            'option_a': q.get('option_a'), 'option_b': q.get('option_b'),
            'option_c': q.get('option_c'), 'option_d': q.get('option_d'),
            'correct_answer': q.get('correct_answer'),
            'source_page': chunk.page_number if chunk else None,
            '_source_chunk_id': chunk.id if chunk else None,
            '_source_document_id': chunk.document_id if chunk else None,
        })
    return results
