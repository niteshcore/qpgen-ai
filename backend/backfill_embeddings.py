"""
Embed every question that has no (or a stale) embedding, then optionally
report near-duplicate pairs already sitting in the bank.

    python backfill_embeddings.py              # embed missing questions
    python backfill_embeddings.py --report     # + duplicate report per subject
    python backfill_embeddings.py --report --threshold 0.88

Gemini's free tier allows 100 embedded texts per minute, so a full run over
~420 questions takes about 5 minutes. Re-runs only embed new/edited questions.

Set APP_CONFIG=production (with DATABASE_URL) to run against Supabase.
"""
import argparse
import os
import time
from collections import defaultdict

import numpy as np

from app import create_app
from app.extensions import db
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding
from app.services import embedding_service


def backfill(chunk_size, pause):
    questions = Question.query.order_by(Question.id).all()
    total = 0
    for start in range(0, len(questions), chunk_size):
        chunk = questions[start:start + chunk_size]
        embedded = embedding_service.index_questions(chunk)
        db.session.commit()  # commit per chunk so a quota error doesn't lose progress
        total += embedded
        print(f'  {min(start + chunk_size, len(questions))}/{len(questions)} checked, {total} embedded')
        # Free tier counts every text as a request (100/min), so pace the chunks.
        if embedded and start + chunk_size < len(questions):
            time.sleep(pause)
    return total


def report(threshold, top):
    rows = (db.session.query(Question, QuestionEmbedding)
            .join(QuestionEmbedding, QuestionEmbedding.question_id == Question.id).all())
    by_subject = defaultdict(list)
    for q, row in rows:
        by_subject[q.subject_id].append((q, row.embedding))

    all_pairs, all_scores = [], []
    for items in by_subject.values():
        if len(items) < 2:
            continue
        matrix = np.stack([emb for _, emb in items])
        sims = matrix @ matrix.T
        i_idx, j_idx = np.triu_indices(len(items), k=1)
        scores = sims[i_idx, j_idx]
        all_scores.append(scores)
        for i, j, s in zip(i_idx, j_idx, scores):
            if s >= threshold:
                all_pairs.append((float(s), items[i][0], items[j][0]))

    if not all_scores:
        print('Not enough embedded questions to report on.')
        return
    scores = np.concatenate(all_scores)
    flagged_ids = {q.id for _, a, b in all_pairs for q in (a, b)}

    print(f'\nEmbedded questions: {len(rows)}  |  same-subject pairs compared: {len(scores)}')
    print('Similarity percentiles: ' + '  '.join(
        f'p{p}={np.percentile(scores, p):.3f}' for p in (50, 90, 99, 99.9)))
    print(f'Pairs >= {threshold}: {len(all_pairs)}  (involving {len(flagged_ids)} questions, '
          f'{len(flagged_ids) / len(rows):.1%} of bank)')

    for s, a, b in sorted(all_pairs, key=lambda p: -p[0])[:top]:
        print(f'\n  {s:.3f}  [{a.subject.name}]')
        print(f'    #{a.id}: {a.text[:110]}')
        print(f'    #{b.id}: {b.text[:110]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true', help='print near-duplicate pairs after embedding')
    parser.add_argument('--threshold', type=float, default=embedding_service.DUPLICATE_THRESHOLD)
    parser.add_argument('--top', type=int, default=15, help='how many pairs to print')
    parser.add_argument('--chunk-size', type=int, default=90, help='texts per chunk (free tier: 100 texts/min)')
    parser.add_argument('--pause', type=float, default=60, help='seconds to wait between chunks')
    args = parser.parse_args()

    app = create_app(os.getenv('APP_CONFIG', 'development'))
    with app.app_context():
        db.engine.echo = False
        # Backfill is offline, so allow more patient retries on free-tier rate limits.
        embedding_service.set_embedder(embedding_service.GeminiEmbedder(max_attempts=6))
        print('Embedding questions...')
        print(f'Done: {backfill(args.chunk_size, args.pause)} newly embedded.')
        if args.report:
            report(args.threshold, args.top)
