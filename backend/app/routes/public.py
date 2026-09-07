"""
Public, unauthenticated routes — the community paper library.

Anyone can browse or download a paper here, no login required. Only papers
explicitly opted-in by their creator (Paper.is_public=True) are ever exposed
here; nothing behind auth leaks through this blueprint.
"""
from flask import Blueprint, request, jsonify, send_file
from sqlalchemy import text
from app.extensions import db
from app.models.paper import Paper

public_bp = Blueprint('public', __name__)


@public_bp.route('/papers', methods=['GET'])
def list_public_papers():
    """List shared papers, newest first. Optional ?subject_id= filter."""
    limit = min(request.args.get('limit', type=int, default=24), 100)
    offset = request.args.get('offset', type=int, default=0)
    subject_id = request.args.get('subject_id', type=int)

    query = Paper.query.filter_by(is_public=True)
    if subject_id:
        query = query.filter_by(subject_id=subject_id)
    query = query.order_by(Paper.created_at.desc())

    total = query.count()
    papers = query.offset(offset).limit(limit).all()

    return jsonify({
        'papers': [p.to_dict() for p in papers],
        'count': total,
    }), 200


@public_bp.route('/papers/<int:paper_id>', methods=['GET'])
def get_public_paper(paper_id):
    """Full detail (incl. questions) for one shared paper."""
    paper = Paper.query.filter_by(id=paper_id, is_public=True).first()
    if not paper:
        return jsonify({'error': 'Not found'}), 404

    paper_data = paper.to_dict()
    paper_data['questions'] = [q.to_dict() for q in paper.questions]
    return jsonify({'paper': paper_data}), 200


@public_bp.route('/papers/<int:paper_id>/pdf', methods=['GET'])
def download_public_paper_pdf(paper_id):
    """Stream the PDF for one shared paper — no auth, no rate limit beyond what the server already has."""
    from app.services.pdf_generator import generate_paper_pdf

    paper = Paper.query.filter_by(id=paper_id, is_public=True).first()
    if not paper:
        return jsonify({'error': 'Not found'}), 404

    paper_data = paper.to_dict()
    paper_data['questions'] = [q.to_dict() for q in paper.questions]
    paper_data['subject_name'] = paper.subject.name if paper.subject else 'Examination'
    paper_data['subject_code'] = paper.subject.code if paper.subject else ''

    pdf_buffer = generate_paper_pdf(paper_data)
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"{paper.title.replace(' ', '_')}.pdf",
    )


# ─────────────────────────────────────────────────────────────────────────
# LIBRARY — search/filter/stats endpoints for the dedicated library page.
#
# Written in raw SQL (JOIN / WHERE / GROUP BY via SQLAlchemy Core `text()`)
# rather than the ORM query builder. All user-supplied values go through
# bound parameters — never string-interpolated into the SQL — so this stays
# injection-safe despite skipping the ORM.
# ─────────────────────────────────────────────────────────────────────────

@public_bp.route('/library/subjects', methods=['GET'])
def library_subjects():
    """Subjects that have at least one public paper."""
    rows = db.session.execute(text("""
        SELECT DISTINCT s.id, s.name, s.code
        FROM subjects s
        JOIN papers p ON p.subject_id = s.id
        WHERE p.is_public = :is_public
        ORDER BY s.name ASC
    """), {'is_public': True}).mappings().all()

    return jsonify({'subjects': [dict(r) for r in rows]}), 200


@public_bp.route('/library/search', methods=['GET'])
def library_search():
    """
    Search/filter the public paper library.

    Query params:
      q             - text search on paper title
      subject_id    - exact subject match
      marks         - exact total_marks match (10/20/40/50/80/100 presets)
      question_type - mcq/short/long; paper must contain at least one
                      question of this type (omit or 'mixed' = no filter)
      limit, offset - pagination (limit capped at 50)
    """
    q = (request.args.get('q') or '').strip()
    subject_id = request.args.get('subject_id', type=int)
    marks = request.args.get('marks', type=int)
    question_type = (request.args.get('question_type') or '').strip().lower()
    limit = min(request.args.get('limit', type=int, default=12), 50)
    offset = max(request.args.get('offset', type=int, default=0), 0)

    conditions = ["p.is_public = :is_public"]
    params = {'is_public': True, 'limit': limit, 'offset': offset}

    if q:
        conditions.append("LOWER(p.title) LIKE LOWER(:q)")
        params['q'] = f"%{q}%"

    if subject_id:
        conditions.append("p.subject_id = :subject_id")
        params['subject_id'] = subject_id

    if marks:
        conditions.append("p.total_marks = :marks")
        params['marks'] = marks

    if question_type and question_type != 'mixed':
        conditions.append("""
            EXISTS (
                SELECT 1 FROM paper_questions pq
                JOIN questions qq ON qq.id = pq.question_id
                WHERE pq.paper_id = p.id AND qq.question_type = :qtype
            )
        """)
        params['qtype'] = question_type

    where_clause = " AND ".join(conditions)

    total = db.session.execute(
        text(f"SELECT COUNT(*) FROM papers p WHERE {where_clause}"),
        params,
    ).scalar_one()

    rows = db.session.execute(text(f"""
        SELECT
            p.id, p.title, p.total_marks, p.duration_minutes, p.status, p.created_at,
            s.id AS subject_id, s.name AS subject_name, s.code AS subject_code,
            u.username AS created_by_username,
            COUNT(pq.question_id) AS question_count
        FROM papers p
        JOIN subjects s ON s.id = p.subject_id
        JOIN users u ON u.id = p.created_by
        LEFT JOIN paper_questions pq ON pq.paper_id = p.id
        WHERE {where_clause}
        GROUP BY p.id, s.id, s.name, s.code, u.username
        ORDER BY p.created_at DESC
        LIMIT :limit OFFSET :offset
    """), params).mappings().all()

    papers = []
    for r in rows:
        d = dict(r)
        # Raw SQL returns a native datetime on Postgres but a plain string on
        # SQLite (no ORM type decoration) — normalize both to ISO text.
        created_at = d.get('created_at')
        d['created_at'] = created_at.isoformat() if hasattr(created_at, 'isoformat') else created_at
        papers.append(d)

    return jsonify({'papers': papers, 'count': total}), 200


@public_bp.route('/library/stats', methods=['GET'])
def library_stats():
    """Aggregate stats for the public library: overall totals + a per-subject GROUP BY breakdown."""
    overall = db.session.execute(text("""
        SELECT
            COUNT(*) AS total_papers,
            COUNT(DISTINCT subject_id) AS total_subjects,
            COALESCE(SUM(total_marks), 0) AS total_marks
        FROM papers
        WHERE is_public = :is_public
    """), {'is_public': True}).mappings().one()

    total_questions = db.session.execute(text("""
        SELECT COUNT(*)
        FROM paper_questions pq
        JOIN papers p ON p.id = pq.paper_id
        WHERE p.is_public = :is_public
    """), {'is_public': True}).scalar_one()

    by_subject = db.session.execute(text("""
        SELECT
            s.id AS subject_id, s.name AS subject_name, s.code AS subject_code,
            COUNT(p.id) AS paper_count,
            COALESCE(SUM(p.total_marks), 0) AS total_marks
        FROM subjects s
        JOIN papers p ON p.subject_id = s.id AND p.is_public = :is_public
        GROUP BY s.id, s.name, s.code
        ORDER BY paper_count DESC
    """), {'is_public': True}).mappings().all()

    return jsonify({
        'overall': {
            'total_papers': overall['total_papers'],
            'total_subjects': overall['total_subjects'],
            'total_marks': overall['total_marks'],
            'total_questions': total_questions,
        },
        'by_subject': [dict(r) for r in by_subject],
    }), 200
