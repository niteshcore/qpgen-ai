"""
Daily top-up for the public library: a few AI-written questions per run, across
different subjects, bundled into public papers so the library keeps growing.

    cd backend
    python daily_generate.py --dry-run          # show the plan, call nothing
    python daily_generate.py                     # 2 subjects (or DAILY_SUBJECTS)
    python daily_generate.py --subjects 3

What one run does, per subject picked:
  1. one Gemini call -> a batch of new questions on that day's topic
  2. drop near-duplicates of what is already in the bank (embedding check)
  3. save the rest to the question bank and embed them
  4. bundle them into a paper marked public, owned by the admin account

Subject, topic and paper shape all rotate with the calendar day, so consecutive
runs never repeat and every subject gets its turn. It is deliberately small
(a handful of Gemini calls a day) to stay inside the free-tier quota; if the
quota is hit mid-run it stops cleanly and keeps what it already saved.

Runs against whatever DATABASE_URL points at. Scheduled by
.github/workflows/daily-generate.yml.
"""
import argparse
import logging
import os
import sys
from datetime import date

from app import create_app
from app.extensions import db
from app.models.paper import Paper
from app.models.question import Question
from app.models.subject import Subject
from app.models.user import User
from app.services import embedding_service
from app.services.ai_service import AIService
from app.services.audit_service import log_action

logger = logging.getLogger('daily_generate')

MIN_QUESTIONS_FOR_PAPER = 5

# Topics to rotate through, keyed by subject name. A subject not listed here is skipped.
TOPICS = {
    'Theory of Computation': ['Finite automata and regular expressions', 'Context-free grammars and pushdown automata',
                              'Turing machines', 'Decidability and the halting problem', 'Pumping lemma',
                              'Regular vs context-free languages', 'NP-completeness and reductions', 'Closure properties'],
    'DBMS': ['Normalization and functional dependencies', 'Transactions and ACID properties', 'Indexing and B+ trees',
             'SQL joins and subqueries', 'Concurrency control and locking', 'ER modelling', 'Query optimization',
             'Recovery and logging'],
    'Software Engineering': ['SDLC models', 'Requirements engineering', 'Software testing strategies',
                             'UML and design patterns', 'Agile and Scrum', 'Software metrics and estimation',
                             'Version control and CI/CD', 'Software maintenance'],
    'Computer Networks': ['OSI and TCP/IP models', 'TCP congestion control', 'IP addressing and subnetting',
                          'Routing algorithms', 'DNS and HTTP', 'Data link layer and error detection',
                          'Network security basics', 'Switching and VLANs'],
    'Operating System': ['Process scheduling', 'Deadlocks', 'Memory management and paging', 'Virtual memory',
                         'Synchronization and semaphores', 'File systems', 'Disk scheduling', 'Threads and IPC'],
    'Cyber Security': ['Symmetric and asymmetric cryptography', 'Hashing and digital signatures',
                       'Web vulnerabilities (XSS, SQL injection)', 'Authentication and access control',
                       'Malware and threat models', 'Firewalls and IDS', 'PKI and TLS', 'Security in the SDLC'],
    'COA': ['Instruction set architecture', 'Pipelining and hazards', 'Cache memory', 'Memory hierarchy',
            'Addressing modes', 'Control unit design', 'I/O organization and interrupts', 'Arithmetic and ALU design'],
    'OOPs': ['Encapsulation and abstraction', 'Inheritance and polymorphism', 'Interfaces and abstract classes',
             'Exception handling', 'SOLID principles', 'Constructors and destructors', 'Operator and method overloading',
             'Design patterns'],
    'Data Structures & Algorithms': ['Arrays and linked lists', 'Stacks and queues', 'Trees and BSTs',
                                     'Graphs and traversals', 'Hashing', 'Heaps and priority queues',
                                     'Sorting algorithms', 'Time and space complexity'],
    'Machine Learning': ['Supervised vs unsupervised learning', 'Overfitting and regularization',
                         'Decision trees and ensembles', 'Neural networks and backpropagation',
                         'Clustering', 'Model evaluation metrics', 'Feature engineering', 'Gradient descent'],
    'Design & Analysis of Algorithms': ['Divide and conquer', 'Greedy algorithms', 'Dynamic programming',
                                        'Graph shortest paths', 'Recurrences and the master theorem',
                                        'Backtracking and branch and bound', 'Amortized analysis', 'NP-hard problems'],
    'Web Development': ['HTML semantics and accessibility', 'CSS layout: flexbox and grid', 'JavaScript closures and async',
                        'REST APIs and HTTP', 'DOM and events', 'Authentication with sessions and JWT',
                        'Browser storage and caching', 'Responsive design'],
    'Computer Science': ['Number systems and Boolean algebra', 'Programming paradigms', 'Compilers and interpreters',
                         'Computer architecture basics', 'Recursion', 'Bits, bytes and data representation',
                         'Algorithms and flowcharts', 'Basics of the internet'],
}

# (label, distribution of {marks: how many questions}, duration in minutes)
PAPER_SHAPES = [
    ('Quick Quiz',    {'1': 8},                  15),
    ('Practice Set',  {'1': 4, '3': 3, '5': 2},  45),
    ('Theory Paper',  {'3': 3, '5': 2, '10': 1}, 60),
]


def plan_for(subjects, per_run, today):
    """Pick which (subject, topic, shape) to generate today. Pure function of the date."""
    usable = [s for s in subjects if s.name in TOPICS]
    if not usable:
        return []
    usable.sort(key=lambda s: s.id)
    ordinal = today.toordinal()
    plan = []
    for i in range(min(per_run, len(usable))):
        subject = usable[(ordinal * per_run + i) % len(usable)]
        topics = TOPICS[subject.name]
        topic = topics[(ordinal // len(usable) + i) % len(topics)]
        shape = PAPER_SHAPES[(ordinal + i) % len(PAPER_SHAPES)]
        plan.append((subject, topic, shape))
    return plan


def drop_duplicates(subject_id, generated):
    """Remove questions that already exist in the bank (or repeat each other). Best-effort."""
    try:
        vectors = embedding_service.get_embedder().embed([q['text'] for q in generated])
    except Exception as e:
        logger.warning('Duplicate check skipped (embedding unavailable): %s', e)
        return generated, None

    kept, kept_vectors = [], []
    for q, vec in zip(generated, vectors):
        if embedding_service.find_similar(vec, subject_ids=[subject_id], limit=1,
                                          min_score=embedding_service.DUPLICATE_THRESHOLD):
            logger.info('  skipped duplicate of an existing question: %.70s', q['text'])
            continue
        if any(float(vec @ other) >= embedding_service.DUPLICATE_THRESHOLD for other in kept_vectors):
            continue
        kept.append(q)
        kept_vectors.append(vec)
    return kept, kept_vectors


def generate_one(ai, admin, subject, topic, shape, today, dry_run):
    label, distribution, duration = shape
    title = f'{subject.name} · {topic} {label}'[:200]
    logger.info('%s | %s | %s %s', subject.name, topic, label, distribution)
    if dry_run:
        return True

    if Paper.query.filter_by(title=title, created_by=admin.id).first():
        logger.info('  already generated earlier, skipping')
        return True

    generated = ai.generate_questions_batch(subject_name=subject.name, topic=topic, distribution=distribution)
    generated, vectors = drop_duplicates(subject.id, generated)
    if len(generated) < MIN_QUESTIONS_FOR_PAPER:
        logger.warning('  only %d usable questions left after de-duplication; not building a paper', len(generated))
        return True

    questions = []
    for q in generated:
        try:
            marks = int(q.get('marks') or 1)
        except (TypeError, ValueError):
            marks = 1
        questions.append(Question(
            text=q['text'],
            question_type=q.get('question_type') if q.get('question_type') in Question.QUESTION_TYPES else 'short',
            blooms_level=q.get('blooms_level') if q.get('blooms_level') in Question.BLOOMS_LEVELS else 'understand',
            difficulty=q.get('difficulty') if q.get('difficulty') in Question.DIFFICULTY_LEVELS else 'medium',
            marks=marks,
            option_a=q.get('option_a'), option_b=q.get('option_b'),
            option_c=q.get('option_c'), option_d=q.get('option_d'),
            correct_answer=q.get('correct_answer'),
            topic=topic[:200],
            subject_id=subject.id,
            created_by=admin.id,
        ))
    db.session.add_all(questions)
    db.session.flush()

    paper = Paper(
        title=title,
        total_marks=sum(q.marks for q in questions),
        duration_minutes=duration,
        config={'daily_generated': True, 'topic': topic, 'distribution': distribution,
                'generated_on': today.isoformat()},
        status='final',
        is_public=True,
        subject_id=subject.id,
        created_by=admin.id,
    )
    db.session.add(paper)
    db.session.flush()
    for q in questions:
        paper.questions.append(q)
        q.times_used = 1
    db.session.commit()

    embedding_service.index_questions_best_effort(questions, vectors=vectors)
    log_action('paper.create_daily', resource_type='paper', resource_id=paper.id,
               details={'subject_id': subject.id, 'topic': topic, 'questions': len(questions)}, user_id=admin.id)
    logger.info('  saved paper #%s: %d questions, %d marks', paper.id, len(questions), paper.total_marks)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', type=int, default=int(os.getenv('DAILY_SUBJECTS', 2)),
                        help='how many subjects to generate for this run (default 2)')
    parser.add_argument('--dry-run', action='store_true', help='print the plan without calling Gemini or writing')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    app = create_app('production')
    with app.app_context():
        admin = User.query.filter(db.func.lower(User.email) == os.getenv('ADMIN_EMAIL', 'admin@qpgen.com').lower()).first()
        if not admin:
            sys.exit('Admin account not found; set ADMIN_EMAIL to an existing user.')

        ai = AIService()
        if not args.dry_run and not ai.model:
            sys.exit('GOOGLE_API_KEY is not set.')

        today = date.today()
        plan = plan_for(Subject.query.all(), max(1, args.subjects), today)
        logger.info('Plan for %s (%d subject(s)):', today, len(plan))

        failures = 0
        for subject, topic, shape in plan:
            try:
                generate_one(ai, admin, subject, topic, shape, today, args.dry_run)
            except Exception as e:
                db.session.rollback()
                failures += 1
                logger.error('  failed: %s', e)
                # A quota/rate-limit error will hit every remaining call too: stop instead of hammering the API.
                if any(word in str(e).lower() for word in ('quota', 'rate', '429', 'exhausted')):
                    logger.error('Looks like the Gemini quota; stopping for today.')
                    break

        logger.info('Done. %d failure(s).', failures)
        sys.exit(1 if failures and not args.dry_run else 0)


if __name__ == '__main__':
    main()
