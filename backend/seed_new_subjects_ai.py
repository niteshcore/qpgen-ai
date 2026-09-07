"""
Adds 4 new subjects and seeds 30-40 REAL Gemini-generated questions each,
spread across Bloom's levels and marks values — so the bank doesn't look thin.

Uses the app's own AIService.generate_questions_batch() (the same pipeline
the "AI Generate" paper flow uses), inserted via the SQLAlchemy ORM — no
hand-written fake questions, no raw DB rows bypassing the app's models.

Requires GOOGLE_API_KEY in backend/.env. Run from backend/:
    python seed_new_subjects_ai.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.extensions import db
from app.models.subject import Subject
from app.models.question import Question
from app.models.user import User
from app.services.ai_service import AIService

NEW_SUBJECTS = [
    ("Data Structures & Algorithms", "DSA",
     "Arrays, linked lists, trees, graphs, sorting, searching, and algorithmic complexity"),
    ("Machine Learning", "ML",
     "Supervised/unsupervised learning, model evaluation, and core ML algorithms"),
    ("Design & Analysis of Algorithms", "DAA",
     "Algorithm design paradigms, complexity analysis, and optimization techniques"),
    ("Web Development", "WEB",
     "Frontend/backend fundamentals, HTTP, REST APIs, and web application architecture"),
]

# 3 topic batches (marks 1/3/5) covering the subject's breadth, plus one
# batch of 10-mark long questions so each subject has depth at every marks
# value the paper generator actually uses.
TOPIC_BATCHES = {
    "DSA": [
        ("Arrays, Linked Lists, Stacks, and Queues", {"1": 5, "3": 3, "5": 2}),
        ("Trees, Graphs, and Traversal Algorithms", {"1": 5, "3": 3, "5": 2}),
        ("Sorting, Searching, and Hashing", {"1": 4, "3": 3, "5": 2}),
        ("Advanced Data Structure Design Problems", {"10": 2}),
    ],
    "ML": [
        ("Supervised Learning: Regression and Classification", {"1": 5, "3": 3, "5": 2}),
        ("Unsupervised Learning and Model Evaluation", {"1": 5, "3": 3, "5": 2}),
        ("Neural Networks, Overfitting, and Regularization", {"1": 4, "3": 3, "5": 2}),
        ("End-to-End ML Pipeline Design Problems", {"10": 2}),
    ],
    "DAA": [
        ("Divide and Conquer and Greedy Algorithms", {"1": 5, "3": 3, "5": 2}),
        ("Dynamic Programming", {"1": 5, "3": 3, "5": 2}),
        ("Graph Algorithms and NP-Completeness", {"1": 4, "3": 3, "5": 2}),
        ("Algorithm Design Case Studies", {"10": 2}),
    ],
    "WEB": [
        ("HTTP, REST APIs, and Client-Server Architecture", {"1": 5, "3": 3, "5": 2}),
        ("Frontend Fundamentals: DOM, CSS, and State", {"1": 5, "3": 3, "5": 2}),
        ("Authentication, Sessions, and Web Security", {"1": 4, "3": 3, "5": 2}),
        ("Full-Stack Application Design Problems", {"10": 2}),
    ],
}


def main():
    app = create_app('development')
    with app.app_context():
        ai = AIService()
        if not ai.api_key:
            print("ERROR: GOOGLE_API_KEY not configured in backend/.env. Aborting.")
            return

        admin = User.query.filter_by(email='admin@qpgen.com').first()
        if not admin:
            print("ERROR: admin user (admin@qpgen.com) not found. Aborting.")
            return

        for name, code, desc in NEW_SUBJECTS:
            subject = Subject.query.filter_by(code=code).first()
            if subject:
                print(f"[{code}] subject already exists (id={subject.id})")
            else:
                subject = Subject(name=name, code=code, description=desc)
                db.session.add(subject)
                db.session.commit()
                print(f"[{code}] created subject '{name}' -> id={subject.id}")

            existing_count = Question.query.filter_by(subject_id=subject.id).count()
            if existing_count >= 30:
                print(f"[{code}] already has {existing_count} questions, skipping generation.")
                continue

            total_added = 0
            for topic, distribution in TOPIC_BATCHES[code]:
                print(f"[{code}] generating batch: '{topic}' {distribution} ...")
                try:
                    questions = ai.generate_questions_batch(
                        subject_name=name, topic=topic, distribution=distribution,
                    )
                except Exception as e:
                    print(f"[{code}]   FAILED: {e}")
                    continue

                for q in questions:
                    question = Question(
                        text=q['text'],
                        question_type=q.get('question_type', 'short'),
                        blooms_level=q.get('blooms_level', 'understand'),
                        difficulty=q.get('difficulty', 'medium'),
                        marks=q.get('marks', 1),
                        option_a=q.get('option_a'),
                        option_b=q.get('option_b'),
                        option_c=q.get('option_c'),
                        option_d=q.get('option_d'),
                        correct_answer=q.get('correct_answer'),
                        subject_id=subject.id,
                        created_by=admin.id,
                    )
                    db.session.add(question)
                    total_added += 1
                db.session.commit()
                print(f"[{code}]   +{len(questions)} questions (running total this run: {total_added})")
                time.sleep(2)  # be polite to the API between batches

            final_count = Question.query.filter_by(subject_id=subject.id).count()
            print(f"[{code}] done — {final_count} questions total.\n")

        print("All subjects processed.")


if __name__ == '__main__':
    main()
