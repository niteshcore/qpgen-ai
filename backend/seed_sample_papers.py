"""
Generates ~30 sample papers through the REAL running app (HTTP calls against
a live server), across a spread of subjects, marks (10/20/40/50/80/100), and
question types (MCQ/Short/Long/Mixed), then marks each public so the
Community Library and landing page look populated with real content.

No fake rows are inserted — every paper goes through the actual
/api/papers/generate or /api/papers/generate-ai-syllabus endpoints, the same
ones a real user hits from the UI. The AI-syllabus combos call real Gemini.

Requires the Flask server already running at BASE_URL. Run from backend/:
    python seed_sample_papers.py
"""
import requests
import time

BASE_URL = "http://127.0.0.1:5000/api"

ADMIN = {"email": "admin@qpgen.com", "password": "admin123"}
PUBLIC_USER = {"email": "publicuser1@example.com", "password": "testpass123"}

DEFAULT_BLOOMS = {"remember": 20, "understand": 30, "apply": 30, "analyze": 10, "evaluate": 5, "create": 5}
DEFAULT_DIFFICULTY = {"easy": 30, "medium": 50, "hard": 20}

# duration scales loosely with marks
def duration_for(marks):
    return {10: 30, 20: 45, 40: 90, 50: 120, 80: 150, 100: 180}.get(marks, 90)


# ── Combos ──────────────────────────────────────────────────────────────────
# mode 'quick'  -> POST /api/papers/generate      (bank pull, supports exact question_type)
# mode 'ai'     -> POST /api/papers/generate-ai-syllabus (real Gemini, always a natural mix)
#
# AI-mode "marks" is achieved via an explicit distribution dict (mark_value: count)
# summing exactly to the target, since that endpoint has no total_marks input.

QUICK_COMBOS = [
    ("TOC",  "TOC Quick Quiz",                 10, "mcq"),
    ("DBMS", "DBMS Fundamentals Test",         20, "mixed"),
    ("SE",   "SE Midterm",                     40, "short"),
    ("CN",   "Networks Practice Paper",        20, "mcq"),
    ("OS",   "OS Concepts Quiz",               10, "mcq"),
    ("CS",   "Cybersecurity Basics",           20, "short"),
    ("COA",  "COA Unit Test",                  40, "mixed"),
    ("OOPS", "OOP Concepts Quiz",              10, "mcq"),
    ("DSA",  "DSA Practice Set",               50, "mixed"),
    ("ML",   "ML Fundamentals Quiz",           20, "mcq"),
    ("DAA",  "Algorithm Design Test",          40, "short"),
    ("WEB",  "Web Dev Basics",                 20, "mcq"),
    ("TOC",  "Automata Theory Long Answers",   40, "long"),
    ("DBMS", "Normalization Deep Dive",        50, "long"),
    ("SE",   "SDLC Quick Test",                10, "mcq"),
    ("CN",   "OSI Model Mixed Paper",          40, "mixed"),
    ("OS",   "Deadlocks and Memory Mgmt",      50, "mixed"),
    ("DSA",  "Trees and Graphs Quiz",          20, "mcq"),
]

AI_COMBOS = [
    ("DBMS", "DBMS Semester Paper",     "Normalization and Transaction Management",                 {"1": 10, "5": 6}),
    ("TOC",  "Automata Theory Test",    "Finite Automata and Regular Languages",                     {"5": 4}),
    ("SE",   "SE Full Paper",           "SDLC Models and Agile Practices",                           {"1": 10, "5": 8}),
    ("CN",   "Networks End-Term",       "TCP/IP, Routing, and Network Security",                     {"1": 10, "5": 10, "10": 2}),
    ("OS",   "OS End-Term",             "Process Scheduling, Deadlocks, and Memory Management",      {"1": 10, "5": 10, "10": 4}),
    ("CS",   "Cybersecurity Test",      "Cryptography and Network Attacks",                          {"1": 10, "5": 6}),
    ("COA",  "COA Quiz",                "Instruction Cycle and Pipelining",                          {"5": 4}),
    ("OOPS", "OOP Full Paper",          "Inheritance, Polymorphism, and Design Patterns",            {"1": 10, "5": 8}),
    ("ML",   "ML End-Term",             "Supervised Learning, Model Evaluation, and Neural Networks", {"1": 10, "5": 10, "10": 2}),
    ("DAA",  "Algorithms End-Term",     "Dynamic Programming, Greedy Algorithms, and NP-Completeness", {"1": 10, "5": 10, "10": 4}),
    ("WEB",  "Web Dev Quiz",            "REST APIs and HTTP Fundamentals",                           {"5": 2}),
    ("DSA",  "DSA Semester Paper",      "Sorting Algorithms and Hashing",                            {"1": 10, "5": 6}),
]


def login(creds):
    r = requests.post(f"{BASE_URL}/auth/login", json=creds, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def get_subjects(token):
    r = requests.get(f"{BASE_URL}/subjects/", headers={"Authorization": f"Bearer {token}"}, timeout=10)
    r.raise_for_status()
    return {s["code"]: s["id"] for s in r.json()["subjects"]}


def make_public(token, paper_id):
    r = requests.put(
        f"{BASE_URL}/papers/{paper_id}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"is_public": True},
        timeout=10,
    )
    r.raise_for_status()


def run_quick(token, subjects, code, title, marks, qtype):
    payload = {
        "title": title,
        "subject_id": subjects[code],
        "duration_minutes": duration_for(marks),
        "total_marks": marks,
        "config": {
            "question_type": qtype,
            "blooms_distribution": DEFAULT_BLOOMS,
            "difficulty_distribution": DEFAULT_DIFFICULTY,
        },
    }
    r = requests.post(
        f"{BASE_URL}/papers/generate",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload, timeout=30,
    )
    if r.status_code != 201:
        raise RuntimeError(r.json().get("error", r.text))
    return r.json()["paper"]["id"]


def run_ai(token, subjects, code, title, syllabus, distribution):
    total_marks = sum(int(m) * c for m, c in distribution.items())
    payload = {
        "title": title,
        "subject_id": subjects[code],
        "syllabus": syllabus,
        "distribution": distribution,
        "duration_minutes": duration_for(total_marks),
    }
    r = requests.post(
        f"{BASE_URL}/papers/generate-ai-syllabus",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload, timeout=90,
    )
    if r.status_code != 201:
        raise RuntimeError(r.json().get("error", r.text))
    return r.json()["paper"]["id"]


def gemini_quota_available():
    """Cheap pre-check so we don't burn the AI combos through 3x-retried, doomed calls."""
    import os
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        genai.GenerativeModel("gemini-flash-latest").generate_content("Say OK")
        return True
    except Exception as e:
        print(f"Gemini quota check failed ({e.__class__.__name__}) — skipping AI-syllabus combos this run.\n")
        return False


def main():
    admin_token = login(ADMIN)
    user_token = login(PUBLIC_USER)
    subjects = get_subjects(admin_token)
    print(f"Found {len(subjects)} subjects: {sorted(subjects.keys())}\n")

    ok, failed = 0, 0

    for i, (code, title, marks, qtype) in enumerate(QUICK_COMBOS):
        if code not in subjects:
            print(f"SKIP quick '{title}' — subject {code} not found")
            continue
        token = admin_token if i % 2 == 0 else user_token
        try:
            pid = run_quick(token, subjects, code, title, marks, qtype)
            make_public(token, pid)
            print(f"OK   [quick] {title} ({code}, {marks}m, {qtype}) -> paper #{pid}, public")
            ok += 1
        except Exception as e:
            print(f"FAIL [quick] {title} ({code}, {marks}m, {qtype}): {e}")
            failed += 1

    if gemini_quota_available():
        for i, (code, title, syllabus, distribution) in enumerate(AI_COMBOS):
            if code not in subjects:
                print(f"SKIP ai '{title}' — subject {code} not found")
                continue
            token = admin_token if i % 2 == 0 else user_token
            try:
                pid = run_ai(token, subjects, code, title, syllabus, distribution)
                make_public(token, pid)
                print(f"OK   [ai]    {title} ({code}) -> paper #{pid}, public")
                ok += 1
            except Exception as e:
                print(f"FAIL [ai]    {title} ({code}): {e}")
                failed += 1
            time.sleep(1)
    else:
        print(f"Skipped all {len(AI_COMBOS)} AI-syllabus combos (Gemini daily quota exhausted). "
              f"Re-run this script after quota resets to fill those in.")

    print(f"\nDone. {ok} papers created and shared publicly, {failed} failed.")


if __name__ == "__main__":
    main()
