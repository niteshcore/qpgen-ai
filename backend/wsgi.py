"""WSGI entry point for production servers (Render, gunicorn).

Not named app.py on purpose: `app` is already the Flask package in this
directory, so `gunicorn app:app` would import the package instead of a file.
"""
from app import create_app

app = create_app('production')
