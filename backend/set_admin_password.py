"""Change a user's password in whichever database backend/.env points at.

    cd backend
    ./venv/bin/python set_admin_password.py                 # admin@qpgen.com
    ./venv/bin/python set_admin_password.py other@email.com

The new password is typed at a hidden prompt: it is never passed on the
command line, stored in shell history, or printed.
"""
import getpass
import sys

from app import create_app
from app.extensions import db
from app.models.user import User
from app.routes.auth import MIN_PASSWORD_LENGTH

email = (sys.argv[1] if len(sys.argv) > 1 else 'admin@qpgen.com').strip().lower()

app = create_app('production')
with app.app_context():
    user = User.query.filter(db.func.lower(User.email) == email).first()
    if not user:
        sys.exit(f'No user with email {email}')

    print(f'Database: {app.config["SQLALCHEMY_DATABASE_URI"].split("@")[-1]}')
    print(f'Changing password for {user.email} (id {user.id})')
    password = getpass.getpass('New password: ')
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f'Password must be at least {MIN_PASSWORD_LENGTH} characters.')
    if password != getpass.getpass('Repeat it: '):
        sys.exit('Passwords do not match. Nothing changed.')

    user.set_password(password)
    db.session.commit()
    print('Done. The old password no longer works.')
