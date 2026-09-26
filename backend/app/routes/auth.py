import re

from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError
from flask_jwt_extended import create_access_token
from app.extensions import db
from app.models.user import User
from app.authorization import require_permission, get_current_user
from app.services.audit_service import log_action

auth_bp = Blueprint('auth', __name__)

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
MIN_PASSWORD_LENGTH = 8


@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}

    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not username or not email or not password:
        return jsonify({'error': 'username, email and password are required'}), 400
    if len(username) > 80:
        return jsonify({'error': 'Username is too long'}), 400
    if not _EMAIL_RE.match(email):
        return jsonify({'error': 'Please enter a valid email address'}), 400
    if len(password) < MIN_PASSWORD_LENGTH:
        return jsonify({'error': f'Password must be at least {MIN_PASSWORD_LENGTH} characters'}), 400

    if User.query.filter(db.func.lower(User.email) == email).first():
        return jsonify({'error': 'Email already registered'}), 409

    if User.query.filter(db.func.lower(User.username) == username.lower()).first():
        return jsonify({'error': 'Username already taken'}), 409

    from app.models.role import Role
    user = User(username=username, email=email)
    user.set_password(password)
    # Every self-registered account gets the open 'user' role — full access to
    # browse subjects/questions and generate/download papers, no bank editing.
    default_role = Role.query.filter_by(name='user').first()
    if default_role:
        user.roles = [default_role]

    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        # Two simultaneous sign-ups with the same email/username slipped past the checks above.
        db.session.rollback()
        return jsonify({'error': 'Email or username already registered'}), 409
    log_action('auth.register', resource_type='user', resource_id=user.id,
               details={'username': user.username}, user_id=user.id)

    return jsonify({
        'message': 'User registered successfully',
        'user': user.to_dict()
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}

    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({'error': 'email and password are required'}), 400

    user = User.query.filter(db.func.lower(User.email) == email).first()

    if not user or not user.check_password(password):
        log_action('auth.login', status='failure', details={'email': email})
        return jsonify({'error': 'Invalid email or password'}), 401

    if user.is_active is False:
        log_action('auth.login', status='failure', user_id=user.id, details={'reason': 'inactive'})
        return jsonify({'error': 'This account is disabled'}), 403

    access_token = create_access_token(identity=str(user.id))
    log_action('auth.login', user_id=user.id)

    return jsonify({
        'message': 'Login successful',
        'access_token': access_token,
        'user': user.to_dict()
    }), 200


@auth_bp.route('/me', methods=['GET'])
@require_permission('users.read_self')
def me():
    """Get current logged-in user info."""
    return jsonify({'user': get_current_user().to_dict()}), 200
