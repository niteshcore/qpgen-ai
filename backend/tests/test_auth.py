def test_register_success(client):
    response = client.post('/api/auth/register', json={
        'username': 'newuser',
        'email': 'new@test.com',
        'password': 'password123'
    })
    assert response.status_code == 201
    data = response.get_json()
    assert data['message'] == 'User registered successfully'
    assert data['user']['username'] == 'newuser'

def test_register_duplicate_email(client, init_database):
    response = client.post('/api/auth/register', json={
        'username': 'anotheruser',
        'email': 'test@test.com', # Assuming test@test.com is seeded
        'password': 'password123'
    })
    assert response.status_code == 409
    assert response.get_json()['error'] == 'Email already registered'

def test_register_missing_fields(client):
    response = client.post('/api/auth/register', json={
        'username': 'missingemail',
        'password': 'password123'
    })
    assert response.status_code == 400

def test_login_success(client, init_database):
    response = client.post('/api/auth/login', json={
        'email': 'test@test.com',
        'password': 'password123'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'access_token' in data
    assert data['user']['username'] == 'testuser'

def test_login_invalid_password(client, init_database):
    response = client.post('/api/auth/login', json={
        'email': 'test@test.com',
        'password': 'wrongpassword'
    })
    assert response.status_code == 401
    assert response.get_json()['error'] == 'Invalid email or password'

def test_login_nonexistent_user(client):
    response = client.post('/api/auth/login', json={
        'email': 'nobody@test.com',
        'password': 'password123'
    })
    assert response.status_code == 401

def test_me_endpoint_success(client, auth_headers):
    response = client.get('/api/auth/me', headers=auth_headers)
    assert response.status_code == 200
    assert response.get_json()['user']['username'] == 'testuser'

def test_me_endpoint_unauthorized(client):
    response = client.get('/api/auth/me')
    assert response.status_code == 401 # Missing Authorization Header


def test_register_short_password_rejected(client):
    response = client.post('/api/auth/register', json={
        'username': 'shortpw', 'email': 'short@test.com', 'password': 'abc'
    })
    assert response.status_code == 400
    assert 'at least' in response.get_json()['error']


def test_register_invalid_email_rejected(client):
    response = client.post('/api/auth/register', json={
        'username': 'bademail', 'email': 'not-an-email', 'password': 'password123'
    })
    assert response.status_code == 400


def test_register_and_login_are_case_insensitive_on_email(client, init_database):
    dup = client.post('/api/auth/register', json={
        'username': 'caseuser', 'email': 'TEST@Test.com', 'password': 'password123'
    })
    assert dup.status_code == 409  # same address as the seeded test@test.com

    login = client.post('/api/auth/login', json={'email': ' Test@TEST.com ', 'password': 'password123'})
    assert login.status_code == 200


def test_register_and_login_without_json_body_return_400_not_500(client):
    assert client.post('/api/auth/register').status_code == 400
    assert client.post('/api/auth/login').status_code == 400
    assert client.post('/api/auth/login', data='not json', content_type='text/plain').status_code == 400
