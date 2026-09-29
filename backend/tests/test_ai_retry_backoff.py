"""ai_service's retry backoff: must actually wait between attempts, and respect
Gemini's own retry_delay on a 429 rather than hammering the rate limit again."""
from app.services.ai_service import _retry_wait_seconds, AIService


def test_honours_retry_delay_from_a_429_error():
    err = ValueError(
        "429 You exceeded your current quota... [links { ... } , retry_delay { seconds: 31 } ]"
    )
    assert _retry_wait_seconds(err, attempt=1) == 32  # +1s cushion


def test_falls_back_to_a_fixed_wait_for_rate_limit_without_explicit_delay():
    assert _retry_wait_seconds(ValueError('429 rate limit exceeded'), attempt=1) == 40


def test_short_backoff_for_a_non_rate_limit_error():
    assert _retry_wait_seconds(ValueError('some transient network error'), attempt=1) == 2
    assert _retry_wait_seconds(ValueError('some transient network error'), attempt=2) == 4


def test_retry_wait_is_capped():
    err = ValueError("retry_delay { seconds: 9999 }")
    assert _retry_wait_seconds(err, attempt=1) == 40


def test_call_with_retry_sleeps_between_attempts_and_uses_the_given_wait(monkeypatch):
    slept = []
    monkeypatch.setattr('app.services.ai_service.time.sleep', lambda s: slept.append(s))

    service = AIService.__new__(AIService)  # skip __init__ (no API key needed for this)

    class FlakyThenOK:
        calls = 0

        def generate_content(self, prompt, generation_config=None):
            FlakyThenOK.calls += 1
            if FlakyThenOK.calls < 3:
                raise ValueError("429 quota exceeded, retry_delay { seconds: 5 }")
            class R:
                text = '{"ok": true}'
            return R()

    service.model = FlakyThenOK()
    response = service._call_with_retry('prompt', difficulty='medium', max_attempts=3)

    assert response.text == '{"ok": true}'
    assert slept == [6, 6]  # two waits, one before each retry, honouring retry_delay + cushion
