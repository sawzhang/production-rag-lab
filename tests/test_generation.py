import json

import pytest

from rag_lab.generation import generate, resolve_key, validate_generation


def test_key_resolution_explicit_and_never_implicit(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        resolve_key()
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"deepseek": {"type": "api_key", "key": "fake-test-key"}}))
    assert resolve_key(path) == "fake-test-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-test-key")
    assert resolve_key(path) == "env-test-key"


def test_quote_schema_fact_and_budget():
    bundle = {"fact": None, "evidence": [{"id": "a", "text": "支付成功不等于履约完成"}]}
    good = {
        "status": "answered",
        "fact": None,
        "citations": [{"id": "a", "quote": "支付成功不等于履约完成"}],
    }
    assert validate_generation(good, bundle) == good
    for bad in (
        dict(good, fact={"amount_cents": 1}),
        dict(good, citations=[{"id": "other", "quote": "支付成功"}]),
        dict(good, citations=[{"id": "a", "quote": "支付成功等于履约完成"}]),
        dict(good, citations=[]),
        dict(good, status="conflict"),
        dict(good, status="insufficient"),
        dict(good, citations=good["citations"] * 5),
        dict(good, extra="invented"),
    ):
        with pytest.raises(ValueError):
            validate_generation(bad, bundle)
    with pytest.raises(ValueError):
        validate_generation(
            {"status": "answered", "fact": {"amount_cents": 1.0}, "citations": []},
            {"fact": {"amount_cents": 1}, "evidence": []},
        )


def test_api_key_not_in_prompt_and_error_redacted(monkeypatch):
    import urllib.request

    def fail(request, timeout):
        payload = json.loads(request.data)
        assert "fake-secret-token" not in str(payload)
        assert request.full_url == "https://api.deepseek.com/chat/completions"
        raise urllib.error.HTTPError(request.full_url, 401, "fake-secret-token", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="DeepSeek HTTP 401") as error:
        generate({"fact": None, "evidence": []}, "q", "fake-secret-token")
    assert "fake-secret-token" not in str(error.value)


def test_api_response_json_and_truncation(monkeypatch):
    import io
    import urllib.request

    raw = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps({"status": "insufficient", "fact": None, "citations": []})
                },
            }
        ],
        "model": "deepseek-flash",
        "usage": {"total_tokens": 10},
    }
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(raw).encode())
    )
    draft, metadata = generate({"fact": None, "evidence": []}, "q", "fake")
    assert draft["status"] == "insufficient" and metadata["usage"]["total_tokens"] == 10
    raw["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="incomplete"):
        generate({"fact": None, "evidence": []}, "q", "fake")
