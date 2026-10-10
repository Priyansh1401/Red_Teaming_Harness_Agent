import redteam_harness.cli as cli


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_auto_selects_ollama_without_openai_key(monkeypatch, capsys):
    monkeypatch.setattr(cli, "ChatOllama", FakeModel)
    monkeypatch.setattr(cli, "ChatOpenAI", FakeModel)
    config = {"model": "qwen2.5"}

    attacker, judge = cli.create_chat_models(config, environ={})

    assert attacker.kwargs == {"model": "qwen2.5", "temperature": 0.7, "format": "json"}
    assert judge.kwargs == {"model": "qwen2.5", "temperature": 0, "format": "json"}
    assert capsys.readouterr().out == (
        "No OPENAI_API_KEY found — running on local Ollama (qwen2.5). "
        "This will be slower than OpenAI. Set OPENAI_API_KEY or provider: openai "
        "in config.yaml to use OpenAI instead.\n"
    )


def test_auto_selects_openai_when_key_is_set(monkeypatch, capsys):
    monkeypatch.setattr(cli, "ChatOllama", FakeModel)
    monkeypatch.setattr(cli, "ChatOpenAI", FakeModel)

    attacker, judge = cli.create_chat_models({}, environ={"OPENAI_API_KEY": "test-key"})

    expected_kwargs = {"response_format": {"type": "json_object"}}
    assert attacker.kwargs == {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "model_kwargs": expected_kwargs,
    }
    assert judge.kwargs == {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "model_kwargs": expected_kwargs,
    }
    assert capsys.readouterr().out == ""


def test_explicit_provider_wins_over_environment(monkeypatch):
    monkeypatch.setattr(cli, "ChatOllama", FakeModel)
    monkeypatch.setattr(cli, "ChatOpenAI", FakeModel)

    attacker, judge = cli.create_chat_models(
        {"provider": "ollama", "model": "custom-local-model"},
        environ={"OPENAI_API_KEY": "test-key"},
    )

    assert attacker.kwargs["model"] == "custom-local-model"
    assert judge.kwargs["model"] == "custom-local-model"


def test_explicit_openai_without_key_has_actionable_error():
    try:
        cli.select_providers({"provider": "openai"}, environ={})
    except RuntimeError as error:
        assert "Set OPENAI_API_KEY" in str(error)
        assert "provider" in str(error)
    else:
        raise AssertionError("Expected a missing OpenAI API key error")
