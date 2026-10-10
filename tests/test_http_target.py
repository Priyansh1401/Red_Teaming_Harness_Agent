import argparse
import json
import logging

import httpx
import pytest

from redteam_harness.cli import HTTP_TARGET_WARNING, require_http_permission
from redteam_harness.graph import build_graph, initial_state
from redteam_harness.target import HttpTarget


def test_http_target_renders_request_and_extracts_dot_path():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"data": {"answers": [{"text": "Target reply"}]}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    target = HttpTarget(
        url="https://target.invalid/respond",
        headers={"X-Test-Token": "test-token"},
        body_template={"message": "Question: {prompt}", "metadata": {"source": "harness"}},
        response_path="data.answers.0.text",
        client=client,
    )

    result = target("Target rules", "Generated test prompt")

    assert result == "Target reply"
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://target.invalid/respond"
    assert requests[0].headers["X-Test-Token"] == "test-token"
    assert json.loads(requests[0].content) == {
        "message": "Question: Generated test prompt",
        "metadata": {"source": "harness"},
    }
    client.close()


def test_http_network_failure_retries_and_is_recorded_without_crashing(tmp_path, caplog):
    calls = 0

    def fail_request(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("test connection failure", request=request)

    client = httpx.Client(transport=httpx.MockTransport(fail_request))
    target = HttpTarget(
        url="https://target.invalid/respond",
        body_template={"message": "{prompt}"},
        response_path="reply",
        retry_attempts=2,
        client=client,
    )

    class Attacker:
        def invoke(self, prompt):
            return json.dumps({"strategy_category": "test", "prompt": "test prompt"})

    class Judge:
        def invoke(self, prompt):
            raise AssertionError("Failed target requests should not be sent to the judge")

    graph = build_graph(
        attacker_model=Attacker(),
        judge_model=Judge(),
        target=target,
        strategy_categories=["test"],
        stop_threshold=0.8,
        debug_output_path=tmp_path / "http-debug.jsonl",
    )
    with caplog.at_level(logging.WARNING, logger="redteam_harness.graph"):
        result = graph.invoke(initial_state({
            "target_spec": "Test rules",
            "budget_remaining": 1,
            "debug_output_path": str(tmp_path / "http-debug.jsonl"),
        }))

    assert calls == 2
    assert len(result["attempt_history"]) == 1
    attempt = result["attempt_history"][0]
    assert "failed after 2 attempts" in attempt["response"]
    assert attempt["success"] is False
    assert attempt["severity"] == 0.0
    assert "not judged" in attempt["justification"]
    assert "confidence is low" not in caplog.text
    client.close()


def test_http_mode_prints_warning_and_requires_permission(capsys):
    parser = argparse.ArgumentParser()

    with pytest.raises(SystemExit) as exit_info:
        require_http_permission("http", False, parser)
    assert exit_info.value.code == 2
    assert HTTP_TARGET_WARNING in capsys.readouterr().out

    require_http_permission("http", True, parser)
    assert HTTP_TARGET_WARNING in capsys.readouterr().out
