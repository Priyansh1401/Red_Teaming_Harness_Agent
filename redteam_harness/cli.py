import argparse
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from .config import ALLOWED_PROVIDERS, load_config
from .graph import build_graph, initial_state
from .target import HttpTarget, MockTarget, OllamaTarget


OLLAMA_DEFAULT_MODEL = "llama3.2"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
HTTP_TARGET_WARNING = (
    "WARNING: You are about to send automated adversarial prompts to an external endpoint. "
    "Only do this against systems you own or have explicit written permission to test."
)


def require_http_permission(target_mode: str, permission_granted: bool, parser: argparse.ArgumentParser) -> None:
    if target_mode != "http":
        return
    print(f"\n{HTTP_TARGET_WARNING}\n")
    if not permission_granted:
        parser.error("HTTP target mode requires --i-have-permission to continue.")


def select_providers(config: dict, environ: dict[str, str] | None = None) -> tuple[str, str]:
    environment = os.environ if environ is None else environ
    configured_provider = config.get("provider")
    if configured_provider is None:
        provider = "openai" if environment.get("OPENAI_API_KEY") else "ollama"
        if provider == "ollama":
            ollama_model = config.get("attacker", {}).get(
                "model",
                config.get("model", OLLAMA_DEFAULT_MODEL),
            )
            print(
                f"No OPENAI_API_KEY found — running on local Ollama ({ollama_model}). "
                "This will be slower than OpenAI. Set OPENAI_API_KEY or provider: openai "
                "in config.yaml to use OpenAI instead."
            )
    else:
        provider = configured_provider
    if provider not in ALLOWED_PROVIDERS:
        raise ValueError("provider must be either 'openai' or 'ollama'")

    providers = []
    for role in ("attacker", "judge"):
        role_provider = config.get(role, {}).get("provider", provider)
        if role_provider not in ALLOWED_PROVIDERS:
            raise ValueError(f"{role}.provider must be either 'openai' or 'ollama'")
        if role_provider == "openai" and not environment.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OpenAI provider selected but OPENAI_API_KEY is missing. "
                "Set OPENAI_API_KEY or switch config.yaml's provider to 'ollama'."
            )
        providers.append(role_provider)
    return providers[0], providers[1]


def create_chat_models(config: dict, environ: dict[str, str] | None = None):
    attacker_provider, judge_provider = select_providers(config, environ)
    environment = os.environ if environ is None else environ

    def model_name(role: str, provider: str) -> str:
        role_config = config.get(role, {})
        if "model" in role_config:
            return role_config["model"]
        if "model" in config:
            return config["model"]
        if provider == "ollama":
            return OLLAMA_DEFAULT_MODEL
        return environment.get("REDTEAM_MODEL", OPENAI_DEFAULT_MODEL)

    def create(provider: str, role: str, temperature: float):
        name = model_name(role, provider)
        if provider == "ollama":
            return ChatOllama(model=name, temperature=temperature, format="json")
        return ChatOpenAI(
            model=name,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    return create(attacker_provider, "attacker", 0.7), create(judge_provider, "judge", 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an authorized chatbot safety evaluation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--i-have-permission", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    # SAFETY: HTTP mode sends adversarial prompts to an external endpoint. Only test systems
    # you own or have explicit written permission to assess; require deliberate confirmation.
    require_http_permission(config["target_mode"], args.i_have_permission, parser)
    attacker_model, judge_model = create_chat_models(config)
    if config["target_mode"] == "mock":
        target = MockTarget()
    elif config["target_mode"] == "ollama":
        target = OllamaTarget(model_name=config["target_model"])
    else:
        target = HttpTarget(
            url=config["target_url"],
            headers=config["target_headers"],
            body_template=config["request_body_template"],
            response_path=config["response_text_path"],
            timeout=float(config["request_timeout"]),
            retry_attempts=int(config["retry_attempts"]),
        )
    graph = build_graph(
        attacker_model=attacker_model,
        judge_model=judge_model,
        target=target,
        strategy_categories=list(config["strategy_categories"]),
        stop_threshold=float(config["stop_threshold"]),
        debug_output_path=config["debug_output_path"],
    )
    result = graph.invoke(initial_state(config))
    Path(config["report_output_path"]).write_text(result["final_report"], encoding="utf-8")
    print(result["final_report"])


if __name__ == "__main__":
    main()
