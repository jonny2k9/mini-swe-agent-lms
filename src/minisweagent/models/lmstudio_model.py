import os
from typing import Literal

import requests

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig


def _lmstudio_base_url() -> str:
    endpoint = os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234")
    return endpoint.rstrip("/") + "/v1"


class LMStudioModelConfig(LitellmModelConfig):
    lmstudio_endpoint: str = os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234")
    """LM Studio server endpoint (without /v1)."""
    api_key: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    """API key (LM Studio accepts any non-empty string)."""
    cost_tracking: Literal["default", "ignore_errors"] = "ignore_errors"
    """Local models have no cost; always ignore cost tracking errors."""


class LMStudioModel(LitellmModel):
    """Model backend for LM Studio's local OpenAI-compatible API."""

    def __init__(self, **kwargs):
        model_name = kwargs.get("model_name", os.getenv("LMSTUDIO_MODEL", ""))
        if model_name and not model_name.startswith("openai/"):
            kwargs["model_name"] = f"openai/{model_name}"
        super().__init__(config_class=LMStudioModelConfig, **kwargs)
        self.config.model_kwargs.setdefault("api_base", self.config.lmstudio_endpoint.rstrip("/") + "/v1")
        self.config.model_kwargs.setdefault("api_key", self.config.api_key)

    @staticmethod
    def list_models(endpoint: str | None = None) -> list[str]:
        """Return model IDs available in LM Studio."""
        base = (endpoint or os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234")).rstrip("/")
        data = requests.get(f"{base}/v1/models", timeout=5).json()
        return [m["id"] for m in data.get("data", [])]
