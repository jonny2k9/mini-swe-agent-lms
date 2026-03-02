from unittest.mock import MagicMock, patch

from minisweagent.models.lmstudio_model import LMStudioModel, LMStudioModelConfig


class TestLMStudioModelConfig:
    def test_defaults(self):
        assert LMStudioModelConfig(model_name="my-model").lmstudio_endpoint == "http://localhost:1234"
        assert LMStudioModelConfig(model_name="my-model").api_key == "lm-studio"
        assert LMStudioModelConfig(model_name="my-model").cost_tracking == "ignore_errors"

    def test_model_name_prefix_added(self):
        assert LMStudioModel(model_name="qwen2.5-7b").config.model_name == "openai/qwen2.5-7b"

    def test_model_name_prefix_not_doubled(self):
        assert LMStudioModel(model_name="openai/qwen2.5-7b").config.model_name == "openai/qwen2.5-7b"

    def test_api_base_injected(self):
        assert LMStudioModel(model_name="my-model").config.model_kwargs["api_base"] == "http://localhost:1234/v1"

    def test_api_key_injected(self):
        assert LMStudioModel(model_name="my-model").config.model_kwargs["api_key"] == "lm-studio"

    def test_custom_endpoint(self):
        assert (
            LMStudioModel(model_name="m", lmstudio_endpoint="http://192.168.1.5:5678").config.model_kwargs["api_base"]
            == "http://192.168.1.5:5678/v1"
        )


class TestLMStudioModelListModels:
    def test_list_models_parses_response(self):
        fake = MagicMock()
        fake.json.return_value = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        with patch("minisweagent.models.lmstudio_model.requests.get", return_value=fake):
            assert LMStudioModel.list_models() == ["model-a", "model-b"]

    def test_list_models_empty(self):
        fake = MagicMock()
        fake.json.return_value = {"data": []}
        with patch("minisweagent.models.lmstudio_model.requests.get", return_value=fake):
            assert LMStudioModel.list_models() == []

    def test_list_models_custom_endpoint(self):
        fake = MagicMock()
        fake.json.return_value = {"data": [{"id": "x"}]}
        with patch("minisweagent.models.lmstudio_model.requests.get", return_value=fake) as mock_get:
            LMStudioModel.list_models("http://custom:9999")
        mock_get.assert_called_once_with("http://custom:9999/v1/models", timeout=5)


class TestLMStudioModelQuery:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_passes_api_base(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo hi"}'
        tool_call.id = "c1"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [tool_call]
        mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
        mock_response.model_dump.return_value = {}
        mock_completion.return_value = mock_response
        mock_cost.return_value = 0.0

        LMStudioModel(model_name="my-model").query([{"role": "user", "content": "hi"}])
        assert mock_completion.call_args.kwargs["api_base"] == "http://localhost:1234/v1"

    def test_registered_in_model_class_mapping(self):
        from minisweagent.models import _MODEL_CLASS_MAPPING

        assert "lmstudio" in _MODEL_CLASS_MAPPING
        assert "lmstudio_model.LMStudioModel" in _MODEL_CLASS_MAPPING["lmstudio"]

