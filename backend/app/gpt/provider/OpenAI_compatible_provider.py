from typing import Union
from urllib.parse import urlparse

from app.utils.logger import get_logger
from app.utils.openai_client import build_openai_client

logging= get_logger(__name__)


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize root-only provider URLs to the OpenAI-compatible /v1 endpoint."""
    raw = (base_url or "").strip()
    if not raw:
        return raw

    parsed = urlparse(raw)
    normalized = raw.rstrip("/")
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return f"{normalized}/v1"
    return normalized


def is_chat_completion_response(response) -> bool:
    choices = getattr(response, "choices", None)
    return isinstance(choices, (list, tuple)) and len(choices) > 0


def require_chat_completion(response):
    if not is_chat_completion_response(response):
        raise RuntimeError(
            "模型接口返回了非 ChatCompletion 响应，请检查 API 地址是否指向 OpenAI 兼容的 /v1 接口"
        )
    return response


class OpenAICompatibleProvider:
    def __init__(self, api_key: str, base_url: str, model: Union[str, None]=None):
        # build_openai_client：注入全局代理 + 校验 api_key 非空
        self.client = build_openai_client(api_key, base_url, key_label="模型供应商的 API Key")
        self.model = model

    @property
    def get_client(self):
        return self.client

    @staticmethod
    def test_connection(api_key: str, base_url: str, model: str) -> bool:
        """发一条最小化 chat completion 验证 key / base_url / model 三方都通。

        为什么不用 client.models.list()：
          - 部分代理 / 自建供应商不实现 /v1/models（如某些 OpenAI 兼容网关）
          - 部分供应商 key 在没有 inference 权限时 /v1/models 仍返回 200
        最终用户跑的就是 chat.completions.create，所以直接测它最忠实。
        max_tokens=1 + temperature=0 让请求开销 < 0.0001 美元、延迟 < 2s。
        """
        normalized = normalize_openai_base_url(base_url)
        candidates = [normalized]
        if normalized != (base_url or "").strip().rstrip("/"):
            candidates.append((base_url or "").strip().rstrip("/"))

        last_error = None
        for candidate in candidates:
            try:
                client = build_openai_client(
                    api_key, candidate, key_label="模型供应商的 API Key", timeout=15.0,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0,
                )
                require_chat_completion(response)
                logging.info(f"连通性测试成功（model={model}, base_url={candidate}）")
                return True
            except Exception as e:
                last_error = e
                logging.warning(
                    f"连通性测试失败（model={model}, base_url={candidate}）：{e}"
                )
        logging.warning(f"连通性测试失败（model={model}）：{last_error}")
        return False
