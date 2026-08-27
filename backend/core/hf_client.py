"""
Async Hugging Face Spaces inference client.
Forwards image bytes to the HF Space REST API and returns structured JSON results.
No TensorFlow or model weights are loaded here — all heavy inference is on Tier 3.
"""
import httpx
from config import get_settings

# Generous timeout: ZeroGPU can queue before allocating
_TIMEOUT = httpx.Timeout(timeout=150.0, connect=10.0)


def _auth_headers() -> dict:
    token = get_settings().hf_api_token
    return {"Authorization": f"Bearer {token}"} if token else {}


async def call_hf_predict(
    image_bytes: bytes,
    filename: str,
    model_choice: str = "mobilenet",
    explain: bool = True,
) -> dict:
    """
    POST image bytes to HF Space /hub_api/predict.
    """
    url = f"{get_settings().hf_space_url}/hub_api/predict"

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            url,
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={
                "model_choice": model_choice,
                "explain": "true" if explain else "false",
                "generate_report": "false",
            },
            headers=_auth_headers(),
        )
        response.raise_for_status()
        return response.json()


async def call_hf_compare(
    image_bytes: bytes,
    filename: str,
    explain: bool = True,
) -> dict:
    """
    POST image bytes to HF Space /hub_api/compare (4-model ensemble).
    """
    url = f"{get_settings().hf_space_url}/hub_api/compare"

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            url,
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={
                "explain": "true" if explain else "false",
                "generate_report": "false",
            },
            headers=_auth_headers(),
        )
        response.raise_for_status()
        return response.json()


async def ping_hf_space() -> dict:
    """
    GET /hub_api/health from the HF Space to verify it is alive.
    """
    url = f"{get_settings().hf_space_url}/hub_api/health"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(url, headers=_auth_headers())
            response.raise_for_status()
            return {"reachable": True, "detail": response.json()}
    except Exception as exc:
        return {"reachable": False, "detail": str(exc)}

