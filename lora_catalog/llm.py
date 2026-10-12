"""
One way to talk to the LLM: any OpenAI-compatible chat server (LM Studio, Ollama, llama.cpp server, vLLM...).

    chat(settings, model, messages, ...)            -> the answer text (reasoning blocks removed)
    chat_json(settings, model, messages, schema)    -> the answer as a dict

Structured output: the JSON schema is sent as response_format first. A server that refuses it (HTTP 400/422 naming
response_format, or one that answers outside the schema) is remembered for the rest of the run, and from then on the
schema goes into the prompt instead ("Answer with JSON only, matching this schema") and the answer is repaired:
reasoning blocks and code fences removed, the outer {...} taken, trailing commas dropped. One retry on a bad answer.

unload(settings, model) frees the GPU between the card model and the text model: LM Studio through its `lms` command,
Ollama through keep_alive 0; other servers are left alone.

test(settings) is the setup page's Test button: one small text call and one small image call (and, on Ollama, where
the card model sits: GPU or partly CPU, and its context length).

A read timeout raises LLMTimeout; timeout_hint() says what usually causes one. ollama_check() loads the card model
and reads /api/ps before the card pass, to warn when it spills onto the CPU or has too small a context (2026-10-12,
after a tester's Ollama refresh timed out on every card while the setup test passed).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
OPEN_THINK = re.compile(r"^.*?</think>", re.S | re.I)          # a reply that starts inside a reasoning block
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
TRAILING_COMMA = re.compile(r",\s*([}\]])")

_schema_refused: Dict[str, bool] = {}       # chat url -> the server does not take response_format json_schema


class LLMError(RuntimeError):
    pass


class LLMTimeout(LLMError):
    """The server took the request but did not answer within the timeout."""


# The card pass needs about this much context: MAX_EVIDENCE_CHARS of text with 2 images, the schema and the answer
# (cards.py). Ollama's default context is smaller on many setups.
CARD_CONTEXT_MIN = 8192
CARD_CONTEXT_ADVISED = 16384


def _headers(settings) -> Dict[str, str]:
    key = str(getattr(settings, "llm_api_key", "") or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def strip_reasoning(text: str) -> str:
    text = THINK.sub("", text or "")
    if "</think>" in text.lower():
        text = OPEN_THINK.sub("", text)
    return text.strip()


def parse_json(text: str) -> Dict[str, Any]:
    """The answer as a dict, repaired where that is safe. Raises ValueError when there is no JSON object in it."""
    text = FENCE.sub("", strip_reasoning(text)).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the answer (the model may have used up its tokens thinking)")
    body = text[start:end + 1]
    try:
        return json.loads(body)
    except ValueError:
        return json.loads(TRAILING_COMMA.sub(r"\1", body))


def _post(settings, body: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    import requests
    url = settings.chat_url()
    try:
        response = requests.post(url, json=body, headers=_headers(settings), timeout=timeout)
    except requests.ConnectionError as error:
        raise LLMError(f"no LLM server at {url} - start LM Studio / Ollama, or set llm_base_url") from error
    except requests.Timeout as error:             # read timeout: the server has the request but is too slow
        raise LLMTimeout(f"no answer from {url} within {timeout} s (read timeout)") from error
    if response.status_code in (400, 422) and "response_format" in body:
        message = response.text[:400].lower()
        if any(word in message for word in ("response_format", "json_schema", "schema", "grammar")):
            _schema_refused[url] = True
            raise _SchemaRefused(message)
    if response.status_code == 404 and "model" in response.text.lower():
        raise LLMError(f"the server does not have model '{body.get('model')}' ({response.text[:200]})")
    response.raise_for_status()
    return response.json()


class _SchemaRefused(Exception):
    pass


def _message_text(data: Dict[str, Any]) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMError(f"unexpected answer from the server: {str(data)[:200]}") from error
    return strip_reasoning(message.get("content") or "")


def chat(settings, model: str, messages: List[Dict[str, Any]], temperature: float = 0.1, max_tokens: int = 1500,
         timeout: int = 300, response_format: Optional[Dict[str, Any]] = None) -> str:
    """The answer text. response_format is passed through as given (chat_json handles schema refusal)."""
    body: Dict[str, Any] = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    if response_format:
        body["response_format"] = response_format
    return _message_text(_post(settings, body, timeout))


def _schema_of(response_format: Dict[str, Any]) -> Dict[str, Any]:
    return ((response_format or {}).get("json_schema") or {}).get("schema") or response_format or {}


def _with_schema_prompt(messages: List[Dict[str, Any]], response_format: Dict[str, Any]) -> List[Dict[str, Any]]:
    note = ("\n\nAnswer with one JSON object only - no prose, no code fence - matching this JSON schema:\n"
            + json.dumps(_schema_of(response_format), ensure_ascii=False))
    out = [dict(m) for m in messages]
    for message in out:
        if message.get("role") == "system":
            message["content"] = str(message.get("content") or "") + note
            return out
    return [{"role": "system", "content": note.strip()}] + out


def chat_json(settings, model: str, messages: List[Dict[str, Any]], response_format: Dict[str, Any],
              temperature: float = 0.1, max_tokens: int = 1500, timeout: int = 300, retries: int = 1,
              with_text: bool = False) -> Any:
    """The answer as a dict (or (dict, answer text) with with_text, for callers that continue the conversation).
    response_format is the OpenAI-style {"type": "json_schema", "json_schema": {...}}."""
    url = settings.chat_url()
    last: Exception = LLMError("no attempt")
    attempts = 0
    while attempts <= retries:
        refused = _schema_refused.get(url, False)
        text = ""
        try:
            if refused:
                text = chat(settings, model, _with_schema_prompt(messages, response_format), temperature, max_tokens,
                            timeout)
            else:
                text = chat(settings, model, messages, temperature, max_tokens, timeout, response_format)
            answer = parse_json(text)
            return (answer, text) if with_text else answer
        except _SchemaRefused:
            print(f"[llm] {url} does not take a JSON schema; asking for JSON in the prompt instead", flush=True)
            continue                                    # again at once in prompt mode; not counted as a retry
        except ValueError as error:
            last = error
            attempts += 1
            # Prose instead of JSON while a schema was sent: this server ignores response_format, so put the schema in
            # the prompt for the rest of the run. An empty answer (reasoning used up max_tokens) or a cut-off one
            # (starts with '{') says nothing about the server, so the schema mode stays.
            stripped = FENCE.sub("", text).strip()
            if not refused and stripped and not stripped.startswith("{"):
                _schema_refused[url] = True
    raise last


# -- unloading --------------------------------------------------------------------------------------------------------
def backend(settings) -> str:
    chosen = str(getattr(settings, "llm_backend", "auto") or "auto").lower()
    if chosen != "auto":
        return chosen
    base = str(getattr(settings, "llm_base_url", "") or settings.chat_url())
    if ":11434" in base:
        return "ollama"
    if ":1234" in base:
        return "lmstudio"
    return "other"


def _lms() -> str:
    found = shutil.which("lms")
    if found:
        return found
    for name in ("lms.exe", "lms"):
        path = os.path.join(os.path.expanduser("~"), ".lmstudio", "bin", name)
        if os.path.isfile(path):
            return path
    return ""


def _ollama_root(settings) -> str:
    root = str(getattr(settings, "llm_base_url", "") or settings.chat_url()).rstrip("/")
    for tail in ("/chat/completions", "/v1"):
        if root.endswith(tail):
            root = root[:-len(tail)]
    return root


def ollama_check(settings, model: str) -> List[str]:
    """Load `model` on Ollama and read /api/ps. Returns warnings in plain words (empty = all well); never raises.
    Warns when part of the model runs on the CPU (very slow with images) and when the context is too small for a
    card."""
    import requests
    root = _ollama_root(settings)
    try:
        requests.post(root + "/api/generate", json={"model": model, "keep_alive": "10m", "stream": False},
                      timeout=300).raise_for_status()          # no prompt: Ollama only loads the model
        loaded = requests.get(root + "/api/ps", timeout=30).json().get("models") or []
    except Exception as error:
        return [f"could not ask Ollama about {model}: {type(error).__name__}: {error}"]
    names = {model, model + ":latest"}
    entry = next((m for m in loaded if m.get("name") in names or m.get("model") in names), None)
    if entry is None:
        return [f"Ollama does not list {model} as loaded after loading it - check the model name"]
    notes: List[str] = []
    size, vram = int(entry.get("size") or 0), int(entry.get("size_vram") or 0)
    if size and vram < size:
        cpu = round(100 * (size - vram) / size)
        notes.append(f"{model} runs {cpu}% on the CPU ({vram / 2**30:.1f} of {size / 2**30:.1f} GB in VRAM). Image "
                     f"requests get very slow and time out. Free VRAM first: unload ComfyUI's models, close other "
                     f"GPU apps, or run refresh.bat with ComfyUI closed.")
    context = int(entry.get("context_length") or 0)            # reported by newer Ollama versions
    if context and context < CARD_CONTEXT_MIN:
        notes.append(f"{model} is loaded with a {context}-token context; a card needs about {CARD_CONTEXT_MIN}. "
                     f"Set the environment variable OLLAMA_CONTEXT_LENGTH={CARD_CONTEXT_ADVISED} and restart Ollama.")
    return notes


def timeout_hint(settings, workers: int = 1) -> str:
    """What usually causes a read timeout on this server, for the log (printed once per run)."""
    kind = backend(settings)
    lines = ["[llm] read timeouts mean the server is too slow with these requests. Usual causes:"]
    if kind == "ollama":
        lines += ["  - the model does not fit in VRAM and runs partly on the CPU: `ollama ps` should show 100% GPU. "
                  "ComfyUI holding VRAM is the common reason - unload its models, or run refresh.bat with ComfyUI "
                  "closed.",
                  f"  - the context is too small for the images and text: set OLLAMA_CONTEXT_LENGTH="
                  f"{CARD_CONTEXT_ADVISED} and restart Ollama."]
    elif kind == "lmstudio":
        lines += ["  - the model does not fit in VRAM (partial GPU offload): free VRAM or pick a smaller quant.",
                  f"  - the context is too small: load the model with --context-length {CARD_CONTEXT_ADVISED}."]
    else:
        lines += ["  - the model does not fit in VRAM, or its context is too small for images plus text "
                  f"(about {CARD_CONTEXT_MIN} tokens)."]
    if workers > 1:
        lines.append(f"  - {workers} requests at once: a server that answers one at a time queues the rest, and the "
                     f"wait counts against the timeout. Try --workers 1.")
    lines.append("  Try one card to time it: python run.py cards --limit 1")
    return "\n".join(lines)


def unload(settings, model: str) -> str:
    """Free the model's memory where the server allows it. Returns what was done (for the log); never raises."""
    if not model:
        return "nothing to unload"
    kind = backend(settings)
    try:
        if kind == "lmstudio":
            lms = _lms()
            if not lms:
                return "LM Studio's lms command not found; model left loaded"
            done = subprocess.run([lms, "unload", model], capture_output=True, text=True, timeout=60,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return f"lms unload {model}: " + (done.stdout or done.stderr or "").strip()[:200]
        if kind == "ollama":
            import requests
            root = _ollama_root(settings)
            requests.post(root + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
            return f"ollama: {model} unloaded"
    except Exception as error:                         # unloading is a nicety; a failure must not stop a run
        return f"unload {model} failed: {type(error).__name__}: {error}"
    return f"{kind} server: models are not unloaded by this tool"


# -- setup test -------------------------------------------------------------------------------------------------------
def _tiny_png() -> str:
    """A 32x32 red square with a white bar, as a data URI (for the vision test)."""
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (32, 32), (200, 30, 30))
    ImageDraw.Draw(image).rectangle([4, 13, 27, 18], fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def test(settings) -> Dict[str, Any]:
    """{'ok': bool, 'text': str, 'vision': str, 'json': str} in plain words for the setup page."""
    out: Dict[str, Any] = {"server": settings.chat_url(), "ok": False}
    try:
        answer = chat(settings, settings.genre_model, [{"role": "user", "content": "Reply with the single word: ready"}],
                      temperature=0.0, max_tokens=400, timeout=120)
        out["text"] = f"{settings.genre_model}: answered '{answer[:40]}'"
        out["ok"] = True
    except Exception as error:
        out["text"] = f"{settings.genre_model}: {error}"
    try:
        schema = {"type": "json_schema", "json_schema": {"name": "t", "strict": True, "schema": {
            "type": "object", "properties": {"color": {"type": "string"}}, "required": ["color"],
            "additionalProperties": False}}}
        answer = chat_json(settings, settings.card_model, [{"role": "user", "content": [
            {"type": "text", "text": "What is the main colour of this picture? JSON: {\"color\": \"...\"}"},
            {"type": "image_url", "image_url": {"url": _tiny_png()}}]}], schema, temperature=0.0, max_tokens=600,
            timeout=180)
        out["vision"] = f"{settings.card_model}: sees '{answer.get('color', '?')}'"
        mode = "prompt (server ignores JSON schemas)" if _schema_refused.get(settings.chat_url()) else "schema"
        out["json"] = f"structured answers: {mode}"
        if backend(settings) == "ollama":          # the tiny test passes even when the real cards will time out
            notes = ollama_check(settings, settings.card_model)
            out["memory"] = "\n".join("warning: " + n for n in notes) if notes else f"{settings.card_model}: 100% GPU"
    except Exception as error:
        out["ok"] = False
        out["vision"] = (f"{settings.card_model}: {error} - the card pass needs a vision model "
                         f"(e.g. Qwen3-VL 8B or Qwen2.5-VL 7B)")
    return out
