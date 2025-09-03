
import os, json

# allow reading from streamlit.secrets if available
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore

def _sec(name: str, default=None):
    v = os.environ.get(name)
    if v: return v
    if st is not None:
        try:
            v = st.secrets.get(name)  # type: ignore[attr-defined]
            if v: return str(v)
        except Exception:
            pass
    return default

# ---------------- OpenAI-compatible provider ----------------
class OpenAIChat:
    def __init__(self, model=None):
        self.key  = _sec("OPENAI_API_KEY")
        self.base = _sec("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = model or _sec("OPENAI_MODEL","gpt-4o-mini")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY is missing")

    def chat(self, messages_or_prompt, temperature: float = 0.0, max_tokens=None, **kw) -> str:
        import requests
        if isinstance(messages_or_prompt, str):
            messages = [{"role":"user","content":messages_or_prompt}]
        else:
            messages = messages_or_prompt
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
        r = requests.post(f"{self.base}/chat/completions", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# ---------------- Sber GigaChat provider ----------------
class GigaChat:
    def __init__(self, model=None):
        import base64
        self.model = model or _sec("GIGACHAT_MODEL","GigaChat-Pro")
        self.scope = _sec("GIGACHAT_SCOPE","GIGACHAT_API_PERS")
        self.auth_key = _sec("GIGACHAT_AUTH_KEY") or _sec("GIGACHAT_AUTH")
        if not self.auth_key:
            cid = _sec("GIGACHAT_CLIENT_ID")
            csec = _sec("GIGACHAT_CLIENT_SECRET")
            if cid and csec:
                self.auth_key = base64.b64encode(f"{cid}:{csec}".encode()).decode()
        vr = (_sec("GIGACHAT_VERIFY","true") or "true").strip().lower()
        self.verify = False if vr in ("0","false","no","off") else True
        if not self.auth_key:
            raise RuntimeError("GIGACHAT_AUTH_KEY (или пара CLIENT_ID/CLIENT_SECRET) не задан")
        self._token = None

    def _get_token(self) -> str:
        import requests, uuid
        if self._token: return self._token
        headers = {
            "Authorization": f"Basic {self.auth_key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
        }
        data = {"scope": self.scope}
        r = requests.post("https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                          headers=headers, data=data, timeout=60, verify=self.verify)
        r.raise_for_status()
        self._token = r.json()["access_token"]
        return self._token

    def chat(self, messages_or_prompt, temperature: float = 0.0, max_tokens=None, **kw) -> str:
        import requests, uuid
        tok = self._get_token()
        if isinstance(messages_or_prompt, str):
            messages = [{"role":"user","content":messages_or_prompt}]
        else:
            messages = messages_or_prompt
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
        }
        r = requests.post("https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                          headers=headers, json=payload, timeout=120, verify=self.verify)
        if r.status_code == 401:
            self._token = None
            return self.chat(messages_or_prompt, temperature=temperature, max_tokens=max_tokens, **kw)
        r.raise_for_status()
        j = r.json()
        try:
            return j["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(j, ensure_ascii=False)

def get_chat_provider():
    """
    Автовыбор провайдера:
      - если заданы GIGACHAT_* или PROVIDER=gigachat → GigaChat
      - иначе OpenAI (потребует OPENAI_API_KEY)
    """
    prov = (os.getenv("PROVIDER","") or "").lower()
    if prov == "gigachat" or _sec("GIGACHAT_AUTH_KEY") or _sec("GIGACHAT_AUTH") or _sec("GIGACHAT_CLIENT_ID"):
        return GigaChat()
    return OpenAIChat()
