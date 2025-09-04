import os, time, uuid, requests

def get_chat_provider():
    prov = (os.getenv("PROVIDER") or "openai").lower()
    return GigaChat() if prov == "gigachat" else OpenAIChat()

class OpenAIChat:
    def __init__(self):
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key  = os.getenv("OPENAI_API_KEY","").strip()
        self.model= os.getenv("OPENAI_MODEL","gpt-4o-mini")
        if not self.key: raise RuntimeError("OPENAI_API_KEY is missing")
    def chat(self, messages, temperature=0.0, max_tokens=400):
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.key}","Content-Type":"application/json"}
        payload = {"model":self.model, "messages":messages, "temperature":temperature, "max_tokens":max_tokens}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
        return r.json()["choices"][0]["message"]["content"].strip()

AUTH_URL = os.getenv("GIGACHAT_AUTH_URL","https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
API_BASE = os.getenv("GIGACHAT_API_URL","https://gigachat.devices.sberbank.ru/api/v1")
SCOPE    = os.getenv("GIGACHAT_SCOPE","GIGACHAT_API_PERS")
AUTH_B64 = os.getenv("GIGACHAT_AUTH")
VERIFY   = (os.getenv("GIGACHAT_VERIFY","true").lower() not in {"0","false","no"})
class _Tok: token=None; exp=0.0
def _get_token():
    if time.time() < _Tok.exp and _Tok.token: return _Tok.token
    if not AUTH_B64: raise RuntimeError("GIGACHAT_AUTH (base64(client:secret)) is missing")
    headers = {"Authorization": f"Basic {AUTH_B64.strip()}","Content-Type":"application/x-www-form-urlencoded","RqUID":str(uuid.uuid4())}
    data = {"scope": SCOPE}
    r = requests.post(AUTH_URL, headers=headers, data=data, timeout=40, verify=VERIFY)
    if r.status_code != 200: raise RuntimeError(f"GigaChat OAuth {r.status_code}: {r.text}")
    js = r.json(); tok = js.get("access_token") or js.get("accessToken")
    if not tok: raise RuntimeError(f"GigaChat OAuth token missing: {js}")
    _Tok.token, _Tok.exp = tok, time.time() + 25*60
    return tok

class GigaChat:
    def __init__(self): self.model = os.getenv("GIGACHAT_MODEL","GigaChat")
    def chat(self, messages, temperature=0.0, max_tokens=400):
        token = _get_token()
        url = f"{API_BASE.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {token}","Content-Type":"application/json; charset=utf-8"}
        payload = {"model": self.model, "messages": messages, "temperature":temperature, "max_tokens":max_tokens}
        r = requests.post(url, headers=headers, json=payload, timeout=60, verify=VERIFY)
        if r.status_code != 200: raise RuntimeError(f"GigaChat error {r.status_code}: {r.text}")
        return r.json().get("choices",[{}])[0].get("message",{}).get("content","").strip()
