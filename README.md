# Mini‑RAG (FAQ + цитаты)

Демонстрационный проект RAG для FAQ: ingest → index → retrieve → prompt → ответ **с цитатами**.

- **Ингест**: `.md`/`.txt` документы (образцы в `samples/faq/`), нарезка на чанки.
- **Индекс**: `TF-IDF` (для демо). В проде — `FAISS/Chroma`.
- **Ретривер**: косинусная близость, top‑k.
- **Ответ**: LLM‑чат через **OpenAI** или **GigaChat** (по ENV). Есть режим **без LLM** — ответ на основе лучших пассажей.
- **Цитаты**: для каждого пассажa — источник (название и псевдо‑URL).

## Быстрый старт
```bash
pip install -r requirements.txt
streamlit run demo_streamlit.py
```

## Провайдеры (по желанию)

**OpenAI:**
```
PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

**GigaChat:**
```
PROVIDER=gigachat
GIGACHAT_AUTH=<Authorization Key: base64(client_id:client_secret)>
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_AUTH_URL=https://ngw.devices.sberbank.ru:9443/api/v2/oauth
GIGACHAT_API_URL=https://gigachat.devices.sberbank.ru/api/v1
GIGACHAT_VERIFY=false
GIGACHAT_MODEL=GigaChat
```

Если переменные не заданы — выключите «Использовать LLM» в демо: ответ соберётся из пассажей.
