# demo_streamlit.py — robust query handling via session_state; shows used query
import os, sys, streamlit as st

# Make ./src importable on Streamlit Cloud
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ragmini import read_docs, build_corpus, build_tfidf_index, search, get_chat_provider

st.set_page_config(page_title='Mini‑RAG (FAQ + цитаты)', page_icon='📚', layout='wide')
st.title('Mini‑RAG (FAQ + цитаты)')

# Build index (small corpus — быстро). Use absolute paths.
DOCS_DIR = os.path.join(BASE_DIR, 'samples', 'faq')
docs = read_docs(DOCS_DIR)
corpus, meta = build_corpus(docs)
vect, mat = build_tfidf_index(corpus)

DEFAULT_Q = 'Как перевыпустить карту и сколько это стоит?'

# Ensure defaults in state
st.session_state.setdefault('query', DEFAULT_Q)
st.session_state.setdefault('topk', 5)
st.session_state.setdefault('use_llm', False)
st.session_state.setdefault('hits', None)
st.session_state.setdefault('last_query', None)

with st.form('search_form', clear_on_submit=False):
    query = st.text_input('Вопрос', value=st.session_state['query'])
    topk = st.slider('Top‑k пассажей', 3, 10, st.session_state['topk'])
    use_llm = st.toggle('Использовать LLM для финального ответа',
                        value=st.session_state['use_llm'],
                        help='Если выключено — ответ строится из лучших пассажей.')
    submitted = st.form_submit_button('Искать')

if submitted:
    query = query.strip()
    st.session_state['query'] = query
    st.session_state['topk'] = int(topk)
    st.session_state['use_llm'] = bool(use_llm)
    st.session_state['hits'] = search(query, vect, mat, corpus, meta, k=st.session_state['topk'])
    st.session_state['last_query'] = query

hits = st.session_state['hits']
if hits is not None:
    st.write(f'Запрос: **{st.session_state.get("last_query") or st.session_state.get("query")}**')
    st.subheader('Найденные пассажи')
    for h in hits:
        st.write(f"**{h['title']}** — {h['url']}  \n_рейтинг: {h['score']:.3f}_")
        st.write(h['passage'])
        st.markdown('---')

    st.subheader('Ответ')
    if not st.session_state['use_llm']:
        answer = hits[0]['passage'] if hits else 'Не найдено.'
        citations = [h['url'] for h in hits[:3]]
        st.write(answer)
        st.caption('Цитаты: ' + ' · '.join(citations))
    else:
        provider = get_chat_provider()
        context = '\n\n'.join([f"[{i+1}] {h['passage']} (src: {h['url']})" for i,h in enumerate(hits[:st.session_state['topk']])])
        messages = [
            {"role":"system","content":"Ты ассистент банка. Отвечай кратко и по делу. Обязательно ссылайся на источники [номер] в конце фразы."},
            {"role":"user","content": f"Вопрос: {st.session_state['last_query']}\n\nКонтекст:\n{context}\n\nОтветь по контексту. Если информации нет — скажи, что не нашёлся ответ."}
        ]
        try:
            ans = provider.chat(messages, temperature=0.0, max_tokens=400)
            st.write(ans)
        except Exception as e:
            st.error(f'LLM error: {e}')
else:
    st.caption('Введите вопрос и нажмите «Искать». Можно менять формулировку — индекс пересчитывать не нужно.')
