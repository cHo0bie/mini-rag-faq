# demo_streamlit.py — query is handled via Streamlit form + session_state
import os, sys, streamlit as st

# ensure ./src is importable on Streamlit Cloud
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ragmini import read_docs, build_corpus, build_tfidf_index, search, get_chat_provider

st.set_page_config(page_title='Mini‑RAG (FAQ + цитаты)', page_icon='📚', layout='wide')
st.title('Mini‑RAG (FAQ + цитаты)')

# Build index once per rerun (small corpus, fast). Paths are absolute.
DOCS_DIR = os.path.join(BASE_DIR, 'samples', 'faq')
docs = read_docs(DOCS_DIR)
corpus, meta = build_corpus(docs)
vect, mat = build_tfidf_index(corpus)

# Defaults & state
DEFAULT_Q = 'Как перевыпустить карту и сколько это стоит?'
if 'q' not in st.session_state: st.session_state.q = DEFAULT_Q
if 'topk' not in st.session_state: st.session_state.topk = 5
if 'use_llm' not in st.session_state: st.session_state.use_llm = False
if 'hits' not in st.session_state: st.session_state.hits = None
if 'last_q' not in st.session_state: st.session_state.last_q = None

# Form to lock values on submit (avoids "reset to default" issues)
with st.form(key='search_form', clear_on_submit=False):
    col1, col2 = st.columns([1,1])
    with col1:
        st.session_state.q = st.text_input('Вопрос', st.session_state.q, key='q_input')
    with col2:
        st.session_state.topk = st.slider('Top‑k пассажей', 3, 10, st.session_state.topk, key='topk_slider')

    st.session_state.use_llm = st.toggle('Использовать LLM для финального ответа',
                                         value=st.session_state.use_llm,
                                         help='Если выключено — ответ строится из лучших пассажей.',
                                         key='use_llm_toggle')

    submitted = st.form_submit_button('Искать')

# Compute on submit
if submitted:
    q = st.session_state.q_input.strip()
    topk = int(st.session_state.topk_slider)
    st.session_state.hits = search(q, vect, mat, corpus, meta, k=topk)
    st.session_state.last_q = q

# Show results if available
hits = st.session_state.hits
if hits is not None:
    st.subheader('Найденные пассажи')
    for h in hits:
        st.write(f"**{h['title']}** — {h['url']}  \n_рейтинг: {h['score']:.3f}_")
        st.write(h['passage'])
        st.markdown('---')

    st.subheader('Ответ')
    if not st.session_state.use_llm:
        answer = hits[0]['passage'] if hits else 'Не найдено.'
        citations = [h['url'] for h in hits[:3]]
        st.write(answer)
        st.caption('Цитаты: ' + ' · '.join(citations))
    else:
        provider = get_chat_provider()
        context = '\n\n'.join([f"[{i+1}] {h['passage']} (src: {h['url']})" for i,h in enumerate(hits[:st.session_state.topk])])
        messages = [
            {"role":"system","content":"Ты ассистент банка. Отвечай кратко и по делу. Обязательно ссылайся на источники [номер] в конце фразы."},
            {"role":"user","content": f"Вопрос: {st.session_state.last_q}\n\nКонтекст:\n{context}\n\nОтветь по контексту. Если информации нет — скажи, что не нашёлся ответ."}
        ]
        try:
            ans = provider.chat(messages, temperature=0.0, max_tokens=400)
            st.write(ans)
        except Exception as e:
            st.error(f'LLM error: {e}')

# Small tip if nothing searched yet
else:
    st.caption('Введите вопрос и нажмите «Искать». Можно менять формулировку — индекс пересчитывать не нужно.')
