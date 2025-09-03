import streamlit as st
from ragmini import read_docs, build_corpus, build_tfidf_index, search, get_chat_provider

st.set_page_config(page_title='Mini‑RAG (FAQ + цитаты)', page_icon='📚', layout='wide')
st.title('Mini‑RAG (FAQ + цитаты)')

docs = read_docs('samples/faq')
corpus, meta = build_corpus(docs)
vect, mat = build_tfidf_index(corpus)

col1, col2 = st.columns([1,1])
with col1:
    q = st.text_input('Вопрос', 'Как перевыпустить карту и сколько это стоит?')
with col2:
    topk = st.slider('Top‑k пассажей', 3, 10, 5)

use_llm = st.toggle('Использовать LLM для финального ответа', value=False, help='Если выключено — ответ строится из лучших пассажей.')

if st.button('Искать'):
    hits = search(q, vect, mat, corpus, meta, k=topk)

    st.subheader('Найденные пассажи')
    for h in hits:
        st.write(f"**{h['title']}** — {h['url']}  \n_рейтинг: {h['score']:.3f}_")
        st.write(h['passage'])
        st.markdown('---')

    st.subheader('Ответ')
    if not use_llm:
        answer = hits[0]['passage'] if hits else 'Не найдено.'
        citations = [h['url'] for h in hits[:3]]
        st.write(answer)
        st.caption('Цитаты: ' + ' · '.join(citations))
    else:
        provider = get_chat_provider()
        context = '\n\n'.join([f"[{i+1}] {h['passage']} (src: {h['url']})" for i,h in enumerate(hits[:topk])])
        messages = [
            {"role":"system","content":"Ты ассистент банка. Отвечай кратко и по делу. Обязательно ссылайся на источники [номер] в конце фразы."},
            {"role":"user","content": f"Вопрос: {q}\n\nКонтекст:\n{context}\n\nОтветь по контексту. Если информации нет — скажи, что не нашёлся ответ."}
        ]
        try:
            ans = provider.chat(messages, temperature=0.0, max_tokens=400)
            st.write(ans)
        except Exception as e:
            st.error(f'LLM error: {e}')
