from html import escape
from pathlib import Path

import openai
import streamlit as st

from foundry_local_sdk import Configuration, FoundryLocalManager

from rag_assistant import (
    CHAT_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    MAX_ANSWER_TOKENS,
    MIN_SIMILARITY,
    TOP_K,
    clean_answer_text,
    create_context,
    extract_direct_answer,
)
from retrieval import get_top_chunks, load_document_chunks


FALLBACK_ANSWER = "Bu sorunun cevabı mevcut belgelerde bulunmuyor."
FOUNDRY_DATA_DIR = Path("data/foundry_local").resolve()
FOUNDRY_LOGS_DIR = FOUNDRY_DATA_DIR / "logs"
SAMPLE_QUESTIONS = [
    "Foundry Local internet olmadan çalışabilir mi?",
    "RAG nedir?",
    "SQLite neden yerel uygulamalar için uygundur?",
    "Türkiye'nin başkenti neresidir?",
]


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: #f7f8f5;
            color: #111827;
            font-family: "Segoe UI", Inter, Arial, sans-serif;
        }

        [data-testid="stHeader"] {
            background: transparent;
            height: 0;
            visibility: hidden;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none;
        }

        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            display: block !important;
            min-width: 20rem !important;
            transform: translateX(0) !important;
            visibility: visible !important;
            width: 20rem !important;
        }

        [data-testid="stSidebarContent"] {
            width: 20rem !important;
        }

        .block-container {
            max-width: 980px;
            padding-top: 0.75rem;
            padding-bottom: 6rem;
        }

        .assistant-header {
            border-bottom: 1px solid #d9e2dc;
            margin-bottom: 1.2rem;
            padding-bottom: 1rem;
        }

        .assistant-kicker {
            color: #21725f;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin-bottom: 0.28rem;
            text-transform: uppercase;
        }

        .assistant-title {
            color: #111827;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }

        .assistant-subtitle {
            color: #5c6b64;
            font-size: 1rem;
            line-height: 1.55;
            max-width: 720px;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.85rem;
        }

        .pill {
            background: #ffffff;
            border: 1px solid #d9e2dc;
            border-radius: 999px;
            color: #31423a;
            font-size: 0.82rem;
            font-weight: 700;
            padding: 0.38rem 0.72rem;
        }

        [data-testid="stSidebar"] {
            background: #153c34;
            border-right: 1px solid #0e2b25;
        }

        [data-testid="stSidebar"] * {
            color: #f8fffb;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffffff;
            font-weight: 800;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stCaptionContainer {
            color: rgba(248, 255, 251, 0.78);
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.18);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            padding: 0.72rem 0.82rem;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetric"] label {
            color: rgba(248, 255, 251, 0.78) !important;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-size: 1.45rem;
            font-weight: 800;
        }

        .stButton > button {
            background: #ffffff;
            border: 1px solid #cfd9d3;
            border-radius: 8px;
            color: #17221d;
            font-weight: 700;
            min-height: 2.65rem;
            white-space: normal;
        }

        .stButton > button:hover {
            border-color: #21725f;
            color: #153c34;
        }

        [data-testid="stFormSubmitButton"] button {
            background: #21725f !important;
            border: 1px solid #21725f !important;
            color: #ffffff !important;
            width: 100%;
        }

        [data-testid="stFormSubmitButton"] button:hover {
            background: #195947 !important;
            border-color: #195947 !important;
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            color: #ffffff;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255, 255, 255, 0.18);
            border-color: rgba(255, 255, 255, 0.52);
            color: #ffffff;
        }

        .empty-panel {
            background: #ffffff;
            border: 1px solid #d9e2dc;
            border-radius: 8px;
            margin: 1.2rem 0 1rem;
            padding: 1rem;
        }

        .empty-title {
            color: #17221d;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.28rem;
        }

        .empty-copy {
            color: #5c6b64;
            line-height: 1.55;
        }

        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #d9e2dc;
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(17, 24, 39, 0.06);
            margin: 1rem 0 1.2rem;
            padding: 0.9rem 0.95rem 0.75rem;
        }

        div[data-testid="stForm"] label {
            color: #31423a !important;
            font-weight: 800 !important;
        }

        input, textarea {
            background: #fbfdfb !important;
            border: 1px solid #cfd9d3 !important;
            border-radius: 8px !important;
            color: #111827 !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #75837d !important;
            opacity: 1 !important;
        }

        .message-row {
            display: flex;
            margin: 0.72rem 0;
            width: 100%;
        }

        .message-row.user {
            justify-content: flex-end;
        }

        .message-row.assistant {
            justify-content: flex-start;
        }

        .message-bubble {
            border-radius: 8px;
            line-height: 1.62;
            max-width: min(78%, 740px);
            padding: 0.88rem 1rem;
        }

        .message-bubble.user {
            background: #21725f;
            color: #ffffff;
        }

        .message-bubble.assistant {
            background: #ffffff;
            border: 1px solid #d9e2dc;
            box-shadow: 0 10px 24px rgba(17, 24, 39, 0.05);
            color: #111827;
        }

        .message-label {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }

        .message-bubble.user .message-label {
            color: rgba(255, 255, 255, 0.78);
        }

        .message-bubble.assistant .message-label {
            color: #21725f;
        }

        .sources-title {
            color: #66756e;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin: 0.65rem 0 0.35rem;
            text-transform: uppercase;
        }

        .source-table {
            background: #ffffff;
            border: 1px solid #d9e2dc;
            border-collapse: collapse;
            border-radius: 8px;
            color: #17221d;
            margin: 0 0 0.7rem;
            overflow: hidden;
            width: 100%;
        }

        .source-table th,
        .source-table td {
            border-bottom: 1px solid #edf1ee;
            padding: 0.62rem 0.72rem;
            text-align: left;
        }

        .source-table th {
            background: #edf3ef;
            color: #405249;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .source-table tr:last-child td {
            border-bottom: 0;
        }

        .score-row {
            color: #5c6b64;
            font-size: 0.86rem;
            margin: 0.35rem 0 0.75rem;
        }

        .score-track {
            background: #dfe8e3;
            border-radius: 999px;
            height: 8px;
            margin-top: 0.3rem;
            overflow: hidden;
        }

        .score-fill {
            background: #21725f;
            display: block;
            height: 8px;
        }

        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #d9e2dc;
            border-radius: 8px;
        }

        [data-testid="stAlert"] {
            background: #fff7d6;
            border: 1px solid #e3c45d;
            border-radius: 8px;
        }

        [data-testid="stAlert"] * {
            color: #433814 !important;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .assistant-title {
                font-size: 1.65rem;
            }

            .message-bubble {
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_answer_messages(question: str, context: str) -> list[dict[str, str]]:
    system_prompt = """
Sen yalnızca verilen belge parçalarındaki bilgileri kullanan bir belge soru-cevap sistemisin.

Kesin kurallar:
1. Cevabını yalnızca BELGE BAĞLAMI içindeki bilgilerden oluştur.
2. Belgede yazan terimleri, isimleri ve açılımları aynen kullan.
3. Bağlamda olmayan hiçbir bilgi ekleme.
4. Kendi genel bilgini kesinlikle kullanma.
5. Belge bağlamı soruyla yalnızca dolaylı olarak ilişkiliyse cevap üretme.
6. Sorunun cevabı bağlamda yoksa yalnızca şu cümleyi yaz:
   Bu sorunun cevabı mevcut belgelerde bulunmuyor.
7. Cevabı Türkçe, kısa ve doğrudan yaz.
8. Kullanıcı kaç, hangi veya liste sorusu sorarsa bütün kaynak parçalarını birlikte değerlendir.
9. En fazla iki cümle kur.
10. Talimatları veya kuralları cevabın içinde tekrar etme.
11. Aynı kelimeyi, heceyi veya cümleyi tekrar etme.
""".strip()

    user_prompt = f"""
BELGE BAĞLAMI:

{context}

SORU:

{question}

Soruyu yalnızca yukarıdaki belge bağlamına dayanarak cevapla.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@st.cache_resource(show_spinner=False)
def load_rag_resources() -> dict:
    FOUNDRY_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config = Configuration(
        app_name="local-rag-assistant",
        app_data_dir=str(FOUNDRY_DATA_DIR),
        logs_dir=str(FOUNDRY_LOGS_DIR),
    )
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance
    manager.download_and_register_eps()

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_NAME)
    embedding_model.download()
    embedding_model.load()

    chat_model = manager.catalog.get_model(CHAT_MODEL_NAME)
    chat_model.download()
    chat_model.load()

    manager.start_web_service()

    chat_client = openai.OpenAI(
        base_url=f"{manager.urls[0]}/v1",
        api_key="none",
    )

    return {
        "embedding_client": embedding_model.get_embedding_client(),
        "chat_client": chat_client,
        "chat_model_id": chat_model.id,
        "manager": manager,
        "embedding_model": embedding_model,
        "chat_model": chat_model,
    }


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    chat_client: openai.OpenAI,
    chat_model_id: str,
) -> str:
    context = create_context(retrieved_chunks)

    response = chat_client.chat.completions.create(
        model=chat_model_id,
        messages=build_answer_messages(question, context),
        temperature=0,
        max_tokens=MAX_ANSWER_TOKENS,
    )

    return clean_answer_text(response.choices[0].message.content)


def answer_question(question: str, resources: dict) -> tuple[str, list[dict]]:
    retrieved_chunks = get_top_chunks(
        query=question,
        embedding_client=resources["embedding_client"],
        top_k=TOP_K,
    )

    if not retrieved_chunks:
        return FALLBACK_ANSWER, []

    if retrieved_chunks[0]["similarity"] < MIN_SIMILARITY:
        return FALLBACK_ANSWER, retrieved_chunks

    direct_answer = extract_direct_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    if direct_answer:
        return direct_answer, retrieved_chunks

    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        chat_client=resources["chat_client"],
        chat_model_id=resources["chat_model_id"],
    )

    return answer, retrieved_chunks


def get_chunk_count() -> int:
    try:
        return len(load_document_chunks())
    except FileNotFoundError:
        return 0


def initialize_session_state() -> None:
    if "records" not in st.session_state:
        st.session_state.records = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def render_similarity_bar(similarity: float) -> None:
    score = max(0.0, min(similarity, 1.0))
    st.markdown(
        f"""
        <div class="score-row">
            Benzerlik: {similarity:.4f}
            <div class="score-track">
                <span class="score-fill" style="width: {score * 100:.1f}%"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(chunks: list[dict]) -> None:
    if not chunks:
        st.info("Kaynak bulunamadı.")
        return

    rows = "\n".join(
        f"""
        <tr>
            <td>{index}</td>
            <td>{escape(chunk["source"])}</td>
            <td>{chunk["chunk_index"]}</td>
            <td>{chunk["similarity"]:.4f}</td>
        </tr>
        """.strip()
        for index, chunk in enumerate(chunks, start=1)
    )

    st.markdown(
        f"""
        <div class="sources-title">Kaynaklar</div>
        <table class="source-table">
            <thead>
                <tr>
                    <th>Sıra</th>
                    <th>Dosya</th>
                    <th>Parça</th>
                    <th>Benzerlik</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Kaynak metinleri"):
        for index, chunk in enumerate(chunks, start=1):
            st.markdown(
                f"**{index}. {chunk['source']} / chunk {chunk['chunk_index']}**"
            )
            render_similarity_bar(chunk["similarity"])
            st.write(chunk["content"])


def submit_question(question: str) -> None:
    cleaned_question = question.strip()

    if not cleaned_question:
        return

    with st.spinner("Belgelerde aranıyor..."):
        resources = load_rag_resources()
        answer, sources = answer_question(cleaned_question, resources)

    st.session_state.records.append(
        {
            "question": cleaned_question,
            "answer": answer,
            "sources": sources,
        }
    )


def queue_question(question: str) -> None:
    st.session_state.pending_question = question
    st.rerun()


def format_message(content: str) -> str:
    return escape(content).replace("\n", "<br>")


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Local RAG")
        st.caption("Belge tabanlı yerel asistan")

        st.metric("Belge parçası", get_chunk_count())
        st.metric("Kaynak sayısı", TOP_K)
        st.metric("Eşik", MIN_SIMILARITY)

        st.divider()
        st.subheader("Hazır sorular")

        for index, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key=f"sidebar_sample_{index}", width="stretch"):
                queue_question(question)

        st.divider()

        if st.button("Sohbeti temizle", width="stretch"):
            st.session_state.records = []
            st.session_state.pending_question = None
            st.rerun()


def render_header() -> None:
    st.markdown(
        f"""
        <div class="assistant-header">
            <div class="assistant-kicker">Yerel RAG asistanı</div>
            <div class="assistant-title">Belgelerinle sohbet et</div>
            <div class="assistant-subtitle">
                Sorunu yaz, asistan en yakın belge parçalarını bulup kaynaklı cevap versin.
            </div>
            <div class="pill-row">
                <span class="pill">{get_chunk_count()} belge parçası</span>
                <span class="pill">{TOP_K} kaynak kullanılır</span>
                <span class="pill">Eşik {MIN_SIMILARITY}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-panel">
            <div class="empty-title">Asistan hazır.</div>
            <div class="empty-copy">
                Aşağıdan bir soru yazabilir veya hızlı sorulardan biriyle demo başlatabilirsin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_composer() -> str | None:
    with st.form("assistant_composer", clear_on_submit=True):
        input_column, button_column = st.columns([6, 1])

        with input_column:
            question = st.text_input(
                "Soru",
                placeholder="Belgeler hakkında soru sor...",
                label_visibility="collapsed",
            )

        with button_column:
            submitted = st.form_submit_button("Gönder", width="stretch")

    if not submitted:
        return None

    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning("Lütfen bir soru yaz.")
        return None

    return cleaned_question


def render_conversation() -> None:
    if not st.session_state.records:
        render_empty_state()
        return

    for record in st.session_state.records:
        st.markdown(
            f"""
            <div class="message-row user">
                <div class="message-bubble user">
                    <div class="message-label">Sen</div>
                    {format_message(record["question"])}
                </div>
            </div>
            <div class="message-row assistant">
                <div class="message-bubble assistant">
                    <div class="message-label">Asistan</div>
                    {format_message(record["answer"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_sources(record.get("sources", []))


def main() -> None:
    st.set_page_config(
        page_title="Local RAG Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_page_styles()
    initialize_session_state()
    render_sidebar()
    render_header()

    if st.session_state.pending_question:
        pending_question = st.session_state.pending_question
        st.session_state.pending_question = None
        submit_question(pending_question)

    render_conversation()

    prompt = render_composer()

    if prompt:
        submit_question(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
