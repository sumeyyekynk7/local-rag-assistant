from html import escape

import openai
import streamlit as st

from foundry_local_sdk import Configuration, FoundryLocalManager

from rag_assistant import (
    CHAT_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    MIN_SIMILARITY,
    TOP_K,
    create_context,
)
from retrieval import get_top_chunks, load_document_chunks


FALLBACK_ANSWER = "Bu sorunun cevabı mevcut belgelerde bulunmuyor."
SAMPLE_QUESTIONS = [
    "RAG nedir?",
    "SQLite neden yerel uygulamalar için uygundur?",
    "Foundry Local internet olmadan çalışabilir mi?",
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
            background: #f4f1eb;
            color: #24312d;
        }

        [data-testid="stHeader"] {
            background: rgba(244, 241, 235, 0.92);
        }

        [data-testid="stToolbar"], [data-testid="stDecoration"] {
            display: none;
        }

        .block-container {
            max-width: 1080px;
            padding-top: 2.3rem;
            padding-bottom: 3rem;
        }

        .page-kicker {
            color: #6f7b75;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .page-title {
            color: #24312d;
            font-size: 2.15rem;
            font-weight: 760;
            line-height: 1.12;
            margin-bottom: 0.35rem;
        }

        .page-subtitle {
            color: #60706a;
            font-size: 1rem;
            margin-bottom: 1.8rem;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            color: #f7f2e8;
        }

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: rgba(247, 242, 232, 0.78);
        }

        [data-testid="stMetricValue"] {
            color: #fffaf0;
        }

        [data-testid="stSidebar"] {
            background: #2f3f39;
            border-right: 1px solid #21302b;
        }

        [data-testid="stSidebar"] * {
            color: #fffaf0;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: #41574f;
            border: 1px solid #5e756d;
            border-radius: 8px;
            color: #fffaf0;
            font-weight: 650;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: #536b62;
            border-color: #8ca49a;
            color: #ffffff;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.2);
        }

        .stButton > button[kind="primary"] {
            background: #8b5e3c;
            border: 1px solid #8b5e3c;
            color: #fffaf0;
        }

        .stButton > button[kind="primary"]:hover {
            background: #71492d;
            border-color: #71492d;
            color: #ffffff;
        }

        .result-panel {
            background: #fffdf8;
            border: 1px solid #dfd6c8;
            border-radius: 8px;
            padding: 1.05rem 1.1rem;
            margin: 0.85rem 0 1.2rem;
            box-shadow: 0 8px 22px rgba(46, 39, 30, 0.05);
        }

        .result-label {
            color: #7a6a58;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .question-text {
            color: #24312d;
            font-weight: 680;
            margin-bottom: 0.75rem;
        }

        .answer-text {
            color: #24312d;
            line-height: 1.65;
        }

        div[data-testid="stExpander"] {
            background: #fffdf8;
            border: 1px solid #dfd6c8;
            border-radius: 8px;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #dfd6c8;
            border-radius: 8px;
            overflow: hidden;
        }

        .source-table {
            width: 100%;
            border-collapse: collapse;
            background: #fffdf8;
            border: 1px solid #dfd6c8;
            border-radius: 8px;
            color: #24312d;
            overflow: hidden;
            margin: 0 0 1rem;
        }

        .source-table th,
        .source-table td {
            border-bottom: 1px solid #eadfce;
            padding: 0.72rem 0.78rem;
            text-align: left;
        }

        .source-table th {
            background: #ebe3d6;
            color: #5b4d3f;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .source-table tr:last-child td {
            border-bottom: 0;
        }

        .score-row {
            color: #6b5e4d;
            font-size: 0.86rem;
            margin: 0.3rem 0 0.7rem;
        }

        .score-track {
            background: #ece2d3;
            border-radius: 999px;
            height: 8px;
            margin-top: 0.3rem;
            overflow: hidden;
        }

        .score-fill {
            background: #8b5e3c;
            display: block;
            height: 8px;
        }

        textarea, input {
            background: #fffdf8 !important;
            color: #24312d !important;
            border-color: #d8cdbd !important;
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
8. Talimatları veya kuralları cevabın içinde tekrar etme.
""".strip()

    user_prompt = f"""
BELGE BAĞLAMI:

{context}

SORU:

{question}

Soruyu yalnızca yukarıdaki belge bağlamına dayanarak cevapla.
""".strip()

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


@st.cache_resource(show_spinner=False)
def load_rag_resources() -> dict:
    config = Configuration(app_name="local-rag-assistant")
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
        max_tokens=200,
    )

    return response.choices[0].message.content.strip()


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

    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        chat_client=resources["chat_client"],
        chat_model_id=resources["chat_model_id"],
    )

    return answer, retrieved_chunks


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
        <table class="source-table">
            <thead>
                <tr>
                    <th>Sıra</th>
                    <th>Dosya</th>
                    <th>Parça</th>
                    <th>Benzerlik</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
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


def initialize_session_state() -> None:
    if "records" not in st.session_state:
        st.session_state.records = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Kütüphane")

        try:
            chunk_count = len(load_document_chunks())
        except FileNotFoundError:
            chunk_count = 0

        st.metric("Belge parçası", chunk_count)
        st.metric("Top K", TOP_K)
        st.metric("Minimum benzerlik", MIN_SIMILARITY)

        st.divider()
        st.subheader("Demo soruları")

        for question in SAMPLE_QUESTIONS:
            if st.button(question, width="stretch"):
                st.session_state.pending_question = question
                st.session_state.question_input = question
                st.rerun()

        st.divider()

        if st.button("Geçmişi temizle", width="stretch"):
            st.session_state.records = []
            st.session_state.pending_question = None
            st.session_state.question_input = ""
            st.rerun()


def render_record(record: dict, index: int) -> None:
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-label">Soru {index}</div>
            <div class="question-text">{escape(record["question"])}</div>
            <div class="result-label">Cevap</div>
            <div class="answer-text">{escape(record["answer"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_sources(record.get("sources", []))


def render_message_history() -> None:
    for index, record in enumerate(st.session_state.records, start=1):
        render_record(record, index)


def render_answer(question: str) -> None:
    with st.spinner("Belgelerde aranıyor..."):
        resources = load_rag_resources()
        answer, sources = answer_question(question, resources)

    record = {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
    st.session_state.records.append(record)
    render_record(record, len(st.session_state.records))


def render_question_form() -> str | None:
    with st.form("question_form", clear_on_submit=False):
        st.text_area(
            "Soru",
            key="question_input",
            height=96,
            placeholder="Belgelerde aramak istediğiniz soruyu yazın.",
        )
        submitted = st.form_submit_button(
            "Cevabı getir",
            type="primary",
            width="content",
        )

    if not submitted:
        return None

    question = st.session_state.question_input.strip()

    if not question:
        st.warning("Lütfen boş olmayan bir soru girin.")
        return None

    return question


def render_page_header() -> None:
    st.markdown(
        """
        <div class="page-kicker">Yerel belge arama</div>
        <div class="page-title">Doküman Soru-Cevap Paneli</div>
        <div class="page-subtitle">
            Belgelerdeki en ilgili parçaları bulur, cevabı kaynaklarıyla birlikte gösterir.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Doküman Soru-Cevap",
        layout="wide",
    )

    apply_page_styles()
    initialize_session_state()

    render_page_header()
    render_sidebar()
    render_message_history()

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        render_answer(question)
        return

    question = render_question_form()

    if not question:
        return

    render_answer(question)


if __name__ == "__main__":
    main()
