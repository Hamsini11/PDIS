import streamlit as st
from pathlib import Path
from anthropic import Anthropic
from vector_store import search, get_all_dates, list_documents
from main import run_compliance_scan, extract_text_from_pdf
from router import is_structured_query
from supabase_client import new_session_id, save_message, load_chat_history
import os
from supabase import create_client, Client, ClientOptions
import pandas as pd
import tempfile
import uuid

client = Anthropic()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")

supabase: Client = create_client(
    url, key,
    options=ClientOptions(postgrest_client_timeout=10)
)

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDIS — Pfizer Document Intelligence",
    page_icon="💊",
    layout="wide"
)

# ─── SESSION INIT ─────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.session_id = new_session_id()
    st.session_state.messages = []

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.title("💊 PDIS")
    st.caption("Pharmaceutical Document Intelligence System")
    st.divider()

    st.subheader("📂 Upload Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded:
        save_path = Path("data/sample_docs") / uploaded.name
        if not save_path.exists():
            save_path.write_bytes(uploaded.read())
            with st.spinner(f"Processing {uploaded.name}..."):
                run_compliance_scan(save_path)
            st.success(f"✅ {uploaded.name} indexed!")
        else:
            st.info(f"Already indexed: {uploaded.name}")

    st.divider()
    st.subheader("📄 Indexed Documents")
    docs = list_documents()
    if docs:
        for doc in docs:
            st.markdown(f"• `{doc}`")
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    selected_doc = st.selectbox(
        "Filter by document (optional)",
        ["All documents"] + docs
    )

    if st.button("🚩 Show Flagged Dates"):
        filter_doc = None if selected_doc == "All documents" else selected_doc
        flagged = get_all_dates(filter_doc=filter_doc, flagged_only=True)
        if flagged:
            st.warning(f"{len(flagged)} dates flagged for review")
            for f in flagged:
                st.markdown(f"**{f['raw_date']}** — Page {f['page']} — `{f['doc'][:30]}`")
        else:
            st.success("No flagged dates!")

    st.divider()
    st.subheader("🔄 Version Comparison")
    docs_for_compare = list_documents()
    if len(docs_for_compare) < 1:
        st.caption("Index at least 1 document to compare.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Doc A (older)")
            doc_a_choice = st.selectbox(
                "Doc A", ["— select —"] + docs_for_compare,
                key="doc_a", label_visibility="collapsed"
            )
            upload_a = st.file_uploader("Or upload Doc A", type=["pdf"], key="upload_a")
        with col2:
            st.caption("Doc B (newer)")
            doc_b_choice = st.selectbox(
                "Doc B", ["— select —"] + docs_for_compare,
                key="doc_b", label_visibility="collapsed"
            )
            upload_b = st.file_uploader("Or upload Doc B", type=["pdf"], key="upload_b")

        if st.button("🔄 Compare Versions"):
            from comparator import run_comparison
            if doc_a_choice != "— select —":
                name_a, texts_a = doc_a_choice, None
            elif upload_a:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(upload_a.read())
                _, texts_a = extract_text_from_pdf(f.name)
                name_a = upload_a.name
            else:
                st.error("Please select or upload Doc A")
                st.stop()

            if doc_b_choice != "— select —":
                name_b, texts_b = doc_b_choice, None
            elif upload_b:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(upload_b.read())
                _, texts_b = extract_text_from_pdf(f.name)
                name_b = upload_b.name
            else:
                st.error("Please select or upload Doc B")
                st.stop()

            if name_a == name_b:
                st.error("Please select two different documents")
                st.stop()

            with st.spinner("Comparing documents..."):
                result = run_comparison(name_a, name_b, texts_a, texts_b)
            st.session_state["comparison_result"] = result


# ─── QUERY HANDLERS ───────────────────────────────────────────
def handle_structured(query: str, filter_doc: str = None) -> str:
    dates = get_all_dates(filter_doc=filter_doc)
    if not dates:
        return "No dates found in the indexed documents."

    date_lines = []
    for d in dates:
        flag = "⚠️ FLAGGED" if d.get("flagged") else "✅"
        date_lines.append(
            f"{flag} | {d['raw_date']} → {d.get('normalized','N/A')} "
            f"| Page {d['page']} | {d['doc'][:40]} | Context: {d.get('text','')[:80]}"
        )

    prompt = f"""You are a pharmaceutical document compliance assistant.
Answer the user's question using ONLY the date records below.
Do not guess. If the answer isn't in the data, say so clearly.

DATE RECORDS:
{chr(10).join(date_lines[:80])}

USER QUESTION: {query}

Answer concisely and accurately. Cite page numbers and document names."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def handle_semantic(query: str, filter_doc: str = None) -> str:
    filter_arg = None if filter_doc == "All documents" else filter_doc
    chunks = search(query, top_k=5, filter_doc=filter_arg)

    if not chunks:
        return "No relevant content found. Please index some documents first."

    context = "\n\n---\n\n".join([
        f"[{c['doc']} | Page {c['page']}]\n{c['text']}"
        for c in chunks
    ])

    prompt = f"""You are a pharmaceutical document compliance assistant.
Answer the user's question using ONLY the context below.
Always cite the document name and page number.
If the answer isn't in the context, say: "This information was not found in the indexed documents."
Never guess or hallucinate.

CONTEXT:
{context}

USER QUESTION: {query}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ─── MAIN CHAT UI ─────────────────────────────────────────────
st.title("💬 Ask your documents")
st.caption("Powered by Claude Sonnet · FAISS vector search · Pfizer PDIS")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask anything about your documents..."):
    # save_message(st.session_state.user_id,
    #              st.session_state.session_id,
    #              "user", query)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            filter_doc = None if selected_doc == "All documents" else selected_doc
            if is_structured_query(query):
                route = "📋 Structured (date index)"
                answer = handle_structured(query, filter_doc)
            else:
                route = "🔍 Semantic (RAG)"
                answer = handle_semantic(query, selected_doc)
        st.markdown(f"*Query type: {route}*")
        st.markdown(answer)

    # save_message(st.session_state.user_id,
    #              st.session_state.session_id,
    #              "assistant", answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ─── COMPARISON RESULTS ───────────────────────────────────────
if "comparison_result" in st.session_state:
    r = st.session_state["comparison_result"]
    st.divider()
    st.subheader("🔄 Version Comparison Results")
    st.caption(f"**{r['doc_a']}** vs **{r['doc_b']}** · {r['api_calls']} API call(s)")
    st.info(r["summary"])

    rows = []
    for f in r["fields"]:
        rows.append({
            "Field":    f["field"],
            "Doc A":    f["doc_a"],
            "Doc B":    f["doc_b"],
            "Source A": f.get("source_a", "—"),
            "Source B": f.get("source_b", "—"),
            "Changed":  "✅ YES" if f["changed"] else "➖ NO",
            "Flag":     "⚠️" if f["flag"] else ""
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.button("Clear comparison"):
        del st.session_state["comparison_result"]
        st.rerun()