import sys
import streamlit as st
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
from core.vector_store import search, get_all_dates, list_documents
from pipeline import run_compliance_scan, extract_text_from_pdf
from core.supabase_client import new_session_id, save_message, load_chat_history
import os
from supabase import create_client, Client, ClientOptions
import pandas as pd
import tempfile
import uuid
import requests

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

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
            from core.comparator import run_comparison
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

# ─── MAIN CHAT UI ─────────────────────────────────────────────
st.title("💬 Ask your documents")
st.caption("Powered by Claude Sonnet · FAISS vector search · Pfizer PDIS")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            filter_doc = None if selected_doc == "All documents" else selected_doc
            
            response = requests.post(
                f"{BACKEND_URL}/api/chat",
                json={
                    "query": query,
                    "session_id": st.session_state.session_id,
                    "filter_doc": filter_doc
                },
                timeout=60
            )
            result = response.json()
            answer = result["answer"]
            route = result["route"]

        st.markdown(f"*Query type: {route}*")
        st.markdown(answer)

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