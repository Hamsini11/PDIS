import sys
import streamlit as st
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

import os
import uuid
import requests
import pandas as pd
import tempfile

from vector_store import search, get_all_dates, list_documents
from pipeline import run_compliance_scan, extract_text_from_pdf
from supabase_client import new_session_id, save_message, load_chat_history

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ─── PAGE CONFIG (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title="PDIS — Pfizer Document Intelligence",
    page_icon="💊",
    layout="wide"
)

# ─── SESSION INIT (before auth gate) ─────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()

# ─── OAUTH CALLBACK HANDLER ───────────────────────────────────
params = st.query_params
if "state" in params and "access_token" not in st.session_state:
    state = params["state"]
    response = requests.get(f"{BACKEND_URL}/auth/token?state={state}")
    if response.status_code == 200:
        token_data = response.json()
        st.session_state.access_token = token_data["access_token"]
        st.session_state.user_id = token_data["user_id"]
        st.session_state.session_id = token_data["session_id"]
        st.session_state.user_email = token_data["email"]
        st.session_state.user_name = token_data.get("display_name", "")
        st.query_params.clear()
        st.rerun()

# ─── AUTH GATE ────────────────────────────────────────────────
if "access_token" not in st.session_state:
    st.title("🔐 PDIS — Secure Login")
    st.write("Sign in with your Microsoft SSO account.")
    if st.button("Sign in with Microsoft"):
        state = str(uuid.uuid4())
        st.session_state.pending_state = state
        login_url = f"{BACKEND_URL}/auth/login?state={state}"
        st.markdown(f"[Click here to sign in]({login_url})")
    st.stop()

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:

    with st.sidebar:
        st.subheader("My Profile")
        user_email = st.session_state.get('user_email', 'User')
        display_name = st.session_state.get('user_name', '')
        
        st.markdown(f"**👤 {display_name}**" if display_name else user_email.split("@")[0])
        if st.button("Sign out", help="Click to logout"):
            for key in ["access_token", "user_id",
                    "session_id", "user_email", 
                    "user_name", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()
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
# st.title("💊 PDIS")
# st.caption("Pharmaceutical Document Intelligence System")
# st.divider()
st.markdown("""
<style>
.fixed-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: #0e1117;
    padding: 1rem 2rem;
    text-align: center;
    border-bottom: 1px solid #333;
}
</style>
<div class="fixed-header">
    <h2>💊 PDIS</h2>
    <p style="color: gray; font-size: 0.8rem;">
    Pharmaceutical Document Intelligence System
    </p>
</div>
""", unsafe_allow_html=True)
st.subheader("💬 Ask your documents")
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