import streamlit as st
from pathlib import Path
from anthropic import Anthropic
from vector_store import search, get_all_dates, list_documents
from main import run_compliance_scan, extract_text_from_pdf
from router import is_structured_query
from supabase_client import sign_in_with_microsoft, supabase, new_session_id, save_message, load_chat_history, get_or_create_user
import os
from streamlit_oauth import OAuth2Component
import json
import base64
from supabase import create_client, Client, ClientOptions

url  = os.environ.get("SUPABASE_URL")
key  = os.environ.get("SUPABASE_ANON_KEY")

supabase: Client = create_client(
    url, key,
    options=ClientOptions(postgrest_client_timeout=10)
)

def decode_id_token(token: str) -> dict:
    """Decode JWT id_token without verification."""
    payload = token.split(".")[1]
    # add padding
    payload += "=" * (4 - len(payload) % 4)
    decoded = base64.b64decode(payload)
    return json.loads(decoded)

client = Anthropic()

# ─── PAGE CONFIG ─────────────────────────────
st.set_page_config(
    page_title="PDIS — Pfizer Document Intelligence",
    page_icon="💊",
    layout="wide"
)

# ─── AUTH ────────────────────────────────────────────────────
AZURE_CLIENT_ID     = os.environ.get("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")

oauth2 = OAuth2Component(
    client_id=AZURE_CLIENT_ID,
    client_secret=AZURE_CLIENT_SECRET,
    authorize_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
    token_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/token",
    refresh_token_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/token",
    revoke_token_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/logout"
)

if "token" not in st.session_state or not st.session_state.token:
    st.title("🔐 PDIS Login")
    st.write("Sign in with your Microsoft account to continue.")
    result = oauth2.authorize_button(
        name="Sign in with Microsoft",
        redirect_uri="http://localhost:8502",
        scope="openid email profile User.Read",
        icon="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg"
    )
    if result and "token" in result:
        st.session_state.token = result["token"]
        st.rerun()
    st.stop()

#title
st.title("💊 PDIS")
st.caption("Pharmaceutical Document Intelligence System")

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:

    st.title("💊 PDIS")
    st.divider()
    # User info + logout
    st.subheader("👤 Profile")
    id_token = st.session_state.token.get("id_token", "")
    user_info = decode_id_token(id_token) if id_token else {}
    user_email = user_info.get("email") or user_info.get("preferred_username", "User")
    # ─── USER SESSION ─────────────────────────────────────────────
    if "user_id" not in st.session_state:
        user = get_or_create_user(
            ms_user_id=user_info.get("oid", ""),
            email=user_info.get("email", ""),
            display_name=user_info.get("name", "")
        )
        st.session_state.user_id = user["id"]
        st.session_state.session_id = new_session_id()
        st.session_state.messages = load_chat_history(
            st.session_state.user_id,
            st.session_state.session_id
        )
    display_name = user_info.get("name") or \
               f"{user_info.get('given_name', '')} {user_info.get('family_name', '')}".strip() or \
               user_info.get("email", "User")
    st.caption(f"🌐 {user_email}")
    if st.button("Sign out"):
        del st.session_state["token"]
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

# ─── VERSION COMPARISON UI ───────────────────────────────────
st.divider()
st.subheader("🔄 Version Comparison")

docs = list_documents()

if len(docs) < 1:
    st.caption("Index at least 1 document to compare.")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Doc A (older)")
        doc_a_choice = st.selectbox(
            "Doc A",
            ["— select —"] + docs,
            key="doc_a",
            label_visibility="collapsed"
        )
        upload_a = st.file_uploader(
            "Or upload Doc A",
            type=["pdf"],
            key="upload_a"
        )

    with col2:
        st.caption("Doc B (newer)")
        doc_b_choice = st.selectbox(
            "Doc B",
            ["— select —"] + docs,
            key="doc_b",
            label_visibility="collapsed"
        )
        upload_b = st.file_uploader(
            "Or upload Doc B",
            type=["pdf"],
            key="upload_b"
        )

    if st.button("🔄 Compare Versions"):
        from comparator import run_comparison
        from main import extract_text_from_pdf
        import tempfile

        # resolve Doc A
        if doc_a_choice != "— select —":
            name_a = doc_a_choice
            texts_a = None
        elif upload_a:
            with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf") as f:
                f.write(upload_a.read())
                tmp_path = f.name
            _, texts_a = extract_text_from_pdf(tmp_path)
            name_a = upload_a.name
        else:
            st.error("Please select or upload Doc A")
            st.stop()

        # resolve Doc B
        if doc_b_choice != "— select —":
            name_b = doc_b_choice
            texts_b = None
        elif upload_b:
            with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf") as f:
                f.write(upload_b.read())
                tmp_path = f.name
            _, texts_b = extract_text_from_pdf(tmp_path)
            name_b = upload_b.name
        else:
            st.error("Please select or upload Doc B")
            st.stop()

        if name_a == name_b:
            st.error("Please select two different documents")
            st.stop()

        with st.spinner("Comparing documents..."):
            result = run_comparison(
                name_a, name_b, texts_a, texts_b
            )

        # ── Display results in main area ──────────────────
        st.session_state["comparison_result"] = result

def handle_structured(query: str, filter_doc: str = None) -> str:
    """Answer date-specific questions directly from index."""
    dates = get_all_dates(filter_doc=filter_doc)
    if not dates:
        return "No dates found in the indexed documents."

    # build a clean summary for the LLM
    date_lines = []
    for d in dates:
        flag = "⚠️ FLAGGED" if d.get("flagged") else "✅"
        date_lines.append(
            f"{flag} | {d['raw_date']} → {d.get('normalized','N/A')} "
            f"| Page {d['page']} | {d['doc'][:40]} | Context: {d.get('text','')[:80]}"
        )

    dates_context = "\n".join(date_lines[:80])  # cap at 80 entries

    prompt = f"""You are a pharmaceutical document compliance assistant.
Answer the user's question using ONLY the date records below.
Do not guess. If the answer isn't in the data, say so clearly.

DATE RECORDS:
{dates_context}

USER QUESTION: {query}

Answer concisely and accurately. Cite page numbers and document names."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001", #"claude-sonnet-4-5"
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def handle_semantic(query: str, filter_doc: str = None) -> str:
    """Answer open-ended questions using RAG."""
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
        model="claude-haiku-4-5-20251001", #"claude-sonnet-4-5"
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ─── MAIN CHAT UI ────────────────────────────────────────────
st.title("💬 Ask your documents")
st.caption("Powered by Claude Sonnet · FAISS vector search · Pfizer PDIS")

# init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input
if query := st.chat_input("Ask anything about your documents..."):
    save_message(st.session_state.user_id,
                 st.session_state.session_id,
                 "user", query)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            filter_doc = None if selected_doc == "All documents" else selected_doc

            # route the query
            if is_structured_query(query):
                route = "📋 Structured (date index)"
                answer = handle_structured(query, filter_doc)
            else:
                route = "🔍 Semantic (RAG)"
                answer = handle_semantic(query, selected_doc)

        st.markdown(f"*Query type: {route}*")
        st.markdown(answer)

    save_message(st.session_state.user_id,
             st.session_state.session_id,
             "assistant", answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Show comparison result if available
if "comparison_result" in st.session_state:
    r = st.session_state["comparison_result"]
    st.divider()
    st.subheader("🔄 Version Comparison Results")
    st.caption(
        f"**{r['doc_a']}** vs **{r['doc_b']}** "
        f"· {r['api_calls']} API call(s)"
    )

    # Executive summary
    st.info(r["summary"])

    # Diff table
    import pandas as pd
    rows = []
    for f in r["fields"]:
        changed = "✅ YES" if f["changed"] else "➖ NO"
        flag    = "⚠️" if f["flag"] else ""
        rows.append({
            "Field":    f["field"],
            "Doc A":    f["doc_a"],
            "Doc B":    f["doc_b"],
            "Source A": f.get("source_a", "—"),
            "Source B": f.get("source_b", "—"),
            "Changed":  changed,
            "Flag":     flag
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("Clear comparison"):
        del st.session_state["comparison_result"]
        st.rerun()