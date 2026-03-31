import streamlit as st
from pathlib import Path
from anthropic import Anthropic
from vector_store import search, get_all_dates, list_documents
from main import run_compliance_scan, extract_text_from_pdf

client = Anthropic()

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="PDIS — Pfizer Document Intelligence",
    page_icon="💊",
    layout="wide"
)

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.title("💊 PDIS")
    st.caption("Pharmaceutical Document Intelligence System")
    st.divider()

    st.subheader("📂 Upload Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded:
        save_path = Path("data/sample_docs") / uploaded.name
        save_path.write_bytes(uploaded.read())
        with st.spinner(f"Processing {uploaded.name}..."):
            run_compliance_scan(save_path)
        st.success(f"✅ {uploaded.name} indexed!")

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

def is_structured_query(query: str) -> bool:
    q = query.lower()
    
    # ONLY route to structured if explicitly asking about dates/metadata
    # Be very specific — avoid broad matches
    structured_patterns = [
        "what date", "which date", "when was", "when did",
        "what is the date", "expiry date", "expiration date",
        "effective date", "latest date", "earliest date",
        "how many dates", "list all dates", "list dates",
        "show me dates", "flagged dates", "show flagged",
        "ambiguous dates"
    ]
    
    return any(pattern in q for pattern in structured_patterns)

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
        model="claude-sonnet-4-5",
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
        model="claude-sonnet-4-5",
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
                print(f"DEBUG: is_structured={is_structured_query(query)}, query={query}")
            else:
                route = "🔍 Semantic (RAG)"
                answer = handle_semantic(query, selected_doc)

        st.markdown(f"*Query type: {route}*")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})