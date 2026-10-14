import html
import queue
import uuid

import streamlit as st
from langchain_core.messages import AIMessageChunk, HumanMessage

from chatbot_async_mcp import (
    chatbot,
    load_history,
    make_title,
    message_text,
    retrieve_threads_with_titles,
    run_async,
    submit_async_task,
)
from rag_engine import ingest_pdf, thread_document_metadata

st.set_page_config(
    page_title="Zarvis", page_icon=":material/chat_bubble:", layout="centered"
)

# ============================== Styling ==============================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;500&display=swap');

:root {
    --bg: #262624; --side: #262624; --ink: #f3f1ea; --muted: #a3a199; --input: #30302e;
    --line: #3d3c38; --accent: #d97757; --hover: #30302e; --active: #3a3935; --user: #30302e;
}
.stApp {background: var(--bg);}
html, body, .stApp, .stMarkdown, [data-testid="stChatInput"] textarea,
[data-testid="stSidebar"] button {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
header[data-testid="stHeader"] {background: transparent;}
#MainMenu, footer, .stDeployButton, [data-testid="stToolbarActions"],
[data-testid="stDecoration"] {display: none !important;}
.block-container {max-width: 760px; padding-top: 2.2rem; padding-bottom: 7rem;}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {background: var(--side) !important;}
[data-testid="stSidebar"] {border-right: 1px solid var(--line);}
.brand {font-family: 'Source Serif 4', Georgia, serif; font-size: 1.35rem;
    color: var(--ink); padding: .2rem .5rem 1rem;}
.side-label {font-size: .74rem; color: var(--muted); padding: 1.1rem .6rem .35rem;}
[data-testid="stSidebar"] .stButton button {
    width: 100%; justify-content: flex-start; background: transparent; border: none;
    box-shadow: none; color: var(--ink); padding: .5rem .6rem; border-radius: 10px;
    font-size: .9rem; font-weight: 400; min-height: 0;
}
[data-testid="stSidebar"] .stButton button:hover {background: var(--hover); color: var(--ink);}
[data-testid="stSidebar"] .stButton button p {
    text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%;
}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: var(--active) !important; color: var(--ink) !important;
}
.st-key-new_chat button {color: var(--accent) !important; font-weight: 500 !important;}

/* ---------- Welcome ---------- */
.hero {text-align: center; padding: 16vh 0 1.6rem;}
.hero h1 {font-family: 'Source Serif 4', Georgia, serif; font-weight: 400; font-size: 2.5rem;
    color: var(--ink); letter-spacing: -.01em; margin: 0; padding: 0;}
.st-key-pills .stButton button {
    border-radius: 999px; border: 1px solid var(--line); background: transparent;
    color: var(--muted); font-size: .82rem; padding: .4rem .6rem; min-height: 0; width: 100%;
}
.st-key-pills .stButton button:hover {border-color: #55534d; color: var(--ink); background: var(--hover);}

/* ---------- Messages ---------- */
[data-testid="stChatMessage"] {background: transparent; border: none; padding: .4rem 0; gap: 0;}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] {display: none !important;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--user); border-radius: 18px; padding: .7rem 1.1rem;
    width: fit-content; max-width: 92%;
}
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li {
    line-height: 1.7; color: var(--ink);
}
[data-testid="stExpander"], [data-testid="stStatus"] {
    border: 1px solid var(--line) !important; border-radius: 12px; background: transparent;
}

/* ---------- Input ---------- */
[data-testid="stBottom"], [data-testid="stBottom"] > div {background: var(--bg) !important;}
[data-testid="stChatInput"] {
    background: var(--input); border: 1px solid var(--line); border-radius: 22px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, .30);
}
[data-testid="stChatInput"] textarea {color: var(--ink);}
[data-testid="stChatInputSubmitButton"] {
    background: var(--accent) !important; color: #fff !important; border-radius: 10px;
}

/* ---------- Document chip ---------- */
.doc-chip {display: inline-flex; align-items: center; gap: .5rem; border: 1px solid var(--line);
    background: var(--input); border-radius: 999px; padding: .28rem .8rem; font-size: .82rem;
    color: var(--ink); margin: 0 0 .6rem 0;}
.doc-chip span {color: var(--muted);}
.doc-chip svg {color: var(--accent);}

/* ---------- Fixes: left-aligned sidebar items, tighter spacing, no Deploy button ---------- */
[data-testid="stSidebar"] .stButton button > div {justify-content: flex-start !important; width: 100%;}
[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] {width: 100%;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .15rem;}
[data-testid="stAppDeployButton"], .stAppDeployButton, [data-testid="stMainMenu"] {display: none !important;}
[data-testid="stChatInput"] textarea::placeholder {color: var(--muted);}

/* ---------- Dark chat input (override Streamlit's inner white layers) ---------- */
[data-testid="stChatInput"], [data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb], [data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {background: var(--input) !important; color: var(--ink) !important;}
[data-testid="stChatInput"] {border: 1px solid var(--line) !important; border-radius: 22px !important;}
[data-testid="stChatInput"] button:not([data-testid="stChatInputSubmitButton"]) {color: var(--muted) !important;}
.brand {display: flex; align-items: center; gap: .6rem;}
.brand .mark {display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px;
    border-radius: 8px; background: var(--accent); color: #fff; font-size: 1rem; font-weight: 600;}

/* ---------- Sidebar: higher brand, spacing, visible open/close toggle ---------- */
[data-testid="stSidebarHeader"] {height: 2.8rem !important; min-height: 0 !important;
    padding: .7rem .9rem 0 !important; margin-bottom: 0 !important;}
[data-testid="stSidebarUserContent"] {padding-top: .2rem !important;}
.brand {padding: 0 .5rem .9rem !important;}
.side-label {display: block; margin: 1.7rem 0 .8rem !important; padding: 0 .6rem !important; line-height: 1.3;}
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseButton"] button,
[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] button,
[data-testid="collapsedControl"], [data-testid="collapsedControl"] button {
    opacity: 1 !important; visibility: visible !important; color: var(--muted) !important;
}
[data-testid="stSidebarCollapseButton"] button:hover, [data-testid="stExpandSidebarButton"] button:hover,
[data-testid="collapsedControl"] button:hover {color: var(--ink) !important; background: var(--hover) !important;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================== Helpers ==============================
ss = st.session_state

TOOL_DONE = {
    "duckduckgo_search": "Web search",
    "get_stock_price": "Stock price",
    "rag_tool": "PDF search",
    "add": "Calculator",
    "subtract": "Calculator",
    "multiply": "Calculator",
    "divide": "Calculator",
    "percentage": "Calculator",
}
TOOL_RUNNING = {
    "duckduckgo_search": "Searching the web",
    "get_stock_price": "Checking the stock price",
    "rag_tool": "Reading your PDF",
    "add": "Calculating",
    "subtract": "Calculating",
    "multiply": "Calculating",
    "divide": "Calculating",
    "percentage": "Calculating",
}
EXAMPLES = [
    ("Search the web", "What are the latest AI news today?"),
    ("Stock price", "What is the current price of AAPL?"),
    ("Calculate", "What is 18% of 2450?"),
    ("Summarize a PDF", "Summarize the attached PDF"),
]
DOC_ICON = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>'
)


def doc_chip(name: str, detail: str = "") -> str:
    extra = f"<span>{html.escape(detail)}</span>" if detail else ""
    return f'<div class="doc-chip">{DOC_ICON}{html.escape(name)}{extra}</div>'


def set_thread(thread_id: str):
    ss.thread_id = thread_id
    st.query_params["chat"] = thread_id  # keeps the same chat open after a page refresh


def new_chat():
    set_thread(str(uuid.uuid4()))
    ss.message_history = []


def open_thread(thread_id: str):
    set_thread(thread_id)
    ss.message_history = load_history(thread_id)


def use_example(prompt: str):
    ss.pending_prompt = prompt


def valid_uuid(value) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


# ======================= Session initialization ======================
if "thread_id" not in ss:
    ss.threads = dict(retrieve_threads_with_titles())  # {thread_id: title}, newest first
    saved = st.query_params.get("chat")
    if saved and valid_uuid(saved):  # page was refreshed: reopen the same chat
        ss.thread_id = saved
        try:
            ss.message_history = load_history(saved) if saved in ss.threads else []
        except Exception:
            ss.message_history = []
    else:
        ss.thread_id = str(uuid.uuid4())
        ss.message_history = []
    st.query_params["chat"] = ss.thread_id

# ============================== Sidebar ==============================
with st.sidebar:
    st.markdown('<div class="brand"><span class="mark">Z</span>Zarvis</div>', unsafe_allow_html=True)
    st.button("New chat", key="new_chat", icon=":material/add:", on_click=new_chat,
              use_container_width=True)
    st.markdown('<div class="side-label">Recents</div>', unsafe_allow_html=True)
    threads_box = st.container()  # filled after the chat runs so new chats appear instantly


def render_threads():
    with threads_box:
        if not ss.threads:
            st.caption("No conversations yet.")
        for tid, title in ss.threads.items():
            st.button(
                title,
                key=f"thread_{tid}",
                help=title,
                on_click=open_thread,
                args=(tid,),
                type="primary" if tid == ss.thread_id else "secondary",
                use_container_width=True,
            )


# ============================== Main area ==============================
doc_slot = st.container()  # shows the PDF attached to this chat (filled at the end)

submitted = st.chat_input(
    "How can I help you today?", accept_file=True, file_type=["pdf"]
)
pending = ss.pop("pending_prompt", None)

user_text, attachment = "", None
if submitted is not None:
    user_text = (submitted["text"] or "").strip()
    files = list(submitted["files"] or [])
    attachment = files[0] if files else None
elif pending:
    user_text = pending

# Index the attached PDF before answering
attached_name = None
if attachment is not None:
    with st.spinner(f"Reading {attachment.name}"):
        try:
            run_async(ingest_pdf(attachment.getvalue(), ss.thread_id, attachment.name))
            attached_name = attachment.name
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")
    if attached_name and not user_text:
        st.rerun()  # nothing to answer yet; just show the attached document

if user_text:
    if ss.thread_id not in ss.threads:
        ss.threads = {ss.thread_id: make_title(user_text), **ss.threads}
    ss.message_history.append(
        {"role": "user", "content": user_text, "attachment": attached_name}
    )

# Welcome screen (only for an empty chat)
if not ss.message_history:
    st.markdown('<div class="hero"><h1>How can I help you today?</h1></div>', unsafe_allow_html=True)
    with st.container(key="pills"):
        cols = st.columns(len(EXAMPLES))
        for i, (col, (label, prompt)) in enumerate(zip(cols, EXAMPLES)):
            col.button(label, key=f"example_{i}", on_click=use_example, args=(prompt,),
                       use_container_width=True)

# Conversation
for message in ss.message_history:
    with st.chat_message(message["role"]):
        if message.get("attachment"):
            st.markdown(doc_chip(message["attachment"]), unsafe_allow_html=True)
        st.markdown(message["content"])

# Answer the new message
if user_text:
    config = {
        "configurable": {"thread_id": ss.thread_id},
        "metadata": {"thread_id": ss.thread_id},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_area = st.container()
        ui = {"status": None, "tools": []}

        def reply_stream():
            events: queue.Queue = queue.Queue()

            async def pump():
                try:
                    async for chunk, _meta in chatbot.astream(
                        {"messages": [HumanMessage(content=user_text)]},
                        config=config,
                        stream_mode="messages",
                    ):
                        events.put(("chunk", chunk))
                except Exception as exc:
                    events.put(("error", exc))
                finally:
                    events.put(None)

            submit_async_task(pump())

            while True:
                item = events.get()
                if item is None:
                    break
                kind, payload = item
                if kind == "error":
                    raise payload
                if not isinstance(payload, AIMessageChunk):
                    continue  # ignore tool results

                for call in payload.tool_calls:
                    name = call.get("name")
                    if name and name not in ui["tools"]:
                        ui["tools"].append(name)
                        label = TOOL_RUNNING.get(name, "Working")
                        if ui["status"] is None:
                            ui["status"] = status_area.status(label, expanded=False)
                        else:
                            ui["status"].update(label=label, state="running")

                text = message_text(payload.content)
                if text:
                    yield text

        try:
            ai_message = st.write_stream(reply_stream())
        except Exception as exc:
            ai_message = ""
            st.error(f"Something went wrong: {exc}")

        if ui["status"] is not None:
            used = ", ".join(dict.fromkeys(TOOL_DONE.get(t, t) for t in ui["tools"]))
            ui["status"].update(label=f"Used: {used}", state="complete", expanded=False)

    if ai_message:
        ss.message_history.append({"role": "assistant", "content": ai_message})

render_threads()

with doc_slot:
    doc = thread_document_metadata(ss.thread_id)
    if doc:
        st.markdown(
            doc_chip(doc["filename"], f'{doc["documents"]} pages'), unsafe_allow_html=True
        )