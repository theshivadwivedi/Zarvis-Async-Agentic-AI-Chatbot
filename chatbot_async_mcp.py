"""
Async LangGraph chatbot backend.

Gemini 2.5 Flash + tools: DuckDuckGo search, stock price, PDF RAG (FAISS),
and a calculator served over MCP (stdio). Chats persist in SQLite.
"""
import asyncio
import atexit
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Annotated, Dict, Optional, TypedDict

import aiosqlite
import httpx
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from rag_engine import ingest_pdf, rag_tool, thread_document_metadata  # noqa: F401 (re-exported for the UI)

load_dotenv()
log = logging.getLogger("chatbot")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chatbot.db"
CALCULATOR_SERVER = BASE_DIR / "calculator_server.py"

# -------------------
# Background event loop (Streamlit is sync; everything async runs here)
# -------------------
_ASYNC_LOOP = asyncio.new_event_loop()
threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True).start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    """Run a coroutine on the backend loop and wait for the result."""
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend loop without waiting."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(CALCULATOR_SERVER)],
        }
    }
)


@tool
async def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a symbol (e.g. 'AAPL', 'TSLA') using Alpha Vantage."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY is not set in the .env file."}
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.get(
                "https://www.alphavantage.co/query",
                params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as exc:
        return {"error": f"Stock API request failed: {exc}"}


# -------------------
# 3. State, graph, checkpointer
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when they help: web search for current "
    "information, the stock tool for prices, and the calculator tools for arithmetic."
)

checkpointer: Optional[AsyncSqliteSaver] = None
_DB_CONN = None


async def build_graph():
    global checkpointer, _DB_CONN

    tools = [search_tool, get_stock_price, rag_tool]
    try:
        tools.extend(await client.get_tools())
    except Exception as exc:  # keep the app usable if the MCP server fails to start
        log.warning("MCP calculator unavailable, continuing without it: %s", exc)

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState, config: RunnableConfig):
        thread_id = (config.get("configurable") or {}).get("thread_id")
        meta = thread_document_metadata(thread_id) if thread_id else {}
        doc_note = (
            f" A PDF named '{meta['filename']}' is attached to this chat. For any question about it, "
            "or vague requests like 'explain it' or 'summarize', ALWAYS call rag_tool first "
            "(for broad requests use a query like 'main topics and key points of the document') "
            "and answer from the retrieved text. Do not use web search for questions about the PDF."
            if meta
            else " No PDF is indexed for this chat."
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT + doc_note), *state["messages"]]
        response = await llm_with_tools.ainvoke(messages, config)
        return {"messages": [response]}

    _DB_CONN = await aiosqlite.connect(str(DB_PATH))
    if not hasattr(_DB_CONN, "is_alive"):  # compatibility with newer aiosqlite versions
        _DB_CONN.is_alive = lambda: True
    checkpointer = AsyncSqliteSaver(conn=_DB_CONN)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    return graph.compile(checkpointer=checkpointer)


def _shutdown():
    if _DB_CONN is not None:
        try:
            _submit_async(_DB_CONN.close()).result(timeout=5)
        except Exception:
            pass


atexit.register(_shutdown)

# -------------------
# 4. Helpers used by the UI
# -------------------
def message_text(content) -> str:
    """Plain text from a message's content (str or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def make_title(text: str, limit: int = 38) -> str:
    """Short sidebar title from the user's first message."""
    text = " ".join((text or "").split())
    if not text:
        return "New chat"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": str(thread_id)}}


async def _aload_history(thread_id: str) -> list[dict]:
    state = await chatbot.aget_state(_cfg(thread_id))
    history = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:  # tool messages are internal
            continue
        text = message_text(msg.content).strip()
        if text:  # skips AI turns that only contained tool calls
            history.append({"role": role, "content": text})
    return history


def load_history(thread_id: str) -> list[dict]:
    return run_async(_aload_history(thread_id))


async def _alist_threads() -> list[str]:
    seen: Dict[str, None] = {}
    async for cp in checkpointer.alist(None):  # newest checkpoints first
        seen[str(cp.config["configurable"]["thread_id"])] = None
    return list(seen)


async def _athread_title(thread_id: str) -> str:
    state = await chatbot.aget_state(_cfg(thread_id))
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            return make_title(message_text(msg.content))
    return "New chat"


async def _athreads_with_titles(prefix: str = "") -> list[tuple[str, str]]:
    ids = [t for t in await _alist_threads() if t.startswith(prefix)]
    titles = await asyncio.gather(*(_athread_title(t) for t in ids), return_exceptions=True)
    return [(t, ti if isinstance(ti, str) else "Untitled chat") for t, ti in zip(ids, titles)]


def retrieve_threads_with_titles(prefix: str = "") -> list[tuple[str, str]]:
    """[(thread_id, title)] newest first, only threads whose id starts with `prefix`
    (the visitor's id), so users never see each other's chats."""
    if checkpointer is None:
        return []
    return run_async(_athreads_with_titles(prefix))


chatbot = run_async(build_graph())