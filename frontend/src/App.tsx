// === App.tsx (with left sidebar: chat thread selector + rename + Enter-to-send) ===
import React, { useEffect, useMemo, useRef, useState, Suspense} from "react";
import AdminDocs from "./AdminDocs";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/* =========================
   Single-file React frontend for ElderlyCare HK
   + Left sidebar for chat threads (localStorage)
   + Rename chat titles (inline or via button)
   + Textarea: Enter=Send, Shift+Enter=newline
   ========================= */

// ===== Configuration =====
const DEFAULT_BACKEND = import.meta.env.VITE_BACKEND_URL || "http://localhost:8001";
const API_TOKEN = import.meta.env.VITE_API_TOKEN || "";
console.log("API_TOKEN=", API_TOKEN);

const ENABLE_TESTS = false;
const HISTORY_TURNS = 10;

// 新增：严格从“当前线程”的 localStorage 读取历史
function buildChatMessagesFromHistoryStrict(threadId: string, latestUserInput: string) {
  const hist: RenderMsg[] = loadThreadMessages(threadId) || [];
  const stripCitations = (t: string) => {
    const idx = t.lastIndexOf("\nCITATIONS:");
    return idx >= 0 ? t.slice(0, idx) : t;
  };
  const msgs = hist
    .filter(m => m.role === "user" || m.role === "assistant")
    .map(m => ({ role: m.role, content: m.role === "assistant" ? stripCitations(m.text) : m.text }));

  // 只取最近 N 轮（2*N 条）
  const cut = msgs.slice(-HISTORY_TURNS * 2);

  // 末尾追加当前这条 user 输入
  cut.push({ role: "user", content: latestUserInput });
  return cut;
}

// ===== i18n (minimal) =====
const I18N = {
  en: {
    title: "ElderlyCare HK",
    subtitle: "AI chatbot for Hong Kong elderly care policy documents, with source citations.",
    chats: "Chats",
    new: "New",
    delete: "Delete",
    rename: "Rename",
    backendPlaceholder: "http://localhost:8000  or  http://localhost:8000/chat",
    language: "Language",
    streaming: "Streaming",
    font: "Font",
    highContrast: "High contrast",
    askPlaceholder: "Type your question…",
    send: "Send",
    stop: "Stop",
    helpful: "Helpful",
    notHelpful: "Not helpful",
    sources: "Sources:",
    emptyHint:
      "Ask about Hong Kong elderly care policies. Example: “Please explain Public Accountability of Non-governmental Organisations.”",
    tip: "Tips: ElderlyCare can also make mistakes. Please check the important information.",
  },
  "zh-Hant": {
    title: "ElderlyCare HK",
    subtitle: "針對香港安老政策文件的 AI 聊天助理，附來源引用。",
    chats: "對話",
    new: "新增",
    delete: "刪除",
    rename: "重新命名",
    backendPlaceholder: "http://localhost:8000  或  http://localhost:8000/chat",
    language: "語言",
    streaming: "串流回覆",
    font: "字體",
    highContrast: "高對比",
    askPlaceholder: "輸入你的問題…",
    send: "送出",
    stop: "停止",
    helpful: "有幫助",
    notHelpful: "沒有幫助",
    sources: "來源：",
    emptyHint: "可詢問香港安老政策相關問題。範例：「解釋非政府機構的公眾問責性。」",
    tip: "提示：ElderlyCare 亦可能出錯，重要資訊請再次核對。",
  },
} as const;

type Lang = keyof typeof I18N;
const tr = (lang: Lang, key: keyof typeof I18N["en"]) => I18N[lang][key];

// ===== Fallback CSS (for builds with no Tailwind) =====
function injectFallbackCSS() {
  if (typeof document === "undefined") return;
  if (document.getElementById("ec-fallback-style")) return;
  const css = `
  html, body, #root { height:100%; width:100%; margin:0; padding:0; }
  * { box-sizing: border-box; }
  .ec-root {
    min-height: 100dvh; width: 100%;
    background: #f8fafc; color: #0f172a;
    display: flex; flex-direction: column;
    font-family: system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
    font-size: calc(1rem * var(--ec-fscale, 1));
  }
  .ec-root.ec-contrast { background:#000; color:#fff; }

  /* ===== High Contrast Mode Enhancements ===== */
  .ec-contrast .ec-chat {
    background: #000;
    border-color: #444;
    color: #fff;
  }
  .ec-contrast .ec-bubble.assistant {
    background: #111;
    color: #fff;
    border-color: #555;
  }
  .ec-contrast .ec-bubble.user {
    background: #2563eb;
    color: #fff;
  }
  .ec-contrast .ec-btn {
    background: #111;
    color: #fff;
    border: 1px solid #666;
  }
  .ec-contrast .ec-btn:hover {
    background: #222;
  }
  .ec-contrast .ec-btn-primary {
    background: #2563eb;
    color: #fff;
    border: none;
  }
  .ec-contrast .ec-btn-secondary {
    background: #444;
    color: #fff;
  }

  /* ===== Feedback buttons: normal mode ===== */
  .ec-btn-feedback{
    background:#f8fafc;          /* slate-50 */
    color:#334155;               /* slate-700 */
    border:1px solid #e2e8f0;    /* slate-200 */
  }
  .ec-btn-feedback.is-active-up{
    background:#dcfce7;          /* green-50 */
    color:#166534;               /* green-800 */
    border-color:#22c55e;        /* green-500 */
  }
  .ec-btn-feedback.is-active-down{
    background:#fee2e2;          /* red-50 */
    color:#991b1b;               /* red-800 */
    border-color:#ef4444;        /* red-500 */
  }
  .ec-btn-feedback:hover{ filter:brightness(0.98); }

  .ec-contrast .ec-sidebar {
    background: #000;
    border-color: #333;
  }
  .ec-contrast .ec-thread-btn {
    background: #111;
    color: #fff;
    border-color: #444;
  }
  .ec-contrast .ec-thread-btn.active {
    background: #0a2540;
    border-color: #2563eb;
  }
  .ec-contrast .ec-mini {
    background: #111;
    color: #fff;
    border-color: #444;
  }
  .ec-contrast .ec-input,
  .ec-contrast .ec-text,
  .ec-contrast .ec-select {
    background: #000;
    color: #fff;
    border-color: #555;
  }
  .ec-contrast .ec-footer {
    color: #888;
  }
  /* ===== Feedback buttons: high-contrast ===== */
  .ec-contrast .ec-btn-feedback{
    background:#111;
    border:1px solid #666;
    color:#fff;
  }
  .ec-contrast .ec-btn-feedback.is-active-up{
    background:#064e3b;          /* 深綠 */
    border-color:#10b981;        /* emerald-500 */
    color:#fff;
  }
  .ec-contrast .ec-btn-feedback.is-active-down{
    background:#7f1d1d;          /* 深紅 */
    border-color:#f87171;        /* red-400 */
    color:#fff;
  }
  .ec-contrast .ec-btn-feedback:hover{ background:#222; }
  
  /* ===== General Styles ===== */
  .ec-root input,.ec-root select,.ec-root button,.ec-root textarea,.ec-root label,.ec-root option { font: inherit !important; font-size: 1em !important; }

  .ec-container { width: 100vw; max-width: 100vw; margin:0; padding-inline: clamp(12px,2.2vw,28px); padding-block:16px; display:flex; flex-direction:column; min-height:100dvh; }
  .ec-header { display:flex; flex-direction:column; align-items:center; text-align:center; gap:12px; margin-bottom:16px; }
  .ec-title { font-weight:800; letter-spacing:-0.02em; font-size:2.6em; line-height:1.1; }
  .ec-sub { margin-top:4px; color:#475569; font-size:1em; }
  .ec-contrast .ec-sub { color:#d1d5db; }
  .ec-controls { display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:12px; margin-top:4px; font-size:.95em; }

  .ec-input,.ec-select,.ec-text { border:1px solid #cbd5e1; border-radius:12px; padding:8px 12px; background:#fff; color:#0f172a; font-size:1em; }
  .ec-contrast .ec-input,.ec-contrast .ec-select,.ec-contrast .ec-text { background:#000; border-color:#4b5563; color:#fff; }
  .ec-range { vertical-align: middle; }


  /* ====== two-column layout (sidebar + main) ====== */
  .ec-layout { display:flex; gap:16px; min-height:0; flex:1 1 auto; }
  .ec-sidebar {
    flex: 0 0 260px;
    max-height: calc(100dvh - 210px);
    border:1px solid #e2e8f0; border-radius:14px; background:#fff; padding:10px;
    display:flex; flex-direction:column; gap:8px; overflow:auto;
    box-shadow:0 1px 2px rgba(0,0,0,.04);
  }
  .ec-contrast .ec-sidebar { background:#000; border-color:#334155; }
  .ec-side-head { display:flex; flex-direction:column; align-items:center; gap:8px; margin-bottom:6px; }
  .ec-side-title { font-weight:800; font-size:1.2rem; }
  .ec-side-list { display:flex; flex-direction:column; gap:6px; }

  /* Sidebar chat title behaviour */
  .ec-thread-btn {
    white-space: normal !important;   /* 允许换行 */
    word-break: break-word;           /* 单词太长可断行 */
    overflow-wrap: anywhere;
  }

  .ec-thread-btn div:first-child {
    display: -webkit-box;
    -webkit-line-clamp: 2;            /* 最多显示两行 */
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;          /* 溢出自动加 … */
    line-height: 1.3;
    font-weight: 600;
  }

  .ec-thread-btn:hover div:first-child {
    white-space: normal;              /* Hover时完全展开*/
  }

  .ec-thread-btn:hover { background:#f8fafc; }
  .ec-thread-btn.active { border-color:#2563eb; outline:2px solid #93c5fd; background:#eff6ff; }
  .ec-contrast .ec-thread-btn { background:#000; color:#fff; border-color:#334155; }
  .ec-contrast .ec-thread-btn:hover { background:#0b0b0b; }
  .ec-contrast .ec-thread-btn.active { outline-color:#2563eb; background:#0a2540; }

  .ec-side-actions { display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }
  .ec-mini { padding:6px 10px; border-radius:10px; border:1px solid #e5e7eb; background:#fff; cursor:pointer; }
  .ec-mini:disabled{opacity:.6;cursor:not-allowed}
  .ec-contrast .ec-mini { background:#000; color:#fff; border-color:#334155; }

  /* Right main column (chat) */
  .ec-main { flex:1 1 auto; display:flex; flex-direction:column; min-height:0; }
  .ec-chat{
    width:100%; flex:1 1 auto; min-height:0; overflow:auto;
    border:1px solid #e2e8f0; border-radius:18px; padding:16px; background:#fff;
    height:calc(100dvh - 260px);
    box-shadow:0 1px 2px rgba(0,0,0,.04); font-size:1em;
  }
  .ec-contrast .ec-chat{background:#000;border-color:#334155}
  .ec-empty{padding-top:64px;text-align:center;color:#64748b;font-size:1em;}
  .ec-contrast .ec-empty{color:#9ca3af}
  .ec-row{display:flex;gap:12px;margin-top:12px;align-items:flex-end;flex-wrap:wrap}
  .ec-grow{flex:1 1 auto;min-width:0}
  .ec-actions{display:flex;gap:8px;align-items:center;flex:0 0 auto}

  @media (max-width: 860px){
    .ec-layout{flex-direction:column}
    .ec-sidebar{flex:0 0 auto; max-height:none;}
  }

  .ec-btn{display:inline-flex;align-items:center;gap:8px;padding:12px 16px;border-radius:16px;font-weight:600;border:none;cursor:pointer;font-size:1em;}
  .ec-btn:disabled{opacity:.6;cursor:not-allowed}
  .ec-btn-primary{background:#2563eb;color:#fff}
  .ec-btn-secondary{background:#111827;color:#fff}
  .ec-contrast .ec-btn-secondary{background:#fff;color:#000}

  .ec-bubble{max-width:85%;border-radius:16px;padding:12px 16px;box-shadow:0 1px 2px rgba(0,0,0,.06);font-size:1em;}
  .ec-bubble.user{margin-left:auto;background:#2563eb;color:#fff}
  .ec-bubble.assistant{margin-right:auto;background:rgba(255,255,255,.9);color:#111827;border:1px solid #e5e7eb}

  .ec-md { line-height: 1.6; white-space: normal; word-break: break-word; }
  .ec-md p { margin: 0.4em 0; }
  .ec-md h1, .ec-md h2, .ec-md h3, .ec-md h4, .ec-md h5, .ec-md h6 {
    margin: 0.6em 0 0.3em; font-weight: 700; line-height: 1.25;
  }
  .ec-md ul, .ec-md ol { padding-left: 1.25em; margin: 0.4em 0; }
  .ec-md code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: .95em; background: rgba(0,0,0,.05); padding: .1em .35em; border-radius: .3em; }
  .ec-md pre { background: rgba(0,0,0,.05); padding: .8em; border-radius: .6em; overflow: auto; }
  .ec-contrast .ec-md code, .ec-contrast .ec-md pre { background: rgba(255,255,255,.08); }
  .ec-md a { text-decoration: underline; }

  /* ===== Normal Mode: Sources Section ===== */
  .ec-sources { 
    font-size: inherit; 
    line-height: 1.55;
  }
  .ec-sources li { line-height: 1.55; }

  .ec-src-title{
    font-weight:700;
    margin:8px 0 4px;
    /* 关键：不要再写固定 font-size，保持继承 */
    color:#1e3a8a; 
    text-transform:uppercase;
    letter-spacing:.02em;
  }

  .ec-src-list{
    margin:0;
    padding-left:18px;
    /* 关键：不要固定 0.95em，保持继承 */
    color:#1e293b;
  }

  .ec-src-item{
    word-break:break-word;
    color:#334155;
    transition:all .2s ease;
  }

  .ec-src-item span.font-medium{
    color:#2563eb;
    font-weight:600;
  }

  .ec-src-item:hover{
    background:rgba(37,99,235,.08);
    color:#0f172a;
    border-radius:6px;
  }

  /* ===== High Contrast Mode: Sources Section ===== */
  .ec-contrast .ec-sources { 
    font-size: inherit; /* 同样继承，确保随滑杆缩放 */
    line-height: 1.6;
  }

  .ec-sources li { line-height: 1.55; }

  .ec-contrast .ec-src-title{
    font-weight:700;
    margin:8px 0 4px;
    color:#ffd166;
  }

  .ec-contrast .ec-src-list{
    margin:0;
    padding-left:18px;
  }

  .ec-contrast .ec-src-item{
    word-break:break-word;
    color:#d4d4d4;
  }

  .ec-contrast .ec-src-item span.font-medium{
    color:#80bfff;
    font-weight:600;
  }

  .ec-contrast .ec-src-item:hover{
    color:#fff;
    background:rgba(255,255,255,.08);
    border-radius:6px;
  }


  .ec-footer { margin-top: 16px; font-size: .9em; color:#64748b; text-align:center; }
  .ec-contrast .ec-footer{color:#9ca3af}

  .ec-tests{margin-top:16px;border:1px solid #e5e7eb;background:rgba(255,255,255,.7);border-radius:12px;padding:12px;font-size:0.9em;}
  .ec-ico{display:inline-block}
  `;
  const style = document.createElement("style");
  style.id = "ec-fallback-style";
  style.textContent = css;
  document.head.appendChild(style);
}

// ===== Types =====
interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface ChatRequest {
  messages: ChatMessage[];
  stream?: boolean;
  language?: "en" | "zh-Hant";
  threadId?: string; 
}

export interface Citation {
  file: string;
  page?: number | null;
  snippet?: string | null;
}

interface ChatAnswer {
  answer: string;
  citations: Citation[];
}

// Small helper to join class names
const cx = (...xs: Array<string | false | null | undefined>) => xs.filter(Boolean).join(" ");

// ===== Pure helpers (testable) =====
export function normalizeNewlines(s: string | undefined | null): string {
  return s?.replace(/\\n/g, "\n") ?? "";
}
export function tryExtractCitations(buffer: string): { cites: Citation[]; cutIndex: number } | undefined {
  const marker = "\nCITATIONS:";
  const idx = buffer.lastIndexOf(marker);
  if (idx === -1) return undefined;
  const after = buffer.slice(idx + marker.length);
  const lastClose = Math.max(after.lastIndexOf("]"), after.lastIndexOf("}"));
  if (lastClose === -1) return undefined;
  const jsonStr = after.slice(0, lastClose + 1).trim();
  try {
    const parsed = JSON.parse(jsonStr);
    if (Array.isArray(parsed)) {
      return { cites: parsed as Citation[], cutIndex: idx };
    }
  } catch {}
  return undefined;
}

// ===== Icons =====
const IconSend = () => (
  <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/>
  </svg>
);
const IconStop = () => (
  <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="6" y="6" width="12" height="12" />
  </svg>
);

// ====== threads (localStorage) ======
type RenderMsg = { id: string; role: "user" | "assistant"; text: string; citations?: Citation[] };
type ThreadMeta = { id: string; title: string; createdAt: number; updatedAt: number };

const LS_THREADS = "ec-threads";
const LS_THREAD_PREFIX = "ec-thread:";

function loadThreads(): ThreadMeta[] {
  try { return JSON.parse(localStorage.getItem(LS_THREADS) || "[]"); } catch { return []; }
}
function saveThreads(ts: ThreadMeta[]) {
  localStorage.setItem(LS_THREADS, JSON.stringify(ts));
}
function loadThreadMessages(id: string): RenderMsg[] {
  try { return JSON.parse(localStorage.getItem(LS_THREAD_PREFIX + id) || "[]"); } catch { return []; }
}
function saveThreadMessages(id: string, msgs: RenderMsg[]) {
  localStorage.setItem(LS_THREAD_PREFIX + id, JSON.stringify(msgs));
}
function titleFromFirstUserMessage(msgs: RenderMsg[]): string {
  const firstUser = msgs.find(m => m.role === "user");
  const base = firstUser?.text?.trim() || "New chat";
  return base || "New chat";
}

// ===== Message bubble =====
function Bubble({ role, children }: { role: "user" | "assistant"; children: React.ReactNode }) {
  const isUser = role === "user";
  return (
    <div className={cx("flex mb-3", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cx(
          "max-w-[85%] rounded-2xl px-4 py-3 shadow-sm",
          isUser ? "bg-blue-600 text-white" : "bg-white/80 text-gray-900 border border-gray-200",
          "ec-bubble",
          isUser ? "user" : "assistant"
        )}
        role="group"
        aria-label={isUser ? "User message" : "Assistant message"}
      >
        {children}
      </div>
    </div>
  );
}

// ===== Citation list =====
function Citations({ items, lang }: { items: Citation[]; lang: Lang }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="mt-3 ec-sources">
      <div className="font-semibold mb-1 ec-src-title">
        <span className="font-bold">{tr(lang, "sources")}</span>
      </div>
      <ul className="list-disc pl-5 space-y-1 ec-src-list">
        {items.map((c, i) => (
          <li key={i} className="break-words ec-src-item">
            <span className="font-medium">[Source {i + 1}]</span> {c.file}
            {c.page != null ? (
              <span>{lang === "zh-Hant" ? `，頁 ${c.page}` : `, page ${c.page}`}</span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

// ===== Feedback buttons =====
function Feedback({
  messageId, lang, backendBase, threadId, userQuery, answer, citations
}: {
  messageId: string; lang: Lang; backendBase: string; threadId: string;
  userQuery: string; answer: string; citations: Citation[];
}) {
  const [state, setState] = useState<"up"|"down"|null>(() => (localStorage.getItem(`fb:${messageId}`) as any) || null);

  async function send(label: "up"|"down") {
    localStorage.setItem(`fb:${messageId}`, label);
    setState(label);
    // fire-and-forget
    try {
      await fetch(`${backendBase}/feedback`, {
        method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${API_TOKEN}`,
          },
        body: JSON.stringify({
          threadId, messageId, label,
          userQuery, answer, language: lang,
          citations,
          meta: { ua: navigator.userAgent }
        })
      });
    } catch { /* 静默失败，不打断体验 */ }
  }

  const base = "inline-flex items-center gap-1 px-2 py-1 rounded-md border text-sm";
  return (
    <div className="flex gap-2 mt-2">
      <button
        aria-label={tr(lang, "helpful")}
        className={cx(base, "ec-btn","ec-btn-feedback", state==="up" && "is-active-up")}
        onClick={() => send("up")}
      >
        <span className="ec-ico" aria-hidden>👍</span> {tr(lang, "helpful")}
      </button>
      <button
        aria-label={tr(lang, "notHelpful")}
        className={cx(base,"ec-btn","ec-btn-feedback", state==="down" && "is-active-down")}
        onClick={() => send("down")}
      >
        <span className="ec-ico" aria-hidden>👎</span> {tr(lang, "notHelpful")}
      </button>
    </div>
  );
}

// ===== Main App =====
export default function App() {
  useEffect(() => { injectFallbackCSS(); }, []);

  // ===== UI State =====
  const [backendUrl, setBackendUrl] = useState<string>(() => localStorage.getItem("backendUrl") || DEFAULT_BACKEND);
  const [language, setLanguage] = useState<"en" | "zh-Hant">(
    () => (localStorage.getItem("language") as any) || "en"
  );
  const [stream, setStream] = useState(true);
  const [fontScale, setFontScale] = useState(1.0);
  const [highContrast, setHighContrast] = useState(false);

  // messages kept for UI rendering only (current thread)
  const [history, setHistory] = useState<RenderMsg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const [showAdmin, setShowAdmin] = useState(false);

  // ===== threads state =====
  const [threads, setThreads] = useState<ThreadMeta[]>(() => {
    const ts = loadThreads();
    return ts.sort((a, b) => b.updatedAt - a.updatedAt);
  });
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(() => threads[0]?.id ?? null);

  // ===== Rename state & helpers =====
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState<string>("");

  function sanitizeInlineCitations(text: string, maxK: number): string {
    if (!text) return text;
    return text.replace(/\[Source\s+(\d+)\]/g, (_m, g1) => {
      const n = parseInt(g1, 10);
      return (n >= 1 && n <= maxK) ? `[Source ${n}]` : ""; // 越界则移除
    });
  }

  function beginRename(thread: ThreadMeta) {
    setRenamingId(thread.id);
    setRenameDraft(thread.title);
  }
  function applyRename(id: string, nextTitle?: string) {
    const title = (nextTitle ?? renameDraft).trim();
    setThreads(prev => {
      const now = Date.now();
      const out = prev
        .map(t => (t.id === id ? { ...t, title: title || t.title, updatedAt: now } : t))
        .sort((a, b) => b.updatedAt - a.updatedAt);
      saveThreads(out);
      return out;
    });
    setRenamingId(null);
    setRenameDraft("");
  }

  // keep persisted
  useEffect(() => { localStorage.setItem("backendUrl", backendUrl); }, [backendUrl]);
  useEffect(() => { localStorage.setItem("language", language); }, [language]);
  useEffect(() => { listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" }); }, [history, loading]);

  // create a default thread on first run OR load current thread messages
  useEffect(() => {
    if (!currentThreadId) {
      const id = crypto.randomUUID();
      const now = Date.now();
      const meta: ThreadMeta = { id, title: "New chat", createdAt: now, updatedAt: now };
      const nts = [meta, ...threads];
      setThreads(nts);
      saveThreads(nts);
      setCurrentThreadId(id);
      saveThreadMessages(id, []);
      setHistory([]);
      // immediately rename new thread
      beginRename(meta);
    } else {
      const msgs = loadThreadMessages(currentThreadId);
      setHistory(msgs);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentThreadId]);

  // Theme classes
  const rootClass = useMemo(
    () => ["ec-root", highContrast ? "ec-contrast" : null, highContrast ? "bg-black text-white" : "bg-slate-50 text-slate-900"].filter(Boolean).join(" "),
    [highContrast]
  );

  // Normalize endpoint: accept base URL or full /chat endpoint
  function makeChatEndpoint(raw: string): string {
    const trimmed = (raw || "").trim().replace(/\/$/, "");
    if (!trimmed) return `${DEFAULT_BACKEND}/chat`;
    return /\/chat$/i.test(trimmed) ? trimmed : `${trimmed}/chat`;
  }

  // thread helpers
  function updateCurrentThreadTitleIfNeeded(msgs: RenderMsg[]) {
    setThreads(prev => {
      const now = Date.now();
      const out = prev
        .map(t =>
          t.id === currentThreadId
            ? { ...t, title: t.title === "New chat" ? titleFromFirstUserMessage(msgs) : t.title, updatedAt: now }
            : t
        )
        .sort((a, b) => b.updatedAt - a.updatedAt);
      saveThreads(out);
      return out;
    });
  }

  function createThread() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);

    const id = crypto.randomUUID();
    const now = Date.now();
    const meta: ThreadMeta = { id, title: "New chat", createdAt: now, updatedAt: now };
    const nts = [meta, ...threads];
    setThreads(nts);
    saveThreads(nts);
    saveThreadMessages(id, []);
    setCurrentThreadId(id);
    setHistory([]);
    setInput("");
    beginRename(meta);
  }

  function deleteThread(id: string) {
    const nts = threads.filter(t => t.id !== id);
    setThreads(nts);
    saveThreads(nts);
    localStorage.removeItem(LS_THREAD_PREFIX + id);
    if (currentThreadId === id) {
      setCurrentThreadId(nts[0]?.id ?? null);
    }
  }

  function selectThread(id: string) {
    if (id === currentThreadId) return;
    // 关键：切线程前先停流，清 loading
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);

    setCurrentThreadId(id);
  }

  // ===== Sending message =====
  async function sendMessage() {
    if (!input.trim() || loading || !currentThreadId) return;
    setError(null);

    const userMsg: RenderMsg = { id: crypto.randomUUID(), role: "user", text: input };
    const assistantId = crypto.randomUUID();

    // local echo
    const newHistory = [...history, userMsg, { id: assistantId, role: "assistant", text: "" }];
    setHistory(newHistory);
    saveThreadMessages(currentThreadId, newHistory);
    updateCurrentThreadTitleIfNeeded(newHistory);

    setLoading(true);
    try {
      const payload: ChatRequest = {
        messages: buildChatMessagesFromHistoryStrict(currentThreadId!, input),
        stream,
        language,
        threadId: currentThreadId!
      };      
      const endpoint = makeChatEndpoint(backendUrl);

      if (stream) {
        abortRef.current = new AbortController();
        const resp = await fetch(endpoint, {
          method: "POST",
          mode: "cors",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${API_TOKEN}`,
            },
          body: JSON.stringify(payload),
          signal: abortRef.current.signal,
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const reader = resp.body!.getReader();
        const decoder = new TextDecoder("utf-8");

        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          setHistory(h => {
            const hh = h.map(m => m.id === assistantId ? { ...m, text: buffer } : m);
            saveThreadMessages(currentThreadId, hh);
            return hh;
          });

          const ext = tryExtractCitations(buffer);
          if (ext) {
            setHistory(h => {
              const clean = sanitizeInlineCitations(buffer.slice(0, ext.cutIndex), ext.cites.length);
              const hh = h.map(m => m.id === assistantId ? { ...m, citations: ext.cites, text: clean } : m);
              saveThreadMessages(currentThreadId, hh);
              return hh;
            });
          }
        }
      } else {
        const resp = await fetch(endpoint, {
          method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${API_TOKEN}`,
            },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data: ChatAnswer = await resp.json();
        setHistory(h => {
          const hh = h.map(m => m.id === assistantId ? { ...m, text: normalizeNewlines(data.answer), citations: data.citations } : m);
          saveThreadMessages(currentThreadId, hh);
          return hh;
        });
      }

      // bump updatedAt
      updateCurrentThreadTitleIfNeeded(loadThreadMessages(currentThreadId));
    } catch (e: any) {
      setError(e?.message || String(e));
      setHistory(h => {
        const hh = h.map(m => (m.id === assistantId && !m.text ? { ...m, text: "(Request failed)" } : m));
        saveThreadMessages(currentThreadId, hh);
        return hh;
      });
    } finally {
      setLoading(false);
      setInput("");
      abortRef.current = null;
    }
  }

  function stopStreaming() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  // ===== Render =====
  return (
    <div className={rootClass} style={{ ["--ec-fscale" as any]: fontScale }}>
      <div className="ec-container">
        {/* Header */}
        <header className={cx("ec-header", "flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-4")}>
          <div>
            <h1 className={cx("ec-title", "text-2xl md:text-3xl font-bold tracking-tight")}>{tr(language, "title")}</h1>
            <p className={cx("ec-sub", "mt-1", highContrast ? "text-gray-300" : "text-slate-600")}>
              {tr(language, "subtitle")}
            </p>
          </div>
          <div className={cx("ec-controls", "flex flex-wrap items-center gap-3")}>
            <label className="flex items-center gap-2">
              <span className="sr-only">Backend URL or /chat endpoint</span>
              <input
                value={backendUrl}
                onChange={e => setBackendUrl(e.target.value)}
                className={cx("ec-input", "px-3 py-2 rounded-xl border w-64", highContrast ? "bg-black border-gray-600 text-white" : "bg-white border-gray-300")}
                aria-label="Backend URL or /chat endpoint"
                placeholder={tr(language, "backendPlaceholder")}
              />
            </label>
            <label className="flex items-center gap-2">
              <span className="text-sm">{tr(language, "language")}</span>
              <select
                value={language}
                onChange={e => setLanguage(e.target.value as any)}
                className={cx("ec-select", "px-2 py-2 rounded-xl border", highContrast ? "bg-black border-gray-600 text-white" : "bg-white border-gray-300")}
                aria-label="Language"
              >
                <option value="en">English</option>
                <option value="zh-Hant">繁體中文</option>
              </select>
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={stream} onChange={e => setStream(e.target.checked)} />
              <span className="text-sm">{tr(language, "streaming")}</span>
            </label>
            <label className="flex items-center gap-2">
              <span className="text-sm">{tr(language, "font")}</span>
              <input type="range" min={0.9} max={1.6} step={0.05} value={fontScale} onChange={e => setFontScale(parseFloat(e.target.value))} aria-label="Font size" className="ec-range" />
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={highContrast} onChange={e => setHighContrast(e.target.checked)} />
              <span className="text-sm">{tr(language, "highContrast")}</span>             
            </label>
            
            <button
              className="ec-mini"
              onClick={() => setShowAdmin(v => !v)}
              aria-label="Admin panel"
            >
              {showAdmin ? (language === "zh-Hant" ? "返回對話" : "Back to Chat") : "Admin"}
            </button>
          </div>
        </header>

        {/* Two-column layout */}
          {showAdmin ? (
            <div className="ec-layout">
              <main className="ec-main">
                <AdminDocs />
              </main>
            </div>
          ) : (
          <div className="ec-layout">
            {/* Sidebar: thread list */}
            <aside className="ec-sidebar" aria-label="Conversation list">
              <div className="ec-side-head">
                <div className="ec-side-title">{tr(language, "chats")}</div>
                <div className="ec-side-actions">
                  <button className="ec-mini" 
                    onClick={createThread} 
                    aria-label="New chat">
                    ＋ {tr(language, "new")}
                  </button>
                  <button
                    className="ec-mini"
                    onClick={() => currentThreadId && deleteThread(currentThreadId)}
                    disabled={!currentThreadId}
                    aria-label="Delete current chat"
                  >
                    🗑 {tr(language, "delete")}
                  </button>
                  <button
                    className="ec-mini"
                    onClick={() => {
                      const t = threads.find(x => x.id === currentThreadId);
                      if (t) beginRename(t);
                    }}
                    disabled={!currentThreadId}
                    aria-label="Rename current chat"
                  >
                    ✏️ {tr(language, "rename")}
                  </button>
                </div>
              </div>

              <div className="ec-side-list" role="list">
                {threads.map(t => (
                  <div key={t.id}>
                    {renamingId === t.id ? (
                      <form
                        className="ec-thread-btn active"
                        onSubmit={(e) => { e.preventDefault(); applyRename(t.id); }}
                        style={{ display: "flex", flexDirection: "column", gap: 6 }}
                      >
                        <input
                          autoFocus
                          value={renameDraft}
                          onChange={(e) => setRenameDraft(e.target.value)}
                          onBlur={() => applyRename(t.id)}
                          onKeyDown={(e) => {
                            if (e.key === "Escape") { setRenamingId(null); setRenameDraft(""); }
                            if (e.key === "Enter") { e.preventDefault(); applyRename(t.id); }
                          }}
                          className="ec-input"
                          aria-label="Edit chat title"
                          placeholder={tr(language, "backendPlaceholder")}
                          style={{ width: "100%", whiteSpace: "normal", wordBreak: "break-word" }}
                        />
                        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                          <button type="button" className="ec-mini" onClick={() => { setRenamingId(null); setRenameDraft(""); }}>
                            {language === "zh-Hant" ? "取消" : "Cancel"}
                          </button>
                          <button type="submit" className="ec-mini">
                            {language === "zh-Hant" ? "儲存" : "Save"}
                          </button>
                        </div>
                      </form>
                    ) : (
                      <button
                        role="listitem"
                        className={cx("ec-thread-btn", t.id === currentThreadId ? "active" : "")}
                        onClick={() => selectThread(t.id)}
                        onDoubleClick={() => beginRename(t)}
                        title={new Date(t.updatedAt).toLocaleString() + "\n" + t.title}
                        style={{ whiteSpace: "normal", wordBreak: "break-word", overflow: "visible" }}
                      >
                        <div style={{fontWeight:600, marginBottom:2, whiteSpace:"normal", wordBreak:"break-word"}}>{t.title}</div>
                        <div style={{opacity:.65, fontSize:'.85em'}}>
                          {new Date(t.updatedAt).toLocaleString()}
                        </div>
                      </button>
                    )}
                  </div>
                ))}
                {threads.length === 0 && (
                  <div style={{opacity:.6, fontSize:'.95em'}}>
                    {language === "zh-Hant" ? "目前沒有對話。" : "No chats yet."}
                  </div>
                )}
              </div>
            </aside>
            {/* Main content (chat + composer) */}
            <main className="ec-main">
              {/* Chat window */}
              <section
                ref={listRef}
                className={cx("ec-chat", "rounded-2xl p-4 border", highContrast ? "bg-black border-gray-700" : "bg-white border-gray-200")}
                aria-live="polite"
              >
                {history.length === 0 && (
                  <div className={cx("ec-empty", "text-center pt-16", highContrast ? "text-gray-400" : "text-slate-500")}>
                    {tr(language, "emptyHint")}
                  </div>
                )}

                {history.map((m, idx) => {
                  const prevUser = [...history.slice(0, idx)].reverse().find(x => x.role === "user");
                  return (
                    <div key={m.id}>
                      <Bubble role={m.role}>
                        {m.role === "assistant" ? (
                          <div className="ec-md">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              // 不渲染原始 HTML，避免 XSS；我们只支持 Markdown 语法
                              skipHtml
                              // 允许换行行为更自然（可选）
                              // linkTarget="_blank" 也可以在这里加
                            >
                              {m.text || ""}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          <div className="whitespace-pre-wrap break-words">{m.text}</div>
                        )}

                        {m.role === "assistant" && m.citations && <Citations items={m.citations} lang={language} />}
                      </Bubble>
                      {m.role === "assistant" && (
                        <Feedback
                          messageId={m.id}
                          lang={language}
                          backendBase={makeChatEndpoint(backendUrl).replace(/\/chat$/i, "")}
                          threadId={currentThreadId!}
                          userQuery={prevUser?.text || ""}
                          answer={m.text}
                          citations={m.citations || []}
                        />
                      )}
                    </div>
                  );
                })}

                {loading && <div className="mt-2 text-sm opacity-70">Generating…</div>}
                {error && <div className="mt-2 text-sm text-red-600">Error: {error}</div>}
              </section>

              {/* Composer */}
              <form className={cx("ec-row", "mt-4 flex flex-wrap items-end gap-3")} onSubmit={(e) => { e.preventDefault(); sendMessage(); }}>
                <label className="flex-1 min-w-0 ec-grow">
                  <span className="sr-only">Your question</span>
                  <textarea
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    placeholder={tr(language, "askPlaceholder")}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    className={cx("ec-text","w-full min-h-[64px] max-h-[160px] p-3 rounded-2xl border resize-y", highContrast ? "bg-black border-gray-700 text-white placeholder-gray-400" : "bg-white border-gray-300")}
                  />
                </label>
                <div className="flex gap-2 ec-actions flex-shrink-0 w-full sm:w-auto justify-end">
                  <button type="submit" disabled={loading || !input.trim()} className={cx("ec-btn ec-btn-primary","inline-flex items-center gap-2 px-4 py-3 rounded-2xl font-medium", loading || !input.trim() ? "opacity-60 cursor-not-allowed" : "")} aria-label="Send message">
                    <IconSend /> {tr(language, "send")}
                  </button>
                  {loading && (
                    <button type="button" onClick={stopStreaming} className={cx("ec-btn ec-btn-secondary", "inline-flex items-center gap-2 px-4 py-3 rounded-2xl font-medium")} aria-label="Stop streaming">
                      <IconStop /> {tr(language, "stop")}
                    </button>
                  )}
                </div>
              </form>
            </main>
          </div>
      )}
        {/* Footer */}
        <footer className={cx("ec-footer", "mt-6 text-xs", highContrast ? "text-gray-400" : "text-slate-500")}> 
          {tr(language, "tip")}        
        </footer>

        {ENABLE_TESTS && <DevTestPanel />}
      </div>
    </div>
  );
}

// ===== In-browser tests (no external test runner) =====
function DevTestPanel() {
  type T = { name: string; pass: boolean; detail?: string };
  const [tests, setTests] = useState<T[]>([]);

  useEffect(() => {
    const results: T[] = [];
    const push = (name: string, fn: () => void) => { try { fn(); results.push({ name, pass: true }); } catch (e: any) { results.push({ name, pass: false, detail: e?.message || String(e) }); } };

    // Test 1
    push("normalizeNewlines converts \\n to newline", () => {
      const out = normalizeNewlines("line1\\nline2");
      if (out !== "line1\nline2") throw new Error(`got ${JSON.stringify(out)}`);
    });

    // Test 2
    push("tryExtractCitations extracts trailing CITATIONS", () => {
      const mock = "Answer text here.\nCITATIONS:[{\"file\":\"a.md\",\"page\":2}]";
      const res = tryExtractCitations(mock);
      if (!res) throw new Error("no extraction");
      if (!Array.isArray(res.cites) || res.cites.length !== 1) throw new Error("bad cites");
      if (res.cutIndex <= 0 || res.cutIndex >= mock.length) throw new Error("bad cutIndex");
    });

    // Test 3
    push("tryExtractCitations returns undefined without marker", () => {
      const res = tryExtractCitations("No citations here.");
      if (res !== undefined) throw new Error("expected undefined");
    });

    // Test 4 (inline)
    const makeEndpointInline: any = (raw: string) => (raw || "").trim().replace(/\/$/, "").match(/\/chat$/) ? raw : `${(raw || "").trim().replace(/\/$/, "")}/chat`;
    push("makeChatEndpoint normalizes base URL to /chat", () => {
      const a = makeEndpointInline("http://localhost:8000");
      const b = makeEndpointInline("http://localhost:8000/chat");
      if (!/\/chat$/.test(a)) throw new Error("base not normalized");
      if (!/\/chat$/.test(b)) throw new Error("already-/chat not preserved");
    });

    setTests(results);
  }, []);

  return (
    <div className="ec-tests mt-6 rounded-xl border p-3 bg-white/70">
      <div className="font-semibold mb-2">Dev Tests</div>
      <ul className="space-y-1 text-sm">
        {tests.map((t, i) => (
          <li key={i} className={t.pass ? "text-green-700" : "text-red-700"}>
            {t.pass ? "✅" : "❌"} {t.name}
            {t.detail ? <span className="ml-2 opacity-70">— {t.detail}</span> : null}
          </li>
        ))}
      </ul>
      <div className="text-xs mt-2 opacity-70">These lightweight tests run in the browser to verify core helpers. They don't mock network.</div>
    </div>
  );
}