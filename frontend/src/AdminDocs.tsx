import React, { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_BACKEND_URL || "http://localhost:8001";
const API_TOKEN = import.meta.env.VITE_API_TOKEN || "";
const MD_ACCEPT = ".md,.markdown,text/markdown";

type DocItem = { filename: string; size: number; modified: number };
type Status = { status: string; note?: string; last_built?: number };

export default function AdminDocs() {
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [status, setStatus] = useState<Status>({ status: "loading" });
  const [busy, setBusy] = useState(false);
    async function refresh() {
      try {
        const [listRes, statRes] = await Promise.all([
          fetch(`${API_BASE}/docs/list`, {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
          }),
          fetch(`${API_BASE}/docs/status`, {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
          }),
        ]);
        if (!listRes.ok || !statRes.ok) {
          const msg = `HTTP ${listRes.status}/${statRes.status}`;
          setStatus({ status: "error", note: msg });
          return;
        }
        const list = await listRes.json();
        const stat = await statRes.json();
        setDocs(list.docs || []);
        setStatus(stat);
      } catch (e: any) {
        setStatus({ status: "error", note: e?.message || "network error" });
      }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;

    // 前端兜底校验：仅允许 .md / .markdown
    const nameOk = /\.(md|markdown)$/i.test(f.name);
    const mimeOk = (f.type || "").toLowerCase().includes("markdown");
    if (!nameOk && !mimeOk) {
      alert("僅支援上傳 Markdown 檔（.md / .markdown）");
      e.currentTarget.value = ""; // 清空選擇
      return;
    }

    const fd = new FormData();
    fd.append("file", f);
    setBusy(true);
    await fetch(`${API_BASE}/docs/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${API_TOKEN}` },
      body: fd,
    }).then((r) => r.json());
    setBusy(false);
    await refresh();
  }

  async function onDelete(name: string) {
    if (!confirm(`Delete ${name}?`)) return;
    setBusy(true);
    await fetch(`${API_BASE}/docs/${encodeURIComponent(name)}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${API_TOKEN}` },
    }).then((r) => r.json());
    setBusy(false);
    await refresh();
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h1 className="text-xl font-semibold mb-3">Document Admin</h1>

      <div className="mb-4 flex items-center gap-3">
        <input type="file" accept={MD_ACCEPT} onChange={onUpload} disabled={busy} />
        <span className="text-sm text-gray-600">
          Status:{" "}
          <b className={status.status === "error" ? "text-red-600" : ""}>
            {status.status}
          </b>{" "}
          {status.note ? `(${status.note})` : ""}
        </span>
        <button
          className="px-3 py-1 border rounded"
          onClick={async () => {
            setBusy(true);
            await fetch(`${API_BASE}/docs/cancel`, {
              method: "POST",
              headers: { Authorization: `Bearer ${API_TOKEN}` }
            }).then(r => r.json());
            setBusy(false);
            await refresh();
          }}
          disabled={busy || status.status !== "indexing"}
        >
          Cancel
        </button>
        <button
          className="px-3 py-1 border rounded"
          onClick={refresh}
          disabled={busy}
        >
          Refresh
        </button>
      </div>

      <table className="w-full text-sm border">
        <thead>
          <tr className="bg-gray-50">
            <th className="p-2 text-left">Filename</th>
            <th className="p-2 text-left">Size</th>
            <th className="p-2 text-left">Modified</th>
            <th className="p-2 text-left">Action</th>
          </tr>
        </thead>
        <tbody>
          {docs.map((d) => (
            <tr key={d.filename} className="border-t">
              <td className="p-2">{d.filename}</td>
              <td className="p-2">{(d.size / 1024).toFixed(1)} KB</td>
              <td className="p-2">
                {new Date(d.modified * 1000).toLocaleString()}
              </td>
              <td className="p-2">
                <button
                  className="px-2 py-1 border rounded"
                  onClick={() => onDelete(d.filename)}
                  disabled={busy}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
          {docs.length === 0 && (
            <tr>
              <td className="p-2 text-gray-500" colSpan={4}>
                No documents.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
