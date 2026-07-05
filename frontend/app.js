// MLSim Assistant frontend — vanilla, no build step.
// Talks to the FastAPI backend; one chat turn streams Server-Sent Events
// (progress… -> orchestrator/implementer handoffs -> done) which we render live.

const $ = (sel) => document.querySelector(sel);

const state = {
  conversations: [],
  currentId: null,
  sandbox: localStorage.getItem("mlsim_sandbox") || "workspace-write",
  streaming: false,
  abortController: null,
};

const els = {
  list: $("#convList"),
  empty: $("#convEmpty"),
  messages: $("#messages"),
  title: $("#chatTitle"),
  input: $("#input"),
  send: $("#send"),
  stop: $("#stop"),
  composer: $("#composer"),
  sandbox: $("#sandbox"),
  sandboxIco: $("#sandboxIco"),
  newChat: $("#newChat"),
};

// ---------- API ----------
const api = {
  async listConversations() {
    const r = await fetch("/api/conversations");
    return r.json();
  },
  async createConversation(sandbox) {
    const r = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sandbox }),
    });
    return r.json();
  },
  async getConversation(id) {
    const r = await fetch(`/api/conversations/${id}`);
    if (!r.ok) return null;
    return r.json();
  },
  async deleteConversation(id) {
    await fetch(`/api/conversations/${id}`, { method: "DELETE" });
  },
};

// ---------- Markdown (compact) ----------
function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function normalizeBackendText(s) {
  return (s || "").replace(/\\r\\n/g, "\n").replace(/\\n/g, "\n").replace(/\\t/g, "\t");
}
function inlineMd(t) {
  let s = escHtml(t);
  s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^\w])\*([^*\n]+)\*(?=[^\w]|$)/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}
// Rewrite local image srcs (e.g. /workspace/logs/.../plot.png) to the backend file route.
function rewriteImages(el) {
  el.querySelectorAll("img").forEach((img) => {
    const src = img.getAttribute("src") || "";
    if (!/^(https?:|data:|\/api\/file)/i.test(src)) {
      const cid = state.currentId ? "&cid=" + encodeURIComponent(state.currentId) : "";
      img.setAttribute("src", "/api/file?path=" + encodeURIComponent(src) + cid);
    }
    img.setAttribute("loading", "lazy");
  });
}

// Full markdown (tables, images, etc.) via marked + DOMPurify; fall back to the mini
// renderer if the vendor libs are unavailable (e.g. offline first load).
function renderMarkdown(src) {
  src = normalizeBackendText(src || "");
  if (window.marked && window.DOMPurify) {
    try {
      window.marked.setOptions({ gfm: true, breaks: true });
      const dirty = window.marked.parse(src || "");
      return window.DOMPurify.sanitize(dirty, { ADD_ATTR: ["target"] });
    } catch (_) {
      /* fall through */
    }
  }
  return renderMarkdownFallback(src);
}

function renderMarkdownFallback(src) {
  const lines = (src || "").replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      out.push(`<pre class="code"><code>${escHtml(buf.join("\n"))}</code></pre>`);
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { out.push(`<h${h[1].length}>${inlineMd(h[2])}</h${h[1].length}>`); i++; continue; }
    if (/^---+\s*$/.test(line)) { out.push("<hr>"); i++; continue; }
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(`<li>${inlineMd(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`); i++;
      }
      out.push(`<ul>${items.join("")}</ul>`); continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(`<li>${inlineMd(lines[i].replace(/^\s*\d+\.\s+/, ""))}</li>`); i++;
      }
      out.push(`<ol>${items.join("")}</ol>`); continue;
    }
    if (/^\s*$/.test(line)) { i++; continue; }
    const para = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !/^```/.test(lines[i]) &&
           !/^#{1,4}\s/.test(lines[i]) && !/^\s*[-*]\s+/.test(lines[i]) &&
           !/^\s*\d+\.\s+/.test(lines[i])) { para.push(lines[i]); i++; }
    out.push(`<p>${inlineMd(para.join("\n")).replace(/\n/g, "<br>")}</p>`);
  }
  return out.join("\n");
}

// ---------- Rendering ----------
function sandboxClass(mode) {
  if (mode === "read-only") return "ro";
  if (mode === "danger-full-access") return "full";
  return "";
}
function syncSandboxIco() {
  els.sandboxIco.className = "sandbox-ico " + sandboxClass(state.sandbox);
}

function renderSidebar() {
  els.list.innerHTML = "";
  els.empty.style.display = state.conversations.length ? "none" : "block";
  for (const c of state.conversations) {
    const li = document.createElement("li");
    li.className = "conv-item" + (c.id === state.currentId ? " active" : "");
    li.innerHTML =
      `<span class="tick"></span><span class="label"></span><button class="del" title="Delete">×</button>`;
    li.querySelector(".label").textContent = c.title || "New chat";
    li.querySelector(".label").addEventListener("click", () => selectConversation(c.id));
    li.querySelector(".tick").addEventListener("click", () => selectConversation(c.id));
    li.querySelector(".del").addEventListener("click", (e) => { e.stopPropagation(); deleteConversation(c.id); });
    els.list.appendChild(li);
  }
}

function clearMessages() { els.messages.innerHTML = ""; }

function addMessageEl(role, contentHtml) {
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  const avatar = role === "assistant" ? "M/S" : "you";
  row.innerHTML = `<div class="avatar">${avatar}</div><div class="bubble"></div>`;
  row.querySelector(".bubble").innerHTML = contentHtml;
  els.messages.appendChild(row);
  scrollToEnd();
  return row;
}

function scrollToEnd() { els.messages.scrollTop = els.messages.scrollHeight; }

function normalizeLiveNote(note) {
  if (typeof note === "string") return { role: "", text: note };
  return {
    role: note && note.role ? String(note.role) : "",
    text: note && note.text ? String(note.text) : "",
  };
}

function liveNoteHtml(note) {
  const normalized = normalizeLiveNote(note);
  const role = normalized.role ? `<span class="live-note-role">${escHtml(normalized.role)}</span>` : "";
  const text = escHtml(normalizeBackendText(normalized.text)).replace(/\n/g, "<br>");
  return `<div class="live-note">${role}<div class="live-note-text">${text}</div></div>`;
}

function liveNotesHtml(notes) {
  if (!Array.isArray(notes) || !notes.length) return "";
  return (
    `<div class="live-notes">
       <div class="role-title">Live Updates</div>
       <div class="live-note-list">${notes.map(liveNoteHtml).join("")}</div>
     </div>`
  );
}

function assistantMessageHtml(message) {
  return `${liveNotesHtml(message.live_notes)}<div class="final-md">${renderMarkdown(message.content)}</div>`;
}

function appendLiveNote(list, note) {
  if (!list) return;
  const template = document.createElement("template");
  template.innerHTML = liveNoteHtml(note);
  list.appendChild(template.content.firstElementChild);
}

function renderConversation(conv) {
  clearMessages();
  els.title.textContent = conv.title && conv.title !== "New chat" ? conv.title : "MLSim Assistant";
  if (!conv.messages || !conv.messages.length) {
    els.messages.innerHTML =
      `<div class="welcome"><h2>New conversation</h2><p>Ask a question, or run a profiling / simulation task. The assistant works inside <code>../main</code>.</p></div>`;
    return;
  }
  for (const m of conv.messages) {
    const row = addMessageEl(
      m.role,
      m.role === "assistant" ? assistantMessageHtml(m) : escHtml(m.content),
    );
    if (m.role === "assistant") rewriteImages(row.querySelector(".bubble"));
  }
}

// ---------- Actions ----------
async function refreshSidebar() {
  const data = await api.listConversations();
  state.conversations = data.conversations || [];
  renderSidebar();
}

async function selectConversation(id) {
  if (state.streaming) return;
  const conv = await api.getConversation(id);
  if (!conv) { await refreshSidebar(); return; }
  state.currentId = id;
  renderSidebar();
  renderConversation(conv);
  els.input.focus();
}

async function newConversation() {
  if (state.streaming) return;
  const conv = await api.createConversation(state.sandbox);
  state.currentId = conv.id;
  await refreshSidebar();
  renderConversation(conv);
  els.input.focus();
}

async function deleteConversation(id) {
  await api.deleteConversation(id);
  if (state.currentId === id) {
    state.currentId = null;
    clearMessages();
    els.messages.innerHTML = `<div class="welcome"><h2>Ask MLSim anything</h2><p>Start a new conversation from the left.</p></div>`;
    els.title.textContent = "MLSim Assistant";
  }
  await refreshSidebar();
}

// SSE over fetch (POST body needs fetch streaming, not EventSource).
async function streamTurn(cid, text, sandbox, handlers, signal) {
  const res = await fetch(`/api/conversations/${cid}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, sandbox_mode: sandbox }),
    signal,
  });
  if (!res.ok || !res.body) { handlers.done(`(request failed: ${res.status})`); return; }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      let dataStr = "";
      for (const l of chunk.split("\n")) {
        if (l.startsWith("event:")) event = l.slice(6).trim();
        else if (l.startsWith("data:")) dataStr += l.slice(5).trim();
      }
      let data = {};
      try { data = JSON.parse(dataStr); } catch { /* ignore */ }
      if (event === "session") handlers.session && handlers.session(data);
      else if (event === "progress") handlers.progress && handlers.progress(data.text || "");
      else if (event === "live_note" || event === "user_progress") {
        handlers.liveNote && handlers.liveNote(data);
      }
      else if (event === "orchestrator") handlers.orchestrator && handlers.orchestrator(data.text || "");
      else if (event === "implementer") handlers.implementer && handlers.implementer(data.text || "");
      else if (event === "done") handlers.done && handlers.done(data.text || "");
    }
  }
}

async function sendMessage(text) {
  if (state.streaming || !text.trim()) return;
  text = text.trim();

  if (!state.currentId) {
    const conv = await api.createConversation(state.sandbox);
    state.currentId = conv.id;
  }

  // Drop welcome screen on first message.
  const welcome = els.messages.querySelector(".welcome");
  if (welcome) welcome.remove();

  addMessageEl("user", escHtml(text));

  // Assistant placeholder with live progress.
  const row = addMessageEl("assistant",
    `<div class="live-notes" hidden>
       <div class="role-title">Live Updates</div>
       <div class="live-note-list"></div>
     </div>
     <div class="live-work">
       <details class="role-output orchestrator pending">
         <summary class="role-title">Orchestrator Raw</summary>
         <pre class="role-raw">Waiting for decision…</pre>
       </details>
       <div class="role-output implementer pending" hidden>
         <div class="role-title">Implementer Summary</div>
         <div class="role-md">Waiting for implementation…</div>
       </div>
       <div class="progress"><div class="dots"><span></span><span></span><span></span></div><div class="line cur"></div></div>
     </div>
     <div class="final-md" hidden></div>`);
  const bubble = row.querySelector(".bubble");
  const liveNotes = bubble.querySelector(".live-notes");
  const liveNoteList = bubble.querySelector(".live-note-list");
  const liveWork = bubble.querySelector(".live-work");
  const finalMd = bubble.querySelector(".final-md");
  const progressLine = bubble.querySelector(".line");
  const progressBox = bubble.querySelector(".progress");
  const orchestratorBox = bubble.querySelector(".role-output.orchestrator");
  const orchestratorRaw = bubble.querySelector(".role-raw");
  const implementerBox = bubble.querySelector(".role-output.implementer");
  const implementerMd = bubble.querySelector(".role-md");

  state.streaming = true;
  const controller = new AbortController();
  state.abortController = controller;
  els.send.disabled = true;
  els.stop.hidden = false;
  els.stop.disabled = false;

  try {
    await streamTurn(state.currentId, text, state.sandbox, {
      session: () => {},
      progress: (line) => {
        if (!line) return;
        if (progressLine) { progressLine.textContent = line; }
        scrollToEnd();
      },
      liveNote: (note) => {
        appendLiveNote(liveNoteList, note);
        if (liveNotes) liveNotes.hidden = false;
        scrollToEnd();
      },
      orchestrator: (text) => {
        if (orchestratorBox) orchestratorBox.classList.remove("pending");
        if (orchestratorRaw) {
          orchestratorRaw.textContent = normalizeBackendText(text) || "(empty orchestrator output)";
        }
        scrollToEnd();
      },
      implementer: (text) => {
        if (implementerBox) {
          implementerBox.hidden = false;
          implementerBox.classList.remove("pending");
        }
        if (implementerMd) {
          implementerMd.innerHTML = renderMarkdown(text || "(empty implementer summary)");
          rewriteImages(implementerMd);
        }
        scrollToEnd();
      },
      done: (answer) => {
        if (progressBox) progressBox.remove();
        if (liveWork) liveWork.remove();
        if (finalMd) {
          finalMd.hidden = false;
          finalMd.innerHTML = renderMarkdown(answer);
          rewriteImages(finalMd);
        } else {
          bubble.innerHTML = renderMarkdown(answer);
          rewriteImages(bubble);
        }
        scrollToEnd();
      },
    }, controller.signal);
  } catch (e) {
    if (e && e.name === "AbortError") {
      if (progressBox) progressBox.remove();
      const note = document.createElement("div");
      note.className = "stop-note";
      note.textContent = "Stopped.";
      bubble.appendChild(note);
      scrollToEnd();
    } else {
      bubble.innerHTML = renderMarkdown(`(error talking to backend: ${e})`);
    }
  } finally {
    state.streaming = false;
    state.abortController = null;
    els.send.disabled = false;
    els.stop.hidden = true;
    els.stop.disabled = false;
    await refreshSidebar();
    els.input.focus();
  }
}

// ---------- Wiring ----------
function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
}

function init() {
  // open sanitized links in a new tab
  if (window.DOMPurify) {
    window.DOMPurify.addHook("afterSanitizeAttributes", (node) => {
      if (node.tagName === "A") {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener");
      }
    });
  }

  // sandbox knob
  els.sandbox.value = state.sandbox;
  syncSandboxIco();
  els.sandbox.addEventListener("change", () => {
    state.sandbox = els.sandbox.value;
    localStorage.setItem("mlsim_sandbox", state.sandbox);
    syncSandboxIco();
  });

  els.newChat.addEventListener("click", newConversation);
  els.stop.addEventListener("click", () => {
    if (state.streaming && state.abortController) {
      els.stop.disabled = true;
      state.abortController.abort();
    }
  });

  els.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = els.input.value;
    els.input.value = "";
    autoGrow();
    sendMessage(text);
  });

  els.input.addEventListener("input", autoGrow);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.composer.requestSubmit();
    }
  });

  // suggestion chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.input.value = chip.dataset.fill;
      autoGrow();
      els.input.focus();
    });
  });

  // initial load: show most recent conversation if any
  refreshSidebar().then(() => {
    if (state.conversations.length) selectConversation(state.conversations[0].id);
  });
}

init();
