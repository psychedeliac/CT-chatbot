/*
 * ct-chat-widget.js — drop-in chat widget for the Corporate Turnaround API.
 *
 * Embed with one tag; there is no build step and nothing is fetched from a CDN
 * (a debt-relief site should not hand a third party a script tag on every page
 * view, and a CDN outage would take the widget down with it):
 *
 *   <script src="/ct-chat-widget.js"
 *           data-api="https://ct-chatbot-production.up.railway.app"
 *           data-accent="#0b5cff"></script>
 *
 * It renders inside a Shadow DOM, so the host page's CSS cannot reach in and
 * the widget's CSS cannot leak out. That also fixes the reported bug where the
 * panel rendered underneath hero images: the root is `position: fixed` in its
 * own stacking context at the top of the z-index range.
 *
 * This is a reference implementation as much as a deliverable -- it is the
 * behaviour the API expects from any client, in one readable file:
 *   - render `done.answer`, never the accumulated deltas (the guards can
 *     replace an answer wholesale);
 *   - assistant text is Markdown, user text is never;
 *   - keep the session id for the tab and send it on every turn.
 */
(function () {
  "use strict";

  var script = document.currentScript;
  var CONFIG = {
    api: (script && script.dataset.api) || "http://localhost:8000",
    title: (script && script.dataset.title) || "Corporate Turnaround",
    subtitle: (script && script.dataset.subtitle) || "Business debt help",
    accent: (script && script.dataset.accent) || "#0b5cff",
    greeting:
      (script && script.dataset.greeting) ||
      "Hi — I can help with business debt, MCAs, creditor calls and more. What's going on?",
  };

  // sessionStorage, not localStorage: a conversation belongs to this tab and
  // this visit. The server expires sessions anyway, so a stale id restored a
  // week later would just be rejected and replaced.
  var SESSION_KEY = "ct-chat-session";
  var PHONE_PATTERN = /\b(1-800-889-0232|1-800-411-1113|988)\b/g;

  /* ── Markdown ────────────────────────────────────────────────────────────
   * A deliberately tiny renderer for the subset the backend emits: bullets,
   * ordered lists, paragraphs, bold, and the published phone numbers.
   *
   * Escaping happens FIRST and unconditionally, so nothing that arrives over
   * the wire can become markup. Only patterns matched after escaping become
   * tags. Raw HTML is never honoured -- there is no case where this API needs
   * to emit any, and allowing it would turn a prompt-injected answer into
   * script execution on the client's site.
   */
  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function inline(text) {
    return escapeHtml(text)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?]|$)/g, "$1<em>$2</em>")
      // Tap-to-call on mobile. Only our own published numbers are linked --
      // auto-linking every digit string would turn a figure in an answer into
      // a phone link.
      .replace(PHONE_PATTERN, function (number) {
        var tel = number.replace(/-/g, "");
        return '<a href="tel:' + tel + '">' + number + "</a>";
      });
  }

  var BULLET = /^\s*[-*]\s+/;
  var ORDERED = /^\s*\d+[.)]\s+/;

  function renderMarkdown(text) {
    // Line by line, grouping consecutive lines of the same kind.
    //
    // Grouping per blank-line-delimited block instead (and requiring the whole
    // block be bullets) does the wrong thing on the shape the model actually
    // produces most often -- a lead-in sentence immediately followed by its
    // bullets, with no blank line between them. That rendered the entire list
    // as one run-on paragraph.
    var lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
    var html = "";
    var run = [];
    var kind = null; // "ul" | "ol" | "p"

    function flush() {
      if (!run.length) return;
      if (kind === "ul" || kind === "ol") {
        html +=
          "<" + kind + ">" +
          run.map(function (l) { return "<li>" + inline(l) + "</li>"; }).join("") +
          "</" + kind + ">";
      } else {
        html += "<p>" + run.map(inline).join("<br>") + "</p>";
      }
      run = [];
      kind = null;
    }

    lines.forEach(function (line) {
      if (!line.trim()) {          // blank line ends whatever was open
        flush();
        return;
      }
      var next, content;
      if (BULLET.test(line)) {
        next = "ul";
        content = line.replace(BULLET, "");
      } else if (ORDERED.test(line)) {
        next = "ol";
        content = line.replace(ORDERED, "");
      } else {
        next = "p";
        content = line;
      }
      if (kind && next !== kind) flush();
      kind = next;
      run.push(content);
    });

    flush();
    return html;
  }

  /* ── Styles ─────────────────────────────────────────────────────────────── */

  var STYLE = `
    :host { all: initial; }
    * { box-sizing: border-box; }

    .root {
      position: fixed; inset: 0; pointer-events: none;
      /* Above every plausible page layer. The panel used to share a stacking
         context with page content and lost to full-bleed hero images. */
      z-index: 2147483000;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   "Helvetica Neue", Arial, sans-serif;
      font-size: 15px; line-height: 1.55; color: #14181f;
    }
    .root > * { pointer-events: auto; }

    .launcher {
      position: fixed; right: 20px; bottom: 20px;
      width: 56px; height: 56px; border-radius: 50%;
      border: none; cursor: pointer; background: var(--accent); color: #fff;
      box-shadow: 0 6px 24px rgba(16, 24, 40, .28);
      display: grid; place-items: center;
      transition: transform .15s ease, box-shadow .15s ease;
    }
    .launcher:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(16,24,40,.34); }
    .launcher:focus-visible { outline: 3px solid #fff; outline-offset: 3px; }
    .launcher svg { width: 26px; height: 26px; fill: currentColor; }
    .launcher[hidden] { display: none; }

    .panel {
      position: fixed; right: 20px; bottom: 20px;
      width: min(400px, calc(100vw - 32px));
      height: min(640px, calc(100vh - 40px));
      background: #fff; border-radius: 16px; overflow: hidden;
      display: flex; flex-direction: column;
      box-shadow: 0 24px 64px rgba(16, 24, 40, .28);
      animation: rise .18s ease-out;
    }
    .panel[hidden] { display: none; }
    @keyframes rise { from { opacity: 0; transform: translateY(12px); } }
    @media (prefers-reduced-motion: reduce) {
      .panel { animation: none; }
      .launcher { transition: none; }
    }

    header {
      background: var(--accent); color: #fff;
      padding: 14px 16px; display: flex; align-items: center; gap: 12px;
    }
    .avatar {
      width: 34px; height: 34px; border-radius: 50%; flex: none;
      background: rgba(255,255,255,.22); display: grid; place-items: center;
      font-weight: 600; font-size: 13px;
    }
    .titles { flex: 1; min-width: 0; }
    .titles b { display: block; font-size: 15px; font-weight: 600; }
    .titles span { display: block; font-size: 12.5px; opacity: .85; }
    .close {
      background: transparent; border: none; color: #fff; cursor: pointer;
      font-size: 22px; line-height: 1; padding: 4px 6px; border-radius: 6px;
    }
    .close:hover { background: rgba(255,255,255,.16); }
    .close:focus-visible { outline: 2px solid #fff; }

    .log {
      flex: 1; overflow-y: auto; padding: 16px;
      display: flex; flex-direction: column; gap: 12px;
      background: #f6f8fb; scroll-behavior: smooth;
    }
    @media (prefers-reduced-motion: reduce) { .log { scroll-behavior: auto; } }

    .msg { max-width: 86%; }
    .msg.user { align-self: flex-end; }
    .msg.bot  { align-self: flex-start; }
    .bubble {
      padding: 10px 13px; border-radius: 14px;
      overflow-wrap: anywhere; word-break: break-word;
    }
    .user .bubble { background: var(--accent); color: #fff; border-bottom-right-radius: 4px; }
    .bot  .bubble { background: #fff; border: 1px solid #e5e9f0; border-bottom-left-radius: 4px; }
    .bubble p { margin: 0 0 8px; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble ul, .bubble ol { margin: 6px 0; padding-left: 20px; }
    .bubble li { margin: 3px 0; }
    .bubble a { color: inherit; font-weight: 600; }
    .user .bubble a { color: #fff; }
    .bot  .bubble a { color: var(--accent); }

    .label { font-size: 11.5px; color: #667085; margin: 0 0 4px 2px; }

    .dots { display: inline-flex; gap: 4px; padding: 4px 2px; }
    .dots i {
      width: 7px; height: 7px; border-radius: 50%; background: #98a2b3;
      animation: blink 1.2s infinite ease-in-out;
    }
    .dots i:nth-child(2) { animation-delay: .18s; }
    .dots i:nth-child(3) { animation-delay: .36s; }
    @keyframes blink { 0%, 80%, 100% { opacity: .3; } 40% { opacity: 1; } }
    @media (prefers-reduced-motion: reduce) { .dots i { animation: none; opacity: .6; } }

    .rate { display: flex; gap: 6px; margin: 6px 0 0 2px; align-items: center; }
    .rate button {
      background: transparent; border: 1px solid #e5e9f0; border-radius: 7px;
      cursor: pointer; padding: 2px 7px; font-size: 13px; line-height: 1.4; color: #667085;
    }
    .rate button:hover { background: #fff; border-color: #cbd5e1; }
    .rate button:focus-visible { outline: 2px solid var(--accent); }
    .rate .thanks { font-size: 11.5px; color: #667085; }

    .chips { display: flex; flex-wrap: wrap; gap: 7px; padding: 0 16px 12px; background: #f6f8fb; }
    .chips:empty { display: none; }
    .chips button {
      background: #fff; border: 1px solid #d7dfea; color: #1f2937;
      border-radius: 999px; padding: 7px 12px; font-size: 13px;
      cursor: pointer; text-align: left; font-family: inherit;
    }
    .chips button:hover { border-color: var(--accent); color: var(--accent); }
    .chips button:focus-visible { outline: 2px solid var(--accent); }

    form { display: flex; gap: 8px; padding: 12px; border-top: 1px solid #e5e9f0; background: #fff; }
    textarea {
      flex: 1; resize: none; border: 1px solid #d7dfea; border-radius: 10px;
      padding: 9px 11px; font: inherit; max-height: 108px; color: inherit;
    }
    textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(11,92,255,.12); }
    .send {
      background: var(--accent); color: #fff; border: none; border-radius: 10px;
      width: 42px; cursor: pointer; display: grid; place-items: center; flex: none;
    }
    .send:disabled { opacity: .5; cursor: default; }
    .send svg { width: 19px; height: 19px; fill: currentColor; }

    .error { color: #b42318; font-size: 13px; }
    .sr {
      position: absolute; width: 1px; height: 1px; overflow: hidden;
      clip: rect(0 0 0 0); white-space: nowrap;
    }
  `;

  /* ── Markup ─────────────────────────────────────────────────────────────── */

  var host = document.createElement("div");
  host.setAttribute("data-ct-chat", "");
  var shadow = host.attachShadow({ mode: "open" });
  shadow.innerHTML =
    "<style>" + STYLE + "</style>" +
    '<div class="root" style="--accent:' + CONFIG.accent.replace(/["\\<>]/g, "") + '">' +
      '<button class="launcher" aria-label="Open chat">' +
        '<svg viewBox="0 0 24 24"><path d="M12 3C6.98 3 3 6.58 3 11c0 2.1.9 4 2.4 5.4L4.5 21l4.9-2.2c.8.2 1.7.3 2.6.3 5.02 0 9-3.58 9-8s-3.98-8-9-8z"/></svg>' +
      "</button>" +
      '<section class="panel" hidden aria-label="Chat with Corporate Turnaround">' +
        "<header>" +
          '<div class="avatar">CT</div>' +
          '<div class="titles"><b></b><span></span></div>' +
          '<button class="close" aria-label="Close chat">&times;</button>' +
        "</header>" +
        '<div class="log" role="log" aria-live="polite" aria-atomic="false"></div>' +
        '<div class="chips"></div>' +
        "<form>" +
          '<label class="sr" for="ct-input">Your message</label>' +
          '<textarea id="ct-input" rows="1" placeholder="Ask about your business debt..." maxlength="2000"></textarea>' +
          '<button class="send" type="submit" aria-label="Send message">' +
            '<svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>' +
          "</button>" +
        "</form>" +
      "</section>" +
    "</div>";

  var $ = function (sel) { return shadow.querySelector(sel); };
  var launcher = $(".launcher"),
      panel = $(".panel"),
      log = $(".log"),
      chips = $(".chips"),
      form = $("form"),
      input = $("textarea"),
      sendButton = $(".send");

  $(".titles b").textContent = CONFIG.title;
  $(".titles span").textContent = CONFIG.subtitle;

  /* ── Rendering ──────────────────────────────────────────────────────────── */

  var busy = false;

  function nearBottom() {
    return log.scrollHeight - log.scrollTop - log.clientHeight < 120;
  }

  function scrollDown(force) {
    // Never yank the view while someone is scrolled up reading an earlier
    // answer -- only follow along if they were already at the bottom.
    if (force || nearBottom()) log.scrollTop = log.scrollHeight;
  }

  function addMessage(role, text) {
    var wrap = document.createElement("div");
    wrap.className = "msg " + role;

    if (role === "bot") {
      var label = document.createElement("div");
      label.className = "label";
      label.textContent = CONFIG.title;
      wrap.appendChild(label);
    }

    var bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "user") {
      // Never Markdown, never innerHTML: this is untrusted input, and the only
      // thing rendering it as markup could add is a way to inject into the page.
      bubble.textContent = text;
    } else {
      bubble.innerHTML = renderMarkdown(text);
    }

    wrap.appendChild(bubble);
    log.appendChild(wrap);
    scrollDown(role === "user");
    return { wrap: wrap, bubble: bubble };
  }

  function addTypingBubble() {
    var msg = addMessage("bot", "");
    msg.bubble.innerHTML = '<span class="dots"><i></i><i></i><i></i></span>';
    scrollDown(true);
    return msg;
  }

  function renderChips(list) {
    chips.textContent = "";
    (list || []).forEach(function (question) {
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = question;
      button.addEventListener("click", function () {
        chips.textContent = "";
        send(question);
      });
      chips.appendChild(button);
    });
  }

  function addRating(msg, answerId) {
    if (!answerId) return;
    var row = document.createElement("div");
    row.className = "rate";

    ["up", "down"].forEach(function (verdict) {
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = verdict === "up" ? "\uD83D\uDC4D" : "\uD83D\uDC4E";
      button.setAttribute(
        "aria-label", verdict === "up" ? "This answer helped" : "This answer missed"
      );
      button.addEventListener("click", function () {
        // Fire and forget. A rating that fails to record is not worth
        // interrupting the conversation over.
        fetch(CONFIG.api + "/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ answer_id: answerId, verdict: verdict }),
        }).catch(function () {});
        row.textContent = "";
        var thanks = document.createElement("span");
        thanks.className = "thanks";
        thanks.textContent = "Thanks — that helps us improve.";
        row.appendChild(thanks);
      });
      row.appendChild(button);
    });

    msg.wrap.appendChild(row);
  }

  /* ── Transport ──────────────────────────────────────────────────────────── */

  function sessionId() {
    try { return sessionStorage.getItem(SESSION_KEY); } catch (e) { return null; }
  }

  function rememberSession(id) {
    try { sessionStorage.setItem(SESSION_KEY, id); } catch (e) { /* private mode */ }
  }

  function setBusy(value) {
    busy = value;
    sendButton.disabled = value;
    input.disabled = value;
  }

  async function send(text) {
    var message = String(text || "").trim();
    if (!message || busy) return;

    setBusy(true);
    chips.textContent = "";
    addMessage("user", message);
    var pending = addTypingBubble();
    var streamed = "";
    var done = null;

    try {
      var response = await fetch(CONFIG.api + "/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message, session_id: sessionId() }),
      });

      if (!response.ok || !response.body) {
        throw new Error("HTTP " + response.status);
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      while (true) {
        var step = await reader.read();
        if (step.done) break;
        buffer += decoder.decode(step.value, { stream: true });

        // SSE frames are separated by a blank line; a frame can arrive split
        // across reads, so only whole frames are consumed and the remainder
        // stays in the buffer.
        var frames = buffer.split("\n\n");
        buffer = frames.pop();

        frames.forEach(function (frame) {
          var line = frame.split("\n").find(function (l) {
            return l.indexOf("data:") === 0;
          });
          if (!line) return;

          var event;
          try { event = JSON.parse(line.slice(5).trim()); } catch (e) { return; }

          if (event.type === "session") {
            rememberSession(event.session_id);
          } else if (event.type === "delta") {
            streamed += event.text;
            pending.bubble.innerHTML = renderMarkdown(streamed);
            scrollDown(false);
          } else if (event.type === "done") {
            done = event;
          } else if (event.type === "error") {
            throw new Error(event.error);
          }
        });
      }

      if (done) {
        // The authoritative text. The guards can replace an answer wholesale
        // (an ungrounded reply becomes a refusal), so what streamed is
        // provisional and must be overwritten rather than appended to.
        pending.bubble.innerHTML = renderMarkdown(done.answer);
        addRating(pending, done.answer_id);
        renderChips(done.suggestions);
      } else if (streamed) {
        pending.bubble.innerHTML = renderMarkdown(streamed);
      } else {
        throw new Error("empty response");
      }
    } catch (error) {
      pending.bubble.innerHTML = "";
      var note = document.createElement("p");
      note.className = "error";
      note.textContent =
        "Sorry — I couldn't reach our team just now. Please try again, or call " +
        "1-800-889-0232.";
      pending.bubble.appendChild(note);
    } finally {
      setBusy(false);
      scrollDown(false);
      input.focus();
    }
  }

  /* ── Wiring ─────────────────────────────────────────────────────────────── */

  function open() {
    panel.hidden = false;
    launcher.hidden = true;
    if (!log.childElementCount) {
      addMessage("bot", CONFIG.greeting);
      renderChips([
        "What services do you offer?",
        "How does the program work?",
        "What happens during a free consultation?",
      ]);
    }
    input.focus();
  }

  function close() {
    panel.hidden = true;
    launcher.hidden = false;
    launcher.focus();
  }

  launcher.addEventListener("click", open);
  $(".close").addEventListener("click", close);

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var text = input.value;
    input.value = "";
    input.style.height = "auto";
    send(text);
  });

  input.addEventListener("input", function () {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 108) + "px";
  });

  input.addEventListener("keydown", function (event) {
    // Enter sends, Shift+Enter is a newline -- what every chat app does.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.dispatchEvent(new Event("submit"));
    }
  });

  shadow.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !panel.hidden) close();
  });

  // Exposed for widget/test-markdown.js only. The renderer is the one piece of
  // real logic in here and it is worth a check that runs against this exact
  // file rather than a copy of it; everything else stays closure-private.
  host.__renderMarkdown = renderMarkdown;

  document.body.appendChild(host);
})();
