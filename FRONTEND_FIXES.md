# Corporate Turnaround Chat Widget — Frontend Integration & Fix Guide

**Audience:** the developer who owns the chat widget on `corpo-nine.vercel.app`.

This document is self-contained. You do not need access to the backend
repository to implement any of it — every endpoint, payload, event and edge case
you need is written out below, with real captured responses from the production
API.

**The backend is live and finished.** Nothing here is blocked on us.

Two of the fixes below (1 and 2) are bugs the client noticed. Three are new
capabilities the API gained that your widget does not use yet.

---

## Contents

1. [Fastest path: embed the hosted widget](#0-fastest-path-embed-the-hosted-widget)
2. [The API](#the-api) — base URL, endpoints, exact wire format, errors, limits
3. [What to fix and build](#what-to-fix-and-build) — the seven items
4. [Complete reference implementation](#complete-reference-implementation) — React + TypeScript
5. [Verification checklist](#verification-checklist)
6. [What to ask the backend team for](#what-to-ask-the-backend-team-for)

---

## 0. Fastest path: embed the hosted widget

Before building anything, know that a complete, working widget is already
hosted and public. If dropping it in is acceptable, this is the whole job:

```html
<script src="https://ct-chatbot-production.up.railway.app/widget.js"
        data-api="https://ct-chatbot-production.up.railway.app"
        data-accent="#0b5cff"></script>
```

It is dependency-free vanilla JS in a Shadow DOM (so it cannot collide with your
CSS), and it implements every item in this document. Configuration is via
`data-` attributes:

| Attribute | Default | Purpose |
|---|---|---|
| `data-api` | `http://localhost:8000` | API base URL — set this |
| `data-title` | `Corporate Turnaround` | Header name and message label |
| `data-subtitle` | `Business debt help` | Header second line |
| `data-accent` | `#0b5cff` | Brand colour |
| `data-greeting` | see source | First message, shown before any request |

**Open that URL in a browser** — the source is readable and commented, and it is
the executable reference for everything described below. Where this document and
that file disagree, the file is correct: it is the version that is tested and
running.

If you need the widget inside your React tree instead (your own styling, your
own state, analytics hooks), keep reading.

---

## The API

**Base URL:** `https://ct-chatbot-production.up.railway.app`

### CORS

The API uses an explicit origin allowlist, not `*`. `https://corpo-nine.vercel.app`
is already allowed. **Any new domain — a custom production domain, a staging
host, Vercel preview URLs, `http://localhost:3000` for your own development —
must be added by the backend team before requests from it will work.** See
[the last section](#what-to-ask-the-backend-team-for).

A blocked origin shows up in the console as a CORS preflight failure, not as an
API error. If requests work in curl and fail in the browser, this is why.

Allowed methods: `GET`, `POST`, `OPTIONS`. Allowed request header:
`Content-Type`. Credentials are not used — do not send cookies.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/chat/stream` | Streaming turn (Server-Sent Events). **Use this one.** |
| `POST /api/chat` | Non-streaming equivalent, single JSON response |
| `POST /api/feedback` | Rate an answer |
| `GET /health` | `{"status":"ok","active_sessions":0}` |
| `GET /widget.js` | The reference widget |

### `POST /api/chat/stream`

Request:

```json
{ "message": "do you help with SBA loan defaults", "session_id": null }
```

- `message` — required, 1 to 2000 characters.
- `session_id` — the id from a previous turn, or `null`/omitted on the first
  message of a conversation. An id we did not issue is ignored and a fresh one
  is returned; you cannot forge your way into someone else's conversation.

Response is `text/event-stream`. Each event is a line beginning `data: `
followed by JSON, and events are separated by a **blank line**. This is real
captured output:

```
data: {"type": "session", "session_id": "w88kkCEf6DCYiPKR5UGpPhhI56SpbfCK"}

data: {"type": "delta", "text": "Yes,"}

data: {"type": "delta", "text": " SBA loan defaults and bank loan workouts are one of the areas we specialize in. We negotiate"}

data: {"type": "delta", "text": " directly with the lender on workout terms and loan modifications, with the goal of preventing foreclosure on business or personal assets pledged against"}

data: {"type": "done", "answer": "Yes, SBA loan defaults and bank loan workouts are one of the areas we specialize in. We negotiate directly with the lender on workout terms and loan modifications, with the goal of preventing foreclosure on business or personal assets pledged against the loan. \n\nBecause an SBA or bank loan usually sits alongside a personal guarantee and other business debt, we look at the whole picture rather than that one loan on its own. What can actually be arranged depends on your lender, where the loan is in the default process, and what your business can afford — which is exactly what the free consultation works out. \n\nCall us at 1-800-889-0232 whenever you are ready to talk through your situation.", "cached": true, "suggestions": ["What can Corporate Turnaround actually do for my business?", "What happens when you default on a Merchant Cash Advance?", "How is an MCA different from a business loan?"], "answer_id": "mZl6IhF40e2FVzGk"}
```

**Event order:** `session` (always first, exactly once) → `delta` (zero or more)
→ `done` (exactly once). On failure, an `error` event replaces `done`.

| Event | Fields |
|---|---|
| `session` | `session_id: string` — store it, send it on every later turn |
| `delta` | `text: string` — append to the bubble. **Provisional** (see below) |
| `done` | `answer: string`, `cached: boolean`, `suggestions: string[]`, `answer_id: string` |
| `error` | `error: string` — a user-safe message, display it as-is |

Notes on the shapes:

- `delta.text` chunks are **not** word- or line-aligned. Concatenate them
  verbatim; do not add spaces or newlines between them.
- `answer` contains `\n` and `\n\n` and Markdown bullets. It is **not** HTML.
- `suggestions` is `[]` when the assistant declines a question.
- `cached: true` means a repeat of a common first question was served from
  cache. Nothing to display — it is there for debugging. Note that a cached
  turn still streams a `delta`, it just arrives as one chunk.

### `POST /api/chat` (non-streaming)

Same request body. Returns one JSON object:

```json
{
  "session_id": "xsJp7eIE7NHuSGFIfIn8ace3Q25RAv3s",
  "answer": "We specialize in helping small businesses get out of debt...",
  "cached": false,
  "suggestions": ["What does Corporate Turnaround do?", "..."],
  "answer_id": "cbJk89426fUAz5uk"
}
```

Simpler, but the user stares at a blank pane for the whole generation. Prefer
the streaming endpoint; this exists for non-browser callers and as a fallback.

### `POST /api/feedback`

```json
{ "answer_id": "mZl6IhF40e2FVzGk", "verdict": "down", "comment": "too vague" }
```

- `verdict` — `"up"` or `"down"` exactly. Anything else is a `422`.
- `comment` — optional, max 1000 characters.
- Success: `200 {"ok": true}`.
- `404 {"error": "That message is no longer available to rate.", "request_id": "..."}`
  — the id expired (they live one hour) or was never issued. Nothing to show the
  user.

Send only the `answer_id`. The server already knows the question, the answer,
and which knowledge-base records produced it — that is what makes a thumbs-down
actionable, and it means you never have to hold or transmit that data.

### Errors and status codes

Every error except FastAPI's own validation errors returns this shape:

```json
{ "error": "Human-readable, user-safe message", "request_id": "e1d02d7ef887" }
```

| Status | Meaning | What to show |
|---|---|---|
| `422` | Validation failed — empty message, or over 2000 chars | Prevent it client-side; don't let the user send an empty or oversized message |
| `429` | Rate limited | "You're sending messages a little quickly — give it a moment." Not an error tone |
| `503` | Server at capacity, or still warming up after a deploy | The `error` field's text; suggest retrying |
| `504` | The turn took too long | The `error` field's text |
| `500` | Unexpected | The `error` field's text — it already includes the phone number |

`422` is the one exception: it is FastAPI's format, not ours.

```json
{"detail":[{"type":"string_too_short","loc":["body","message"],"msg":"String should have at least 1 character","input":"","ctx":{"min_length":1}}]}
```

Don't try to render that. Validate before sending instead.

**Errors mid-stream.** Once the SSE response has started, the HTTP status is
already committed to `200`, so a later failure arrives as an `error` **event**,
not a status code. You must handle both: a non-`ok` response *and* an `error`
event.

`request_id` is also returned as the `X-Request-ID` response header on every
request. If you report a problem to the backend team, include it — it finds the
exact request in the logs.

### Rate limits

Per client IP: **10 requests/minute** and **100 requests/hour**. Responses carry
`X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset` if you want
to surface it.

This is generous for a human typing but easy to trip in a dev loop or a test
suite that hammers the endpoint. A `429` in development is usually your own
tooling.

### Sessions

- A session id is a 32-character URL-safe token.
- Store it for the tab and the visit — `sessionStorage`, not `localStorage`.
  Conversations expire server-side, so a week-old id restored from
  `localStorage` is just rejected and replaced, and reviving a stale thread is
  the wrong behaviour anyway.
- Send it on every turn after the first. **If you don't, every message starts a
  new conversation with no history**, and follow-ups like "what about the fees?"
  lose their subject entirely.
- The server keeps the last several turns and replays them into the model. You
  do not need to send history yourself — only the id.

---

## What to fix and build

| # | Priority | Item |
|---|----------|------|
| 1 | **HIGH** | Render assistant replies as Markdown; user messages stay plain text |
| 2 | **HIGH** | `position: fixed` in a high-z-index root so the panel can't hide under page images |
| 3 | **HIGH** | Typing indicator on send, stream `delta` tokens, swap to `done.answer` |
| 4 | MEDIUM | Follow-up suggestion chips from `done.suggestions` |
| 5 | MEDIUM | Thumbs up/down posting to `/api/feedback` |
| 6 | LOW | Bubble styling, avatar, auto-scroll |
| 7 | INFO | Latency and the occasional garbled token — nothing to do |

### 1. Render the assistant reply as Markdown (HIGH)

**Symptom.** Bullet lists show literal asterisks. A real reply renders as:

```
We help small businesses resolve debt...

* Business debt negotiation and settlement
* Merchant cash advance relief
* Creditor harassment protection
```

Those `*` and the paragraph breaks are **Markdown** — the backend emits it
deliberately. The widget prints it as raw text, so it reads like a debug console
instead of a formatted answer.

**Fix.**

```bash
npm i react-markdown
```

```tsx
import ReactMarkdown from "react-markdown";

<div className="prose prose-sm max-w-none">
  <ReactMarkdown>{message.text}</ReactMarkdown>
</div>
```

Rules that matter:

- **Assistant messages only.** The user's own message must render as plain text.
  It is untrusted input, and rendering it as markup is how someone pastes HTML
  into your page.
- **No `rehype-raw`.** `react-markdown` ignores raw HTML by default; keep it
  that way. The assistant's text is model-generated and can be influenced by
  what a user types, so treat it as untrusted too — the Markdown subset is all
  that is ever needed.
- **Phone numbers stay plain text.** Do not turn them into `tel:` links.
- Constrain Tailwind Typography (`prose-sm`, `max-w-none`) for the narrow chat
  column.
- Render `done.answer` (see item 3). Running each partial through the renderer
  as it streams is fine — half-finished Markdown degrades gracefully.

**The one edge case that will catch you out.** The model very often emits a
lead-in line immediately followed by its bullets, with **no blank line between
them**:

```
Here are the services we provide:
- Business debt negotiation
- MCA debt relief
```

`react-markdown` handles this correctly. A hand-rolled renderer usually does
not — the naive approach splits the text on blank lines and requires a whole
block to be bullets, which turns the above into one run-on paragraph. This was a
real bug in the reference widget before it was fixed. **If you write your own
renderer, test this exact shape first.**

Elements you need to style: `p`, `ul`, `ol`, `li`, `strong`, `em`. Nothing else
appears.

### 2. Widget can render *underneath* page content (HIGH)

**Symptom.** With the chat open and the page scrolled to the hero/CTA image
section, the message input is **not clickable** — clicks land on the background
image instead.

**Root cause.** The panel is positioned `absolute inset-4` and shares a stacking
context with page content, so full-bleed images (`object-cover`, creating their
own stacking context) paint over it depending on scroll position.

**Fix.** Pin the widget to the viewport in its own top-level stacking context,
ideally via a portal so it is a direct child of `<body>` and no ancestor's
`transform`, `filter` or `overflow` can trap it:

```tsx
createPortal(
  <div className="fixed inset-0 z-[9999] pointer-events-none">
    <div className="pointer-events-auto fixed bottom-5 right-5">
      {/* launcher + panel */}
    </div>
  </div>,
  document.body
)
```

Checklist:

- `fixed`, not `absolute`, for the widget root and the panel.
- A `z-index` above every page section, header and image.
- Root `pointer-events-none` with interactive children `pointer-events-auto`, so
  the widget never eats clicks on the page behind it.
- Render through a portal at `document.body` (or a Shadow DOM, which the
  reference widget uses and which also stops page CSS reaching into the panel).

**How to verify** — this is the check that actually proves it. Open the chat,
then at several scroll positions:

```js
const el = document.querySelector('#your-chat-input');
const r = el.getBoundingClientRect();
document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
// Must resolve to your widget (or its portal root). If it returns a page
// element — an <img>, a <section> — the bug is still there.
```

The original bug was scroll-position dependent, so test with the page at the
hero, mid-page, and the footer.

### 3. Stream tokens live + show an instant loading state (HIGH)

**Symptom.** After sending, the panel sits blank until the whole answer appears
at once. Even when the backend is fast, a blank pane reads as "broken."

**Cause.** The widget waits for the full response. The API already streams — this
is purely a render-side fix.

**Fix.**

1. The instant the user sends, append an assistant bubble containing a **typing
   indicator** (three animated dots). Never leave the pane empty.
2. Consume the SSE stream and append each `delta.text` to that bubble.
3. On `done`, **replace** the accumulated text with `done.answer` and drop the
   indicator.
4. On `error` (event or network failure), replace the bubble with the message.

> **The `done` replacement is not optional, and it is not cosmetic.**
>
> The server runs compliance guards after generation. They can replace an answer
> **wholesale** — an ungrounded reply becomes a refusal. If you render the
> accumulated deltas as final, the user watches a non-compliant answer finish
> typing and it stays on screen.
>
> This is a regulated industry (debt relief, FTC). Always render `done.answer`
> over whatever streamed. Treat the deltas as a progress animation, not content.

**Parsing the stream.** SSE frames are separated by a blank line, and a frame can
arrive split across two network reads. Buffer, split on `\n\n`, and keep the
remainder — do not assume one read equals one event:

```ts
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  const frames = buffer.split("\n\n");
  buffer = frames.pop() ?? "";          // incomplete frame stays buffered

  for (const frame of frames) {
    const line = frame.split("\n").find((l) => l.startsWith("data:"));
    if (!line) continue;
    const event = JSON.parse(line.slice(5).trim());
    // ...handle event
  }
}
```

Note `fetch` + `ReadableStream`, not `EventSource` — `EventSource` cannot issue
a POST.

### 4. Follow-up suggestion chips (MEDIUM — new)

`done.suggestions` carries up to three follow-up questions.

**What to build.** Render them as tappable chips under the latest assistant
message. Tapping one sends it as the user's next message. Clear the row as soon
as the user sends anything, so a stale row never sits under a newer answer.

These are drawn from the knowledge base, not invented by the model, so every
chip has a hand-authored answer behind it — tapping one cannot produce a
refusal. Show the text as-is; do not rewrite or truncate it (they are capped at
72 characters server-side).

`suggestions` is `[]` when the assistant declines a question. Render nothing.

You can also show a few starter chips before the first message. Three that
always resolve well:

```
What services do you offer?
How does the program work?
What happens during a free consultation?
```

### 5. Answer feedback — thumbs up/down (MEDIUM — new)

Every `done` event carries an opaque `answer_id`. A thumbs control on assistant
messages posts it back:

```ts
await fetch(`${API}/api/feedback`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ answer_id, verdict: "up" }),
});
```

- Fire and forget. A rating that fails to record is not worth interrupting the
  conversation over — wrap it in `.catch(() => {})`.
- Replace the buttons with a short thank-you on click, so it is obvious it
  registered and cannot be double-submitted.
- Optionally offer a comment box on thumbs-down and send it as `comment`.
- A `404` means the id aged out. Silently ignore.

This is how the knowledge base gets improved, so it is worth doing properly:
each thumbs-down becomes a specific record to fix rather than a vague complaint.

### 6. Message styling / "feel" (LOW, polish)

Current bubbles read as an undifferentiated wall of text. Cheap wins:

- Distinct styling for **user** (right-aligned, brand fill) vs **assistant**
  (left-aligned, light surface).
- A small assistant avatar or label to anchor the conversation.
- Comfortable line-height and spacing between messages.
- Auto-scroll to the newest message as it streams — but **only when the user is
  already near the bottom**, or it yanks the view while they are reading an
  earlier answer.

Accessibility, cheap to add while you are in there: `role="log"` and
`aria-live="polite"` on the message list, real `aria-label`s on the icon
buttons, `Enter` to send with `Shift+Enter` for a newline, `Esc` to close,
visible focus rings, and `prefers-reduced-motion` honoured for the dots.

### 7. Latency and garbled tokens (INFO — nothing to do)

**Latency.** During the audit the first token took ~15s from the test
environment; the client reports it is much faster for them. The causes are
environmental — hosting instances can spin down when idle, and network distance
varies. The typing indicator in item 3 covers the perceived-speed side, which is
the part you control. Measured from a browser against production after the
recent backend work: **1.3 seconds to first token**.

If you see a consistent multi-second first-token time for warm requests from a
normal user location, tell the backend team and include an `X-Request-ID`.

**Garbled token.** One reply printed `MC手数` (stray non-Latin characters) — a
rare generation quirk from the model tier. Not a frontend issue, not an encoding
bug on your side, and nothing to handle in the widget.

---

## Complete reference implementation

React + TypeScript, no dependencies beyond `react-markdown`. Adapt the styling
to your design system — the logic is the part that matters.

> These two files are a direct translation of the vanilla implementation running
> at `/widget.js`, which is the version that has been tested end-to-end against
> production. If something here does not behave, compare against that file.

### `useChat.ts`

```ts
import { useCallback, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_CHAT_API ??
  "https://ct-chatbot-production.up.railway.app";
const SESSION_KEY = "ct-chat-session";

export type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  pending?: boolean;      // show the typing indicator
  failed?: boolean;
  answerId?: string;      // present once `done` arrives; enables the thumbs
};

function readSession(): string | null {
  try { return sessionStorage.getItem(SESSION_KEY); } catch { return null; }
}

function writeSession(id: string) {
  try { sessionStorage.setItem(SESSION_KEY, id); } catch { /* private mode */ }
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const counter = useRef(0);

  const nextId = () => `m${counter.current++}`;

  const patch = (id: string, changes: Partial<Message>) =>
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...changes } : m))
    );

  const send = useCallback(async (raw: string) => {
    const message = raw.trim();
    // Client-side guard for the two things that would 422.
    if (!message || message.length > 2000 || busy) return;

    setBusy(true);
    setSuggestions([]);                       // never leave a stale chip row

    const pendingId = nextId();
    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: "user", text: message },
      { id: pendingId, role: "assistant", text: "", pending: true },
    ]);

    let streamed = "";

    try {
      const response = await fetch(`${API}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: readSession() }),
      });

      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let done: any = null;

      while (true) {
        const step = await reader.read();
        if (step.done) break;
        buffer += decoder.decode(step.value, { stream: true });

        // A frame can arrive split across reads; keep the remainder buffered.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;

          let event: any;
          try { event = JSON.parse(line.slice(5).trim()); } catch { continue; }

          if (event.type === "session") {
            writeSession(event.session_id);
          } else if (event.type === "delta") {
            streamed += event.text;
            patch(pendingId, { text: streamed, pending: false });
          } else if (event.type === "done") {
            done = event;
          } else if (event.type === "error") {
            // The status was already committed to 200 before this arrived.
            throw new Error(event.error);
          }
        }
      }

      if (done) {
        // `done.answer` is authoritative. The guards can replace an answer
        // wholesale, so what streamed is provisional and must be overwritten.
        patch(pendingId, {
          text: done.answer,
          pending: false,
          answerId: done.answer_id,
        });
        setSuggestions(done.suggestions ?? []);
      } else if (streamed) {
        patch(pendingId, { text: streamed, pending: false });
      } else {
        throw new Error("empty response");
      }
    } catch {
      patch(pendingId, {
        pending: false,
        failed: true,
        text:
          "Sorry — I couldn't reach our team just now. Please try again, or " +
          "call 1-800-889-0232.",
      });
    } finally {
      setBusy(false);
    }
  }, [busy]);

  const rate = useCallback((answerId: string, verdict: "up" | "down") => {
    // Fire and forget: a lost rating is not worth interrupting the conversation.
    fetch(`${API}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer_id: answerId, verdict }),
    }).catch(() => {});
  }, []);

  return { messages, suggestions, busy, send, rate };
}
```

### `ChatWidget.tsx`

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import { useChat, type Message } from "./useChat";

const STARTERS = [
  "What services do you offer?",
  "How does the program work?",
  "What happens during a free consultation?",
];

function Bubble({ message, onRate }: {
  message: Message;
  onRate: (verdict: "up" | "down") => void;
}) {
  const [rated, setRated] = useState(false);

  if (message.role === "user") {
    return (
      <div className="self-end max-w-[86%] rounded-2xl rounded-br bg-blue-600 px-3 py-2 text-white">
        {/* Plain text, never Markdown: this is untrusted input. */}
        {message.text}
      </div>
    );
  }

  return (
    <div className="self-start max-w-[86%]">
      <div className="mb-1 ml-0.5 text-[11.5px] text-slate-500">
        Corporate Turnaround
      </div>
      <div className="rounded-2xl rounded-bl border border-slate-200 bg-white px-3 py-2">
        {message.pending ? (
          <span className="flex gap-1 py-1" aria-label="Assistant is typing">
            {[0, 1, 2].map((i) => (
              <i
                key={i}
                className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400 motion-reduce:animate-none"
                style={{ animationDelay: `${i * 0.18}s` }}
              />
            ))}
          </span>
        ) : (
          <div className="prose prose-sm max-w-none prose-p:my-0 prose-p:mb-2 prose-ul:my-1.5">
            <ReactMarkdown>{message.text}</ReactMarkdown>
          </div>
        )}
      </div>

      {message.answerId && (
        <div className="mt-1.5 ml-0.5 flex items-center gap-1.5">
          {rated ? (
            <span className="text-[11.5px] text-slate-500">
              Thanks — that helps us improve.
            </span>
          ) : (
            (["up", "down"] as const).map((verdict) => (
              <button
                key={verdict}
                type="button"
                aria-label={verdict === "up" ? "This answer helped" : "This answer missed"}
                onClick={() => { onRate(verdict); setRated(true); }}
                className="rounded-md border border-slate-200 px-2 py-0.5 text-sm hover:bg-white"
              >
                {verdict === "up" ? "👍" : "👎"}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default function ChatWidget() {
  const { messages, suggestions, busy, send, rate } = useChat();
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [mounted, setMounted] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => setMounted(true), []);   // portals need the DOM

  // Follow the stream only if the user is already at the bottom, so it never
  // yanks the view while they are reading an earlier answer.
  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 120) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  if (!mounted) return null;

  const submit = () => { send(draft); setDraft(""); };
  const chips = messages.length ? suggestions : STARTERS;

  return createPortal(
    // fixed + high z-index + pointer-events-none root: this is fix #2.
    <div className="pointer-events-none fixed inset-0 z-[9999]">
      {!open && (
        <button
          onClick={() => setOpen(true)}
          aria-label="Open chat"
          className="pointer-events-auto fixed bottom-5 right-5 h-14 w-14 rounded-full bg-blue-600 text-white shadow-lg"
        >
          💬
        </button>
      )}

      {open && (
        <section
          aria-label="Chat with Corporate Turnaround"
          className="pointer-events-auto fixed bottom-5 right-5 flex w-[min(400px,calc(100vw-2rem))] flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
          style={{ height: "min(640px, calc(100vh - 2.5rem))" }}
          onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}
        >
          <header className="flex items-center gap-3 bg-blue-600 px-4 py-3 text-white">
            <div className="grid h-9 w-9 place-items-center rounded-full bg-white/20 text-xs font-semibold">
              CT
            </div>
            <div className="flex-1">
              <b className="block text-[15px]">Corporate Turnaround</b>
              <span className="block text-xs opacity-85">Business debt help</span>
            </div>
            <button onClick={() => setOpen(false)} aria-label="Close chat" className="text-xl">
              ×
            </button>
          </header>

          <div
            ref={logRef}
            role="log"
            aria-live="polite"
            className="flex flex-1 flex-col gap-3 overflow-y-auto bg-slate-50 p-4"
          >
            <div className="self-start max-w-[86%] rounded-2xl rounded-bl border border-slate-200 bg-white px-3 py-2">
              Hi — I can help with business debt, MCAs, creditor calls and more.
              What&apos;s going on?
            </div>
            {messages.map((m) => (
              <Bubble
                key={m.id}
                message={m}
                onRate={(verdict) => m.answerId && rate(m.answerId, verdict)}
              />
            ))}
          </div>

          {chips.length > 0 && !busy && (
            <div className="flex flex-wrap gap-2 bg-slate-50 px-4 pb-3">
              {chips.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => send(question)}
                  className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-left text-[13px] hover:border-blue-600 hover:text-blue-600"
                >
                  {question}
                </button>
              ))}
            </div>
          )}

          <form
            onSubmit={(e) => { e.preventDefault(); submit(); }}
            className="flex gap-2 border-t border-slate-200 p-3"
          >
            <label className="sr-only" htmlFor="ct-input">Your message</label>
            <textarea
              id="ct-input"
              rows={1}
              maxLength={2000}
              value={draft}
              disabled={busy}
              placeholder="Ask about your business debt..."
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
              }}
              className="max-h-28 flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2"
            />
            <button
              type="submit"
              disabled={busy || !draft.trim()}
              aria-label="Send message"
              className="w-11 rounded-lg bg-blue-600 text-white disabled:opacity-50"
            >
              ➤
            </button>
          </form>
        </section>
      )}
    </div>,
    document.body
  );
}
```

---

## Verification checklist

Run these against the real API before calling it done.

**Markdown (item 1)**

- [ ] Ask *"tell me all the services you provide, not just MCAs"*. The reply
      must render as a real `<ul>` with ~8 `<li>` — inspect the DOM, don't just
      look at it. No literal `*` or `-` characters anywhere.
- [ ] Paste `<img src=x onerror=alert(1)>` as a user message. It must appear as
      visible text in your own bubble; no dialog, no element created.

**Stacking (item 2)**

- [ ] With the chat open, run the `elementFromPoint` check from item 2 at three
      scroll positions: hero, mid-page, footer. It must resolve to the widget
      every time.
- [ ] With the chat **closed**, links and buttons behind the launcher still work
      (this is what `pointer-events-none` on the root protects).

**Streaming (item 3)**

- [ ] The typing indicator appears within one frame of pressing send — never a
      blank pane.
- [ ] Text visibly types out rather than appearing all at once.
- [ ] Ask something off-topic, e.g. *"how do I fix a leaking kitchen tap"*. It
      must decline rather than answer, and `suggestions` comes back `[]`.
- [ ] **Prove you render `done.answer`, not the deltas.** Temporarily log both
      and compare:
      ```ts
      console.log({ streamed, final: done.answer, same: streamed === done.answer });
      ```
      They are identical on most turns, which is exactly why this bug survives
      testing. What matters is which one you *render* — the guards rewrite the
      answer only on the turns that would otherwise be non-compliant, and those
      are the turns you cannot afford to get wrong. Confirm by reading your
      code, not by watching the screen.
- [ ] Kill your network mid-answer. You get the friendly error, not a stuck
      spinner.

**Chips (item 4)**

- [ ] Chips appear under the answer, and tapping one sends it and produces a
      real answer rather than a refusal.
- [ ] Chips clear the moment a new message is sent.

**Feedback (item 5)**

- [ ] 👍 returns `200` in the network tab and the buttons become a thank-you.
- [ ] It cannot be double-submitted.

**Sessions**

- [ ] Ask *"do you help with merchant cash advances?"*, then *"how do I get out
      of it?"*. The second answer must be about MCAs. If it is generic, the
      `session_id` is not being sent.
- [ ] Open a second tab. It starts a fresh conversation.

---

## What to ask the backend team for

**CORS allowlist.** Send the exact origins, scheme included:

- your production domain (e.g. `https://www.corporateturnaround.com`)
- any staging domain
- `http://localhost:3000` if you want to develop against the live API

Vercel preview deployments get a new hostname per deploy and **cannot** be
allowlisted one by one — develop against `localhost` or a fixed staging domain.

**When reporting a bug**, include the `X-Request-ID` response header. It locates
the exact request in the backend logs.

**Ask for a re-check if** you see consistent multi-second first-token times on
warm requests from a normal location, answers that contradict the site, or the
garbled-token issue becoming frequent rather than rare.
