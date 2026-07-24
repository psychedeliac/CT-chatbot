# Corporate Turnaround Chatbot — Frontend Integration Guide

This document is everything you need to connect the website chat widget to the
live chatbot backend. No backend changes are required — just call the API.

- **Live API base URL:** `https://ct-chatbot-production.up.railway.app`
- **Transport:** HTTPS + Server-Sent Events (SSE) for streaming
- **Auth:** none (public endpoint; protected by CORS + rate limiting)

> The widget currently shows a placeholder reply ("A member of our team will
> follow up shortly"). That's a stub — replace it with the real API calls below.

---

## 1. Quick sanity check (paste into a terminal)

```bash
# Health — should return {"status":"ok",...}
curl https://ct-chatbot-production.up.railway.app/health

# One chat turn (non-streaming)
curl -X POST https://ct-chatbot-production.up.railway.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How much does your program cost?"}'
```

If those work, you're ready to integrate.

---

## 2. Endpoints

| Method | Path                | Use for                                        |
|--------|---------------------|------------------------------------------------|
| POST   | `/api/chat/stream`  | **Primary** — streams the reply token-by-token (loading bubble + live typing) |
| POST   | `/api/chat`         | Fallback — returns the full reply in one JSON response |
| GET    | `/health`           | Uptime check (optional) |

Both POST endpoints take the **same request body**:

```jsonc
{
  "message": "string, required, 1–2000 chars",
  "session_id": "string, optional — omit on the first message"
}
```

**Session handling (important):** the server returns a `session_id`. Store it in
memory (e.g. React state or a module variable) and send it back on every
subsequent message so the conversation keeps context. Do **not** persist it to
localStorage long-term — a fresh page load starting a new conversation is fine.

---

## 3. Streaming response format (`/api/chat/stream`)

The response is an SSE stream (`Content-Type: text/event-stream`). Each event is
a line beginning with `data: ` followed by a JSON object. Events arrive in this
order:

```
data: {"type": "session", "session_id": "WSTWERG1rHuuOW3H0W8eeKEbTTf6WxOf"}

data: {"type": "delta", "text": "Yes,"}

data: {"type": "delta", "text": " we regularly help business owners"}

data: {"type": "delta", "text": " resolve unpaid business credit card debt."}

data: {"type": "done", "answer": "Yes, we regularly help business owners resolve unpaid business credit card debt.", "cached": false}
```

| Event `type` | What to do |
|--------------|------------|
| `session`    | Save `session_id` for the next turn. |
| `delta`      | Append `text` to the assistant bubble as it streams. |
| `done`       | **Replace** the bubble's text with `answer` (see rule below), then stop. |
| `error`      | Show `error` as the assistant message and stop. |

### ⚠️ Critical rule: always render `done.answer` over the streamed deltas

The streamed `delta` tokens are **provisional**. Safety guardrails (grounding,
crisis handling, PII scrubbing) can rewrite the final answer after streaming
finishes — for example, an off-topic or unsafe reply gets swapped for a safe
one. So when `done` arrives, overwrite whatever you accumulated from deltas with
`answer`. If you skip this, a user could briefly see text that the backend then
corrects.

---

## 4. Drop-in client (vanilla JS / TypeScript)

Works in any framework (React, Vue, Svelte, plain JS). Uses `fetch` streaming —
**not** `EventSource`, because `EventSource` can't send a POST body.

```js
// ctChat.js
const API_BASE = "https://ct-chatbot-production.up.railway.app";

let sessionId = null; // remembered across turns in this tab

/**
 * Send a message and stream the reply.
 * @param {string} message
 * @param {{onDelta:(t:string)=>void, onDone:(answer:string)=>void, onError:(msg:string)=>void}} cb
 */
export async function streamChat(message, { onDelta, onDone, onError }) {
  try {
    const res = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });

    if (res.status === 429) return onError("You're sending messages a bit fast — please wait a moment and try again.");
    if (res.status === 503) return onError("We're handling a lot of conversations right now — please retry shortly.");
    if (!res.ok)            return onError("Sorry — something went wrong. Please try again, or call 1-800-889-0232.");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line.
      const events = buffer.split("\n\n");
      buffer = events.pop(); // keep the incomplete trailing chunk

      for (const evt of events) {
        const dataLine = evt.split("\n").find((l) => l.startsWith("data: "));
        if (!dataLine) continue;
        const data = JSON.parse(dataLine.slice(6));

        if (data.type === "session")    sessionId = data.session_id;
        else if (data.type === "delta") onDelta(data.text);
        else if (data.type === "done")  onDone(data.answer);
        else if (data.type === "error") onError(data.error);
      }
    }
  } catch {
    onError("Sorry — something went wrong. Please try again, or call 1-800-889-0232.");
  }
}
```

---

## 5. React example — with loading bubble + live streaming

```jsx
import { useState } from "react";
import { streamChat } from "./ctChat";

export function ChatWidget() {
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hi there! I'm here to help answer questions about our debt relief programs. How can I help you today?" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false); // controls the typing bubble

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text }]);
    setLoading(true);           // 👈 show typing bubble immediately

    let assistant = "";
    let started = false;

    await streamChat(text, {
      onDelta: (chunk) => {
        if (!started) { started = true; setLoading(false); } // hide bubble on 1st token
        assistant += chunk;
        setMessages((m) => replaceLastAssistant(m, assistant, started));
      },
      onDone: (finalAnswer) => {  // authoritative, guard-checked text
        setLoading(false);
        setMessages((m) => replaceLastAssistant(m, finalAnswer, true));
      },
      onError: (msg) => {
        setLoading(false);
        setMessages((m) => [...m, { role: "assistant", text: msg }]);
      },
    });
  }

  return (
    <div className="chat">
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>{m.text}</div>
        ))}
        {loading && <div className="bubble assistant typing"><span>•</span><span>•</span><span>•</span></div>}
      </div>
      <div className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type your message…"
        />
        <button onClick={handleSend} disabled={loading}>➤</button>
      </div>
    </div>
  );
}

// Append the streaming assistant text, or start a new assistant bubble on the
// first token of a turn.
function replaceLastAssistant(messages, text, streaming) {
  const last = messages[messages.length - 1];
  if (streaming && last?.role === "assistant") {
    return [...messages.slice(0, -1), { role: "assistant", text }];
  }
  return [...messages, { role: "assistant", text }];
}
```

The three-dot typing bubble (`.typing`) shows between send and the first token;
after that, text streams in live and is finalized on `done`.

---

## 6. Non-streaming fallback (`/api/chat`)

If you don't want streaming, this returns the whole reply at once. Show a
loading bubble while the request is in flight.

```js
async function sendMessage(message, sessionId) {
  const res = await fetch("https://ct-chatbot-production.up.railway.app/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  const data = await res.json();
  // { session_id, answer, cached }
  return data;
}
```

---

## 7. Error & status handling

| HTTP status | Meaning | Suggested message to the user |
|-------------|---------|-------------------------------|
| 200 | OK | — |
| 422 | Message empty or >2000 chars | "Please enter a message (up to 2000 characters)." |
| 429 | Rate limited | "You're sending messages a bit fast — please wait a moment." |
| 503 | Server at capacity or LLM degraded | "We're busy right now — please retry shortly." |
| 504 | Turn timed out | "That took longer than expected. Please try again." |
| 500 | Unexpected error | "Something went wrong — please try again, or call 1-800-889-0232." |

For the **streaming** endpoint, mid-stream failures arrive as a `data:` event
with `{"type":"error","error":"..."}` — render that text and stop.

---

## 8. Things to know

- **CORS:** the API only accepts browser calls from allow-listed origins.
  `https://corpo-nine.vercel.app` is already allowed. If you deploy the site to
  a different domain (e.g. a custom domain), tell the backend owner to add it to
  `CORS_ALLOWED_ORIGINS` — otherwise the browser will block every request.
- **Message length:** max **2000 characters**. Enforce it client-side too
  (disable send / show a counter) for a nicer UX.
- **Rate limits:** ~10 messages/minute per visitor. Handle `429` gracefully
  (disable the send button briefly, show the "sending too fast" message).
- **First response can be a touch slower** (a few seconds) than later ones;
  common questions are cached and come back almost instantly.
- **Don't** display the `session` or `context` internals to the user — only
  `delta` text (while streaming) and the final `done.answer`.
- **Greeting:** the widget's opening "Hi there!" message can stay hardcoded on
  the frontend (as it is now) — you don't need to call the API just to greet.
  Only call the API when the user actually sends a message.

---

## 9. Contract summary (TL;DR for a dev in a hurry)

1. `POST https://ct-chatbot-production.up.railway.app/api/chat/stream` with
   `{ "message": "...", "session_id": <null first time, then reuse> }`.
2. Read the SSE stream: save `session` → append `delta` text → on `done`,
   replace with `answer`.
3. Show a typing bubble from send until the first `delta`.
4. Handle `429` / `503` / errors with friendly messages.

That's it.
