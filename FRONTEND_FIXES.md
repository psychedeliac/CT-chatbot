# Chat Widget — Frontend Work List

**For the frontend developer who owns the widget on `corpo-nine.vercel.app`.**

Everything in this document is work in **your** codebase. The backend side is
done and deployed — none of it needs waiting on.

There is now a complete working implementation of every item here in
[`widget/ct-chat-widget.js`](widget/ct-chat-widget.js) in the backend repo. It
is plain JS with no dependencies and no build step, written to be read: port
from it rather than from these descriptions where they disagree. You can run it
against production right now to see the target behaviour:

```html
<script src="https://ct-chatbot-production.up.railway.app/widget.js"
        data-api="https://ct-chatbot-production.up.railway.app"></script>
```

Items 1–3 came from the live UX audit (2026-07-27) and are what the client
noticed. Items 6–7 are new API capabilities the widget does not use yet.

| # | Priority | What to do |
|---|----------|-----------|
| 1 | **HIGH** | Render assistant replies as Markdown; keep user messages plain text |
| 2 | **HIGH** | `position: fixed` in a high-z-index root so the panel can't hide under page images |
| 3 | **HIGH** | Typing indicator on send, stream `delta` tokens, swap to `done.answer` |
| 4 | MEDIUM | Follow-up suggestion chips from `done.suggestions` |
| 5 | MEDIUM | Thumbs up/down posting to `/api/feedback` |
| 6 | LOW | Bubble styling, avatar, auto-scroll |
| 7 | INFO | Latency and the occasional garbled token — nothing to do |

---

## 1. Render the assistant reply as Markdown (HIGH)

**Symptom.** Bullet lists show literal asterisks. A real reply renders as:

```
We help small businesses resolve debt...

* Business debt negotiation and settlement
* Merchant cash advance relief
* Creditor harassment protection
```

The `*` and the paragraph breaks are **Markdown** — the backend intentionally
emits it. The widget is printing it as raw text, so it reads like a debug
console instead of a formatted answer.

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

Notes:

- Render **only assistant** messages as Markdown. The **user's** message must
  stay plain text — it's untrusted input, and rendering it as markup is how a
  user pastes HTML into your page.
- `react-markdown` does **not** render raw HTML by default. Keep it that way —
  no `rehype-raw`. The backend only emits bullets, paragraphs and occasional
  bold, and the text is model-generated, so it can be steered by what a user
  types.
- Constrain Tailwind Typography (`prose-sm`, `max-w-none`) for the narrow
  chat column.
- Render `done.answer` — the authoritative final text (see #3). Running each
  partial through the same renderer is fine; half-finished Markdown degrades
  gracefully.
- **Watch the list-grouping edge case.** The model very often emits a lead-in
  line immediately followed by its bullets with *no* blank line between:

  ```
  Here are the services we provide:
  - Business debt negotiation
  - MCA debt relief
  ```

  `react-markdown` handles this correctly. A hand-rolled renderer usually does
  not — the naive version splits on blank lines and needs the whole block to be
  bullets, which turns the above into one run-on paragraph. This was a real bug
  in the reference widget before it was fixed; if you write your own renderer,
  test this exact shape.
- **Phone numbers stay plain text.** Do not turn them into `tel:` links.

---

## 2. Widget can render *underneath* page content (HIGH)

**Symptom.** With the chat open and the page scrolled to the hero/CTA image
section, the message input is **not clickable** — clicks land on the background
image instead. Reproduced with `document.elementFromPoint(inputCenter)`, which
returned the hero `<img class="object-cover">`, not the input.

**Root cause.** The panel is positioned `absolute inset-4` and shares a stacking
context with page content, so full-bleed images (`object-cover`, their own
stacking context) can paint over it depending on scroll position.

**Fix.** Pin the widget to the viewport in its own top-level stacking context:

```tsx
<div className="fixed inset-0 z-[9999] pointer-events-none">
  <div className="pointer-events-auto fixed bottom-4 right-4 ...">
    {/* panel */}
  </div>
</div>
```

Checklist:

- `fixed`, not `absolute`, for the widget root and panel.
- High `z-index` above every page section, header and image.
- Root `pointer-events-none`, interactive children `pointer-events-auto`, so
  the widget never eats clicks on the page behind it when closed.
- Consider rendering the whole widget into a **portal at `document.body`**, or
  a Shadow DOM — the reference widget uses a Shadow DOM, which also stops page
  CSS from reaching into the panel.

**How to verify** (this is the check that proves it, at several scroll
positions with the chat open):

```js
const r = inputEl.getBoundingClientRect();
document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
// must resolve to the widget, never a page element
```

---

## 3. Stream tokens live + show an instant loading state (HIGH)

**Symptom.** After sending, the panel sits blank until the entire answer appears
at once. Even when the backend is fast, a blank pane reads as "broken."

**Cause.** The widget waits for the full response before rendering. The API
already streams — `/api/chat/stream` emits `delta` events token by token — so
this is purely a render-side fix.

**Fix.**

1. The instant the user sends, append an assistant bubble containing a **typing
   indicator** (three animated dots). Never leave the pane empty.
2. Consume the SSE stream and append each `delta.text` to that bubble as it
   arrives.
3. On `done`, **replace** the accumulated text with `done.answer` and drop the
   indicator.
4. On `error`, replace the bubble with the `error` string.

Event order: `session` → `delta …` (many) → `done` → *(or)* `error`.

> **The `done` replacement is not optional.** The server's guards can swap an
> answer wholesale — an ungrounded reply becomes a compliance refusal. If you
> render the accumulated deltas as final, a non-compliant answer finishes typing
> and stays on screen. Always render `done.answer` over whatever you streamed.

Store `session_id` from the `session` event and send it with every subsequent
message, or each turn starts a new conversation with no history.

---

## 4. Follow-up suggestion chips (MEDIUM — new)

The `done` event now carries `suggestions`: up to three follow-up questions,
already filtered so they never repeat what was just answered.

```json
{ "type": "done", "answer": "...", "suggestions": ["What happens during a free consultation?", "..."], "answer_id": "cbJk89426fUAz5uk" }
```

**What to build.** Render them as tappable chips under the latest assistant
message. Tapping one sends it as the user's next message and clears the row.
Clear the chips as soon as the user sends anything, so a stale row never sits
under a newer answer.

These are drawn from the knowledge base, not invented by the model, so every
chip has a hand-authored answer behind it — tapping one cannot produce a
refusal. Show them as-is; don't rewrite the text.

`suggestions` is an empty array when the assistant declines a question. Render
nothing in that case.

---

## 5. Answer feedback — thumbs up/down (MEDIUM — new)

Every `done` event carries an opaque `answer_id`. A thumbs control on assistant
messages posts it back:

```ts
await fetch(`${API}/api/feedback`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ answer_id, verdict: "up" | "down", comment: "" }),
});
```

- `comment` is optional (max 1000 chars) if you want a "tell us more" box on
  thumbs-down.
- Fire and forget — a rating that fails to record is not worth interrupting the
  conversation over. Replace the buttons with a short thank-you on click.
- Don't echo the question or answer text back; the server already has both,
  plus the knowledge-base records that produced the answer. That's what makes a
  thumbs-down actionable.
- `404` means the id has aged out (they live one hour). Nothing to show the
  user.

---

## 6. Message styling / "feel" (LOW, polish)

Current bubbles read as an undifferentiated wall of text. Cheap wins:

- Distinct bubble styling for **user** (right-aligned, brand fill) vs
  **assistant** (left-aligned, light surface).
- A small assistant avatar/label to anchor the conversation.
- Comfortable line-height and spacing between messages.
- Auto-scroll to the newest message as it streams — but **only when the user is
  already near the bottom**, or it yanks the view while they're reading an
  earlier answer.

---

## 7. Latency and garbled tokens (INFO — nothing to do)

**Latency.** During the audit the first token took ~15s from the test
environment; the client reports it is much faster for them. Causes are
environmental — Railway instances can spin down when idle, and network distance
varies. The typing indicator in #3 covers the perceived-speed side, which is the
part you control. Measured from a browser against production after the recent
backend work: **1.3s to first token**.

If you see a consistent multi-second first-token time for warm requests from a
normal user location, tell the backend team — that would be a server concern.

**Garbled token.** One reply printed `MC手数` (stray non-Latin characters) — a
rare generation quirk from the model tier. Not a frontend issue and not an
encoding bug on your side. If it becomes frequent, the backend team can switch
model tiers.

---

## API reference

Base URL: `https://ct-chatbot-production.up.railway.app`

Your origin must be in the backend's `CORS_ALLOWED_ORIGINS` — ask the backend
team when you add a new deployment domain. `corpo-nine.vercel.app` is already
allowed.

| Endpoint | Purpose |
|---|---|
| `POST /api/chat/stream` | SSE stream. `{message, session_id?}` → `session` / `delta` / `done` / `error` events |
| `POST /api/chat` | Non-streaming equivalent. Returns `{session_id, answer, cached, suggestions, answer_id}` |
| `POST /api/feedback` | `{answer_id, verdict, comment?}` → `{ok: true}` |
| `GET /health` | `{status, active_sessions}` |
| `GET /widget.js` | The reference widget, if you want to embed it directly |

Request limits: 2000 characters per message; rate limited per IP (10/minute,
100/hour) — a `429` means slow down, and the copy for it should say so rather
than reading as an error.
