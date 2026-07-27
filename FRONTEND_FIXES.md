# Chat Widget — Frontend Fix List (UX Audit 2026-07-27)

Punch-list of issues found testing the live widget on
`https://corpo-nine.vercel.app/`. **All items here are frontend-side** — the
backend (prompt behaviour, phone-number CTA, greeting logic) has already been
fixed and deployed. For wiring/API details see `FRONTEND_INTEGRATION.md`; this
doc only covers what to change in the widget.

Priority order: **1 and 2 are the ones the client actually noticed.**

---

## 1. Render the assistant reply as Markdown (HIGH)

**Symptom.** Bullet lists show literal asterisks. A real reply renders as:

```
We help small businesses resolve debt...

* Business debt negotiation and settlement
* Merchant cash advance relief
* Creditor harassment protection
```

The `*` (and paragraph breaks) are **Markdown** — the backend intentionally
emits Markdown (`-`/`*` bullets, blank-line paragraphs). The widget is printing
it as raw text, so it reads like a debug console instead of a formatted answer.

**Fix.** Render the assistant message body through a Markdown renderer instead
of dropping it into a `<p>`/text node.

```bash
npm i react-markdown
```

```tsx
import ReactMarkdown from "react-markdown";

// assistant bubble
<div className="prose prose-sm max-w-none">
  <ReactMarkdown>{message.text}</ReactMarkdown>
</div>
```

Notes:
- Render **only assistant** messages as Markdown. Render the **user's** message
  as plain text (never Markdown — it's untrusted input; rendering it invites
  injection and lets a user paste raw HTML/links into their own bubble).
- `react-markdown` does **not** render raw HTML by default — good, keep it that
  way (no `rehype-raw`). The backend only ever emits bullets, paragraphs, and
  the occasional bold; no HTML is needed.
- If you use Tailwind Typography (`prose`), constrain it (`prose-sm`,
  `max-w-none`) so it fits the narrow chat column.
- **Which field to render:** render `done.answer` (the authoritative final
  text). If you also render `delta` tokens live (see #3), it's fine to run each
  partial through the same renderer — half-finished Markdown degrades
  gracefully.

---

## 2. Widget can render *underneath* page content (HIGH)

**Symptom.** With the chat open and the page scrolled to the hero/CTA image
section, the message input is **not clickable** — clicks land on the background
image instead. Reproduced via `document.elementFromPoint(inputCenter)`, which
returned the hero `<img class="object-cover">`, not the input.

**Root cause.** The panel is positioned `absolute inset-4` and shares a stacking
context with page content, so full-bleed images (`object-cover`, their own
stacking context) can paint over it depending on scroll position.

**Fix.** Pin the widget to the viewport in its own top-level stacking context:

```tsx
// widget root wrapper
<div className="fixed inset-0 z-[9999] pointer-events-none">
  {/* launcher button + panel are pointer-events-auto */}
  <div className="pointer-events-auto fixed bottom-4 right-4 ...">
    {/* panel */}
  </div>
</div>
```

Checklist:
- Use `fixed`, not `absolute`, for the widget root and panel.
- Give the root a high `z-index` (e.g. `z-[9999]`) above every page section,
  header, and image.
- Keep the root `pointer-events-none` and the interactive children
  `pointer-events-auto`, so the widget never eats clicks on the page behind it
  when closed/minimised.
- Re-test with the page scrolled to **each** section (hero, calculator,
  results, contact) with the chat open — the old bug was scroll-position
  dependent.

---

## 3. Stream tokens live + show an instant loading state (HIGH for *perceived* speed)

**Symptom.** After sending, the panel sits blank until the entire answer appears
at once. Even when the backend is fast, a blank pane reads as "broken."

**Cause.** The widget appears to wait for the full response before rendering.
The API already streams — `/api/chat/stream` emits `delta` events token-by-token
— so the fix is purely on the render side.

**Fix.**
1. The instant the user sends, append an assistant bubble containing a **typing
   indicator** (three animated dots). Never leave the pane empty.
2. Consume the SSE stream and append each `delta.text` to that bubble as it
   arrives, so the user sees the answer type out.
3. On `done`, **replace** the accumulated text with `done.answer` (it's the
   only guard-checked, authoritative version) and drop the typing indicator.
4. On `error`, replace the bubble with the `error.message` string.

Event order on the stream (see `FRONTEND_INTEGRATION.md` §3):
`session` → `delta …` (many) → `done` → *(or)* `error`.

> Even if backend latency is low for most users, always show the typing
> indicator on send — it's the single biggest perceived-speed win and costs
> nothing.

---

## 4. Latency — mostly environmental, not a frontend bug (INFO)

During the audit the first token took ~15s **from the test environment**. The
client reports it is much faster for them, which points to one of:
- **Cold start** — Railway hobby instances spin down when idle; the *first*
  request after a quiet period pays a startup penalty. Subsequent turns are
  fast.
- **Network distance** from the test machine to the Railway region.

There is nothing to fix in the widget for this. Two optional mitigations:
- Implement the typing indicator in #3 so any wait is visibly "working," not
  "hung."
- (Infra, optional) Keep the instance warm with a periodic `GET /health` ping,
  or move off the spin-down tier, if first-message latency is ever a complaint.

If you *do* see a consistent multi-second first-token time for warm requests
from a normal user location, tell the backend team — that would be a server
concern (retrieval + rerank), not frontend.

---

## 5. Message styling / "feel" (LOW, polish)

Current bubbles read as an undifferentiated wall of text. Cheap wins:
- Distinct bubble styling for **user** (right-aligned, brand fill) vs
  **assistant** (left-aligned, light surface).
- A small assistant avatar/label to anchor the conversation.
- Comfortable line-height and spacing between messages.
- Auto-scroll to the newest message as it streams.

These are optional but they're most of the difference between "feels like a real
assistant" and "feels like a form field."

---

## 6. Occasional garbled token (LOW, INFO — not frontend)

One reply printed `MC手数` (stray non-Latin characters) — a rare generation
quirk from the model tier the backend uses. It is **not** a frontend issue and
nothing to handle in the widget. Logged here only so it's not mistaken for an
encoding bug on your side. If it becomes frequent, the backend team can switch
model tiers.

---

## Summary for the frontend dev

| # | Priority | What to do |
|---|----------|-----------|
| 1 | HIGH | Render assistant replies as Markdown (`react-markdown`); keep user messages plain text |
| 2 | HIGH | Make the widget `position: fixed` in a high-z-index, pointer-events-none root so it can't hide under page images |
| 3 | HIGH | Show a typing indicator on send; stream `delta` tokens; swap to `done.answer` at the end |
| 4 | INFO | Latency is environmental (cold start / network); typing indicator covers it |
| 5 | LOW | User/assistant bubble styling, avatar, auto-scroll |
| 6 | INFO | Garbled-token quirk is backend/model, ignore on frontend |
