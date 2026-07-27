# Chat widget

Reference implementation of the website chat widget. One file, no build step,
no CDN, no dependencies. It is both a working widget you can ship as-is and the
spec for anyone porting the behaviour into an existing React app.

It closes every frontend item in [`../FRONTEND_FIXES.md`](../FRONTEND_FIXES.md).

## Embed

Served by the API itself, so nothing needs hosting:

```html
<script src="https://ct-chatbot-production.up.railway.app/widget.js"
        data-api="https://ct-chatbot-production.up.railway.app"
        data-accent="#0b5cff"></script>
```

The embedding site's origin must be listed in `CORS_ALLOWED_ORIGINS` on the API.
A `<script>` tag itself is not subject to CORS, but the chat requests are.

| Attribute | Default | Purpose |
|---|---|---|
| `data-api` | `http://localhost:8000` | API base URL |
| `data-title` | `Corporate Turnaround` | Header name and message label |
| `data-subtitle` | `Business debt help` | Header second line |
| `data-accent` | `#0b5cff` | Brand colour |
| `data-greeting` | (see source) | First message, shown before any request |

## What it does

- **Markdown** for assistant replies (bullets, ordered lists, paragraphs, bold)
  and **never** for user messages. Everything is HTML-escaped before any markup
  is applied, and raw HTML is never honoured.
- **Streams** `delta` tokens into the bubble, then replaces the whole bubble
  with `done.answer`. The deltas are provisional — the server's guards can
  replace an answer wholesale, so rendering the accumulated tokens as final
  would leave a non-compliant answer on screen.
- **Typing indicator** the instant you hit send. The pane is never blank.
- **Follow-up chips** from `done.suggestions`. Every chip is a question the KB
  has a hand-authored answer for, so tapping one can't produce a refusal.
- **Thumbs up/down** posts to `/api/feedback` with the opaque `answer_id`. The
  server already knows the question and which records produced the answer.
- **Fixed positioning** in its own stacking context at `z-index: 2147483000`,
  root `pointer-events: none` with interactive children `auto`. This is the fix
  for the panel rendering underneath full-bleed hero images.
- **Shadow DOM**, so host page CSS cannot reach in and widget CSS cannot leak.
- Phone numbers shown as plain text — deliberately not `tel:` links.
- `Enter` to send, `Shift+Enter` for a newline, `Esc` to close, `aria-live`
  log, visible focus rings, and `prefers-reduced-motion` respected.
- Auto-scroll only when you're already at the bottom, so it never yanks the
  view while you're reading an earlier answer.
- Session id in `sessionStorage` — one conversation per tab, per visit.

## Running it locally

```bash
# terminal 1 — API, allowing the harness origin
CORS_ALLOWED_ORIGINS=http://localhost:3000 uvicorn api.main:app --port 8000

# terminal 2 — the harness page
cd widget && python -m http.server 3000
# open http://localhost:3000/demo.html
```

`demo.html` deliberately includes a full-bleed hero image, because that is the
layout the z-index bug needed to reproduce.

## Tests

`test-markdown.js` checks the renderer against the shipped file (not a copy).
Load `demo.html`, then in the console:

```js
await fetch('./test-markdown.js').then(r => r.text()).then(eval)
// -> {passed: 10, failed: 0, failures: []}
```

It covers the shape that actually broke — a lead-in line immediately followed
by bullets with no blank line between — plus HTML-escaping.
