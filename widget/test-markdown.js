/*
 * widget/test-markdown.js — checks for the widget's Markdown renderer.
 *
 * Run it in the browser console on widget/demo.html (or paste it into
 * Playwright's evaluate) after the widget has loaded:
 *
 *   copy(await fetch('./test-markdown.js').then(r => r.text()))  // then paste
 *
 * It runs against the real ct-chat-widget.js, not a copy, so it cannot drift
 * from what actually ships. Returns {passed, failed, failures}.
 */
(function () {
  var render = document.querySelector("[data-ct-chat]").__renderMarkdown;
  var failures = [];

  function check(name, actual, expected) {
    if (actual !== expected) {
      failures.push(name + "\n  expected: " + expected + "\n  actual:   " + actual);
    }
  }

  // The shape the model actually emits most: a lead-in line, then bullets, with
  // no blank line between them. Rendering that as one run-on paragraph was the
  // bug this file exists to keep fixed.
  check(
    "lead-in line followed immediately by bullets",
    render("Here are the services:\n- Negotiation\n- MCA relief"),
    "<p>Here are the services:</p><ul><li>Negotiation</li><li>MCA relief</li></ul>"
  );

  check(
    "bullets separated by a blank line",
    render("Services:\n\n- One\n- Two"),
    "<p>Services:</p><ul><li>One</li><li>Two</li></ul>"
  );

  check("asterisk bullets", render("* One\n* Two"), "<ul><li>One</li><li>Two</li></ul>");

  check(
    "ordered list",
    render("1. First\n2. Second"),
    "<ol><li>First</li><li>Second</li></ol>"
  );

  check(
    "a list followed by a closing paragraph",
    render("- One\n- Two\nThat's the list."),
    // The apostrophe is entity-escaped -- everything is escaped before any
    // markup is applied, which is what keeps injected HTML inert.
    "<ul><li>One</li><li>Two</li></ul><p>That&#39;s the list.</p>"
  );

  check("bold", render("We **can** help"), "<p>We <strong>can</strong> help</p>");

  // Untrusted text must never become markup. The API is the only writer today,
  // but its output is model-generated and can be steered by a user's message.
  check(
    "html is escaped, not honoured",
    render("<img src=x onerror=alert(1)>"),
    "<p>&lt;img src=x onerror=alert(1)&gt;</p>"
  );

  check(
    "a script tag cannot be smuggled through a bullet",
    render("- <script>alert(1)</script>"),
    "<ul><li>&lt;script&gt;alert(1)&lt;/script&gt;</li></ul>"
  );

  // Phone numbers are shown, never linked. No tel: anywhere in the output.
  check(
    "the company line stays plain text",
    render("Call 1-800-889-0232 today"),
    "<p>Call 1-800-889-0232 today</p>"
  );

  check("empty input", render(""), "");

  return {
    passed: 10 - failures.length,
    failed: failures.length,
    failures: failures,
  };
})();
