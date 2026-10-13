"""Offline UI test: drives `app.py` through Streamlit's own AppTest harness.

This actually *executes* the Streamlit script (it is not a screenshot or an
import check) and walks the real user journey:

    render -> build index -> ask a question -> read the answer and its sources

As in `test_pipeline_offline.py`, the SentenceTransformer encoder is replaced
by a deterministic offline stub and the corpus is the synthetic fixture set, so
the test needs no network and no model download. It proves the *wiring* of the
app, not the quality of any answer.

Run with:
    python tests/test_app_offline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config as config  # noqa: E402
import src.embeddings as embeddings  # noqa: E402
from tests.test_pipeline_offline import StubEncoder  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"

_failures: List[str] = []
_checks = 0


def check(condition: bool, label: str) -> None:
    global _checks
    _checks += 1
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        _failures.append(label)


def main() -> int:
    # --- Patch the heavy dependency and the corpus location BEFORE app import.
    embeddings.EmbeddingModel = StubEncoder          # type: ignore[assignment]
    config.SAMPLE_PAPERS_DIR = config.TEST_FIXTURES_DIR

    if not list(config.TEST_FIXTURES_DIR.glob("*.pdf")):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import make_test_pdfs

        make_test_pdfs.main()

    from streamlit.testing.v1 import AppTest

    print("=" * 70)
    print("OFFLINE STREAMLIT APP TEST  (stub encoder, synthetic corpus)")
    print("=" * 70)

    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()

    print("\n[1] initial render")
    check(not app.exception, f"app renders without an exception ({app.exception})")
    check(
        any("Clinical Research RAG Assistant" in t.value for t in app.title),
        "title is displayed",
    )
    disclaimer_shown = any(
        "does not provide medical advice" in e.value for e in app.error
    )
    check(disclaimer_shown, "medical-advice disclaimer is displayed prominently")
    check(
        any("No index built yet" in w.value for w in app.warning),
        "empty state tells the user no index exists",
    )
    labels = [b.label for b in app.button]
    check("Build / Rebuild Index" in labels, "build button present")
    check("Ask Research Question" in labels, "ask button present")
    check(
        any("top-k" in s.label for s in app.slider), "top-k slider present"
    )

    print("\n[2] asking before an index exists")
    ask = next(b for b in app.button if b.label == "Ask Research Question")
    ask.click().run()
    check(not app.exception, "no crash when asking without an index")
    check(
        any("Build the index first" in w.value for w in app.warning),
        "user is told to build the index first",
    )

    print("\n[3] building the index")
    build = next(b for b in app.button if b.label == "Build / Rebuild Index")
    build.click().run()
    check(not app.exception, f"index build raises no exception ({app.exception})")
    metrics = {m.label: m.value for m in app.metric}
    check("Papers indexed" in metrics, f"papers metric shown: {metrics}")
    check(
        str(metrics.get("Papers indexed", "")) == "3",
        f"3 papers indexed (got {metrics.get('Papers indexed')})",
    )
    check(int(str(metrics.get("Chunks", 0))) > 0, "chunk count is shown and non-zero")

    print("\n[4] empty question")
    ask = next(b for b in app.button if b.label == "Ask Research Question")
    ask.click().run()
    check(
        any("type a question" in w.value.lower() for w in app.warning),
        "empty question is rejected with a message, not a crash",
    )

    print("\n[5] asking a real question")
    app.text_input[0].set_value("Which biomarkers were examined?").run()
    ask = next(b for b in app.button if b.label == "Ask Research Question")
    ask.click().run()
    check(not app.exception, f"question answered without an exception ({app.exception})")

    body = "\n".join(m.value for m in app.markdown)
    headings = "\n".join(h.value for h in app.subheader)
    code_blocks = "\n".join(c.value for c in app.code)

    check(
        "Evidence-Grounded Answer" in headings,
        "answer section is rendered",
    )
    check("Retrieval-only mode" in body, "retrieval-only mode is declared to the user")
    check("[S1]" in body or "[S1]" in code_blocks, "source markers are shown")
    check("Sources" in body, "Sources heading is rendered")
    check(
        "page" in code_blocks.lower(),
        f"sources block cites pages: {code_blocks.splitlines()[:1]}",
    )
    check(len(app.expander) >= 1, "retrieved passages are shown in expanders")

    expander_labels = " ".join(e.label for e in app.expander)
    check("similarity" in expander_labels, "each source shows a similarity score")
    check(".pdf" in expander_labels, "each source shows its document name")

    print("\n[6] no-evidence path (similarity floor raised to maximum)")
    floor = next(s for s in app.slider if "Minimum similarity" in s.label)
    floor.set_value(0.8).run()
    app.text_input[0].set_value("Completely unrelated tractor maintenance query").run()
    ask = next(b for b in app.button if b.label == "Ask Research Question")
    ask.click().run()
    check(not app.exception, "no crash when nothing passes the threshold")
    infos = " ".join(i.value for i in app.info)
    check(
        "no evidence" in infos.lower() or "No passages passed" in infos,
        "user is told honestly that there is no evidence",
    )

    print("\n" + "=" * 70)
    if _failures:
        print(f"FAILED  {len(_failures)} of {_checks} checks:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print(f"PASSED  all {_checks} checks")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
