"""Integration test: drive the OKF MCP server over a REAL stdio transport
(subprocess + MCP handshake), pointed at a throwaway corpus copy via OKF_ROOT.

This covers the one seam the unit tests never touch — the protocol/transport
itself. Requires fastmcp (a real project dependency) and a corpus checkout
(skips without one, like every corpus-dependent test).

Collected by pytest since 2026-09-07. Until then this was a `__main__`-only
script that `python -m pytest` never ran, and the external review found it
still asserting EIGHT tools a month after the server registered nine — a
contract test nobody runs is a contract nobody checks. Still runnable alone:

    python tests/test_mcp_stdio.py
"""
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpus import skip_if_no_corpus                     # noqa: E402

CORPUS_SRC = skip_if_no_corpus()

from fastmcp import Client                                # noqa: E402
from fastmcp.client.transports import PythonStdioTransport   # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# The tool CONTRACT, in one place. Pinned against server.py's decorators by the
# registry drift test as well; this one asserts what a client SEES over the wire.
EXPECTED_TOOLS = ["audit_corpus", "brief", "create_node", "delete_node", "list_nodes",
                  "propose_deletion", "read_node", "resolve_suggestion", "update_node",
                  "validate_corpus"]


def _fresh_corpus() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="okf_mcp_"))
    non_corpus = {".git", "librarian", "tests", "deploy", "__pycache__",
                  ".venv", "venv", ".pytest_cache", ".claude"}
    for sub in CORPUS_SRC.iterdir():
        if sub.is_dir() and sub.name not in non_corpus:
            shutil.copytree(sub, tmp / sub.name)
    # An empty copy would let the whole MCP handshake "pass" over zero nodes.
    found = len(list(tmp.rglob("*.md")))
    if found < 20:
        raise RuntimeError(
            f"refusing to test against an empty corpus: copied {found} node file(s) "
            f"from {CORPUS_SRC}. Set OKF_TEST_CORPUS to the OKF-corpus checkout.")
    return tmp


async def _roundtrip() -> list[tuple[str, str]]:
    results = []

    def check(label, cond):
        results.append(("PASS" if cond else "FAIL", label))

    tmp = _fresh_corpus()
    env = {k: v for k, v in os.environ.items()
           if k not in ("OKF_GIT_COMMIT", "OKF_GIT_PUSH", "OKF_GIT_PUSH_ASYNC")}
    transport = PythonStdioTransport(
        script_path=str(REPO / "server.py"),
        env={**env, "OKF_ROOT": str(tmp)},
        cwd=str(REPO),
    )
    try:
        async with Client(transport) as client:
            tools = sorted(t.name for t in await client.list_tools())
            check(f"handshake + {len(EXPECTED_TOOLS)} tools listed", tools == EXPECTED_TOOLS)

            r = await client.call_tool("validate_corpus", {})
            baseline = r.data["node_count"]
            check(f"validate_corpus ok ({baseline} nodes)", r.data["ok"] is True)

            r = await client.call_tool("create_node", {
                "type": "note", "title": "Via MCP over stdio", "project": "global-os",
                "body": "created through the transport", "summary": "wire test"})
            nid = r.data["id"]
            check("create_node returns id", isinstance(nid, str) and nid.startswith("okf_"))

            r = await client.call_tool("read_node", {"node_id": nid})
            check("read_node round-trips title",
                  r.data["frontmatter"]["title"] == "Via MCP over stdio")

            await client.call_tool("update_node",
                                   {"node_id": nid, "changes": {"status": "parked"}})
            r = await client.call_tool("read_node", {"node_id": nid})
            check("update_node applied (parked)",
                  r.data["frontmatter"]["status"] == "parked")
            # An EMPTY update is refused, not committed as a no-op: on 2026-09-18 one
            # bumped updated_at on an unrelated log node and added a commit.
            r = await client.call_tool("update_node", {"node_id": nid, "changes": {}})
            check("empty update_node refused", r.data.get("ok") is False)

            try:
                await client.call_tool("create_node",
                                       {"type": "note", "title": "Born dead",
                                        "project": "global-os", "status": "retired"})
                check("born-retired refused over protocol", False)
            except Exception as e:
                check(f"born-retired refused over protocol ({type(e).__name__})", True)

            # v1.7: a node with no home is refused AT THE PROTOCOL — the schema
            # declares `project` required, so the client never reaches the store.
            try:
                await client.call_tool("create_node",
                                       {"type": "note", "title": "Homeless"})
                check("no-project refused over protocol", False)
            except Exception as e:
                check(f"no-project refused over protocol ({type(e).__name__})", True)

            r = await client.call_tool("validate_corpus", {})
            check("corpus still valid, +1 node",
                  r.data["ok"] is True and r.data["node_count"] == baseline + 1)

            # v1.8: a session's close_out must carry `tasks` — "nothing to do" is
            # said with [], never implied by omission (the session close-out playbook's TASKS line, hardened
            # 2026-09-15 on the owner's ruling after the week-38 audit run named it soft).
            three = {"keep": ["x"], "conflicts": [], "nothing_to_keep": False}
            body = "# s\n\n## Purpose\nwire test\n"
            # The close_out gate REFUSES with {ok: false, error}, it does not raise —
            # the store's gates raise, the tool-level gates return. The first version
            # of this check expected an exception and the deploy gate refused 1479861
            # on it (2026-09-15): the gate doing its job on the test of a gate.
            r = await client.call_tool("create_node", {
                "type": "session", "title": "Session — no tasks field",
                "project": "global-os", "summary": "wire test " * 3,
                "body": body, "close_out": three})
            check("session without close_out.tasks refused",
                  r.data.get("ok") is False and "tasks" in str(r.data.get("error", "")))
            r = await client.call_tool("create_node", {
                "type": "session", "title": "Session — with an empty tasks list",
                "project": "global-os", "summary": "wire test " * 3,
                "body": body, "close_out": {**three, "tasks": []}})
            sid = r.data.get("id")
            check("session with tasks: [] accepted", isinstance(sid, str) and sid.startswith("okf_"))

            # resolve_suggestion (2026-09-18): the action surface for audit_corpus
            # suggestions — the graph panel that used to be it went on 2026-08-03.
            r = await client.call_tool("resolve_suggestion",
                                       {"a": nid, "b": sid, "action": "accept"})
            check("resolve_suggestion accept links the pair",
                  r.data.get("ok") is True and r.data.get("already") is False)
            r = await client.call_tool("read_node", {"node_id": nid})
            check("accept wrote relates_to",
                  sid in (r.data["frontmatter"].get("relates_to") or []))
            r = await client.call_tool("resolve_suggestion",
                                       {"a": sid, "b": nid, "action": "accept"})
            check("accept is idempotent in either direction",
                  r.data.get("ok") is True and r.data.get("already") is True)
            r = await client.call_tool("resolve_suggestion",
                                       {"a": nid, "b": sid, "action": "dismiss",
                                        "reason": "wire test"})
            check("resolve_suggestion dismiss records the pair",
                  r.data.get("ok") is True and r.data.get("action") == "dismiss")
            r = await client.call_tool("resolve_suggestion",
                                       {"a": nid, "b": "okf_00000000_000000_0000",
                                        "action": "accept"})
            check("resolve_suggestion refuses an unknown id", r.data.get("ok") is False)
            r = await client.call_tool("resolve_suggestion",
                                       {"a": nid, "b": sid, "action": "merge"})
            check("resolve_suggestion refuses an unknown action", r.data.get("ok") is False)
            # unlink so the wire-test nodes can be deleted (the gates refuse live edges)
            await client.call_tool("update_node", {"node_id": nid, "changes": {"relates_to": []},
                                                   "allow_unlink": True})

            await client.call_tool("update_node", {"node_id": sid, "changes": {"status": "retired"}})
            r = await client.call_tool("delete_node", {"node_id": sid, "reason": "wire test"})
            check("wire-test session cleaned up", r.data["ok"] is True)

            # Deletion policy over the wire (the gated-deletion decision, 2026-08-02): agent-callable,
            # but a node that is not retired fails a gate and nothing is removed.
            r = await client.call_tool("delete_node", {"node_id": nid, "reason": "wire test"})
            check("delete_node refuses a non-retired node",
                  r.data["ok"] is False and r.data.get("blocked_by"))
            r = await client.call_tool("read_node", {"node_id": nid})
            check("refused delete removed nothing", r.data["frontmatter"]["id"] == nid)

            await client.call_tool("update_node",
                                   {"node_id": nid, "changes": {"status": "retired"}})
            r = await client.call_tool("delete_node", {"node_id": nid, "reason": "wire test"})
            check("delete_node removes a retired orphan", r.data["ok"] is True
                  and r.data["deleted"] == nid)
            r = await client.call_tool("validate_corpus", {})
            check("corpus valid, back to baseline",
                  r.data["ok"] is True and r.data["node_count"] == baseline)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def test_mcp_stdio_roundtrip():
    results = asyncio.run(_roundtrip())
    failed = [label for status, label in results if status == "FAIL"]
    assert not failed, "MCP stdio round-trip failed:\n  " + "\n  ".join(failed)


def main() -> int:
    results = asyncio.run(_roundtrip())
    print("=== MCP stdio round-trip ===")
    for status, label in results:
        print(f"  [{status}] {label}")
    n_fail = sum(1 for s, _ in results if s == "FAIL")
    print(f"\n{'ALL PASS' if not n_fail else str(n_fail) + ' FAILED'} — {len(results)} checks")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
