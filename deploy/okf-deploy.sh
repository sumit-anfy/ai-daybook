#!/bin/bash
# Every 2 minutes: make sure the CORPUS reached GitHub, then deploy any new CODE.
#
# Two jobs, in that order, because the corpus one must run even when there is no
# code to deploy — which is almost always.
#
# Pull, never push, for CODE: this box is not a code author, so a dirty or diverged
# code checkout stops the deploy loudly instead of being rewritten. The CORPUS is
# the opposite — this box is its ONLY author, so an unpushed corpus commit is
# never someone else's work and recovering it is unambiguous.
set -uo pipefail

REPO=/home/okf/OKF
CORPUS=/home/okf/OKF-corpus
# Each repo has its own branch. Override in the unit (Environment=…) if yours differ.
CODE_BRANCH=${OKF_CODE_BRANCH:-main}
CORPUS_BRANCH=${OKF_CORPUS_BRANCH:-main}

# Post-restart health probe (built 2026-08-08 on the owner's ruling, after the
# second unwatched crash-loop in two days). The marker is read by
# librarian.audit.compute_deploy_health, so a failed deploy surfaces in every
# audit_corpus call and the weekly digest — not only in this journal.
MARKER=/home/okf/.okf/deploy_health.json

# The test gate (built 2026-09-07 on the owner's ruling, from the external review's
# finding that nothing stood between `git push` and a restarted service — four
# failing tests and no CI). The incoming revision is checked out into a
# throwaway worktree and the suite runs there, against the bundled example
# corpus. Red means: do not fast-forward, do not restart, remember
# the refused revision so the suite is not re-run every 2 minutes, and leave a
# marker librarian.audit.compute_deploy_gate surfaces in every audit.
GATE_WORKTREE=/home/okf/OKF-gate
GATE_MARKER=/home/okf/.okf/deploy_gate.json
GATE_LOG=/home/okf/.okf/deploy_gate.log
GATE_REFUSED=/home/okf/.okf/deploy_gate.refused
GATE_TIMEOUT=${OKF_GATE_TIMEOUT:-900}
GATE_PYTHON=$REPO/.venv/bin/python
# The gate's OWN embedding cache: the suite would otherwise read and rewrite
# ~/.okf/embeddings.npz while two live services do the same. Seeded from the
# live cache once, so the first gated deploy does not re-embed the corpus.
GATE_EMB_CACHE=/home/okf/.okf/embeddings.gate.npz
LIVE_EMB_CACHE=/home/okf/.okf/embeddings.npz
# The chunk-vector cache (2026-10-13) sits beside each node cache, and is
# copied FRESH from the live one every run: the gate never embeds in the
# background (embed._background is service-only), so a gate cache seeded once
# would go stale after the first corpus-wide change and the suite would run
# with the chunk layer off from then on, testing less than production runs.
GATE_CHUNK_CACHE=/home/okf/.okf/embeddings.gate.chunks.npz
LIVE_CHUNK_CACHE=/home/okf/.okf/embeddings.chunks.npz
PROBE_WEB=${OKF_PROBE_WEB:-http://127.0.0.1:8790/graph}
PROBE_MCP=${OKF_PROBE_MCP:-http://127.0.0.1:8765/mcp}
PROBE_DEADLINE=${OKF_PROBE_DEADLINE:-45}
PROBED=0

# A queued push is not a stall. The background pusher (OKF_GIT_PUSH_ASYNC) leaves
# HEAD ahead of the remote for about a second on every write, and this timer will
# land inside that window often. `store.push_pending()` guards the in-process
# check, but this script is a DIFFERENT process and cannot see that state — so it
# uses the commit's own age instead, which is stateless and needs no coordination.
# A push that has not completed 2 minutes after the commit is not in flight.
GRACE_SECONDS=120


# ── 1. the corpus: did the Librarian's writes actually reach GitHub? ───────────
# This exists because making the push asynchronous (2026-08-02) moved a failed
# corpus push from an immediate signal in the tool response to something only
# discoverable when someone next opened the graph, or up to a week later in the
# weekly digest. The owner asked the right question: "how will we know if push from
# VPS to git has failed?" — this is the answer.
check_corpus() {
    [ -d "$CORPUS" ] || return 0

    sudo -u okf git -C "$CORPUS" fetch -q origin "$CORPUS_BRANCH" 2>/dev/null || {
        echo "corpus: fetch failed (network or auth) — cannot confirm sync this cycle"
        return 0        # not an alarm; the next cycle re-checks
    }

    local ahead
    ahead=$(sudo -u okf git -C "$CORPUS" rev-list --count "origin/$CORPUS_BRANCH..HEAD" 2>/dev/null) || return 0
    [ "${ahead:-0}" -eq 0 ] && return 0          # in sync — the normal case, stay silent

    # Old enough to rule out a push that is merely in flight?
    local head_ts now age
    head_ts=$(sudo -u okf git -C "$CORPUS" log -1 --format=%ct HEAD 2>/dev/null) || return 0
    now=$(date +%s)
    age=$(( now - head_ts ))
    [ "$age" -lt "$GRACE_SECONDS" ] && return 0  # queued, not stalled

    echo "corpus: $ahead commit(s) unpushed for ${age}s — attempting recovery"

    # Recover under the SAME lock the Librarian takes for every write
    # (store._repo_lock -> flock on .okf.lock), so this can never race a write or
    # a push already in progress. Run as `okf` so nothing here can leave a
    # root-owned file in a tree the services write to.
    if sudo -u okf flock -x -w 30 "$CORPUS/.okf.lock" \
         bash -c "cd '$CORPUS' && git pull --rebase -q origin '$CORPUS_BRANCH' && git push -q origin '$CORPUS_BRANCH'"; then
        echo "corpus: recovered — $ahead commit(s) pushed to GitHub"
        return 0
    fi

    # Only now is a human needed. Kept rare on purpose: an alarm that fires on the
    # ordinary in-flight window would be ignored within a week (the fail-loudly rule rule 3).
    echo "corpus: ERROR — $ahead commit(s) have been unpushed for ${age}s and" \
         "recovery FAILED. The corpus exists on this machine only. Resolve by hand:" \
         "sudo -u okf git -C $CORPUS pull --rebase origin $CORPUS_BRANCH && ... push"
    return 1
}


# ── the probe: do the services ANSWER, not merely run? ─────────────────────────
# `systemctl is-active` was checked on both 2026-08 crash-loops and its verdict
# went to this journal, which nobody reads; and a unit can be `active` around a
# frozen app (the 2026-08-07 event-loop stall froze every surface while the
# unit stayed green). An HTTP answer over loopback is the behaviour users
# actually need. okf-web must serve /graph with 200 (the login page needs no
# auth); the MCP counts as alive on ANY http status — 401 is its healthy
# unauthenticated answer, and only 000 means nothing answered at all.
probe_services() {
    PROBED=1
    local deadline=$(( $(date +%s) + PROBE_DEADLINE )) web mcp
    while :; do
        web=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$PROBE_WEB" || true)
        mcp=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$PROBE_MCP" || true)
        [ "$web" = "200" ] && [ "$mcp" != "000" ] && break
        if [ "$(date +%s)" -ge "$deadline" ]; then
            local failed=()
            [ "$web" = "200" ] || failed+=("okf-web (http $web)")
            [ "$mcp" != "000" ] || failed+=("okf (no HTTP answer)")
            local joined rev
            joined=$(printf '%s","' "${failed[@]}"); joined=${joined%'","'}
            rev=$(sudo -u okf git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
            echo "probe: UNHEALTHY — ${failed[*]} — marker written for the audit"
            sudo -u okf mkdir -p "$(dirname "$MARKER")"
            printf '{"when": %s, "revision": "%s", "failed": ["%s"]}\n' \
                "$(date +%s)" "$rev" "$joined" \
                | sudo -u okf tee "$MARKER" >/dev/null
            return 1
        fi
        sleep 3
    done
    # Healthy — a lingering marker from a past failure is now stale; clear it
    # so the flag self-heals (a manual service fix counts, not only a deploy).
    if [ -f "$MARKER" ]; then
        sudo -u okf rm -f "$MARKER"
        echo "probe: healthy — cleared the unhealthy-deploy marker"
    fi
    return 0
}


# ── the gate: does the incoming revision pass its own tests? ───────────────────
# Runs as `okf` (the services' user, so the model cache and corpus are readable
# and nothing root-owned is left behind). pytest's own exit code is the verdict
# — never piped, never tailed, because that exact mistake shipped two red tests
# as green once already (the auto-deploy session). Exit 5 (nothing collected)
# is a failure too: a gate that ran no tests has verified nothing.
run_gate() {
    local rev=$1
    sudo -u okf mkdir -p "$(dirname "$GATE_MARKER")"
    # A previous run killed mid-way (timer stopped, box rebooted) leaves the
    # worktree registered; clear it or `worktree add` refuses.
    sudo -u okf git -C "$REPO" worktree remove --force "$GATE_WORKTREE" 2>/dev/null
    sudo -u okf rm -rf "$GATE_WORKTREE"
    if ! sudo -u okf git -C "$REPO" worktree add -q --detach "$GATE_WORKTREE" "$rev"; then
        echo "gate: could not check out ${rev:0:7} into $GATE_WORKTREE"
        return 1
    fi
    [ -x "$GATE_PYTHON" ] || { echo "gate: $GATE_PYTHON missing"; return 1; }
    [ -f "$GATE_EMB_CACHE" ] || sudo -u okf cp "$LIVE_EMB_CACHE" "$GATE_EMB_CACHE" 2>/dev/null
    [ -f "$LIVE_CHUNK_CACHE" ] && sudo -u okf cp "$LIVE_CHUNK_CACHE" "$GATE_CHUNK_CACHE" 2>/dev/null
    # cwd MUST be the worktree: `python -m pytest` puts the cwd on sys.path
    # first, so run from $REPO and `import librarian` would resolve to the code
    # already running — the gate would then test the old revision and pass the
    # new one on its behalf. -p no:cacheprovider: nothing written into the
    # worktree that outlives it.
    ( cd "$GATE_WORKTREE" && sudo -u okf env \
          OKF_EMB_CACHE="$GATE_EMB_CACHE" timeout "$GATE_TIMEOUT" \
          "$GATE_PYTHON" -m pytest -q -p no:cacheprovider tests ) > "$GATE_LOG" 2>&1
    local rc=$?
    sudo -u okf git -C "$REPO" worktree remove --force "$GATE_WORKTREE" 2>/dev/null
    sudo -u okf git -C "$REPO" worktree prune 2>/dev/null
    return $rc
}

refuse_revision() {
    local rev=$1 running=$2 rc=$3 summary
    # pytest's last line is its own summary ("3 failed, 268 passed in 41.2s");
    # a timeout leaves no such line, so say so rather than quoting noise.
    if [ "$rc" -eq 124 ]; then
        summary="the suite did not finish within ${GATE_TIMEOUT}s."
    else
        summary=$(tail -n 1 "$GATE_LOG" 2>/dev/null | tr -d '"' | cut -c1-200)
        summary=${summary:-"pytest exit $rc with no summary line."}
    fi
    echo "gate: REFUSING ${rev:0:7} — $summary (log: $GATE_LOG)"
    echo "$rev" | sudo -u okf tee "$GATE_REFUSED" >/dev/null
    printf '{"when": %s, "revision": "%s", "running": "%s", "summary": "%s"}\n' \
        "$(date +%s)" "${rev:0:7}" "${running:0:7}" "$summary" \
        | sudo -u okf tee "$GATE_MARKER" >/dev/null
}


# ── 2. the code: pull and restart, only when the revision actually changed ─────
deploy_code() {
    cd "$REPO" || { echo "deploy: $REPO missing"; return 1; }

    if ! sudo -u okf git fetch -q origin "$CODE_BRANCH" 2>/dev/null; then
        echo "deploy: fetch failed (network or auth) — leaving the running code alone"
        return 1
    fi

    local local_rev remote_rev
    local_rev=$(sudo -u okf git rev-parse HEAD)
    remote_rev=$(sudo -u okf git rev-parse "origin/$CODE_BRANCH")
    [ "$local_rev" = "$remote_rev" ] && return 0      # nothing to do, stay silent

    # Restarting on a dirty tree would discard whatever is uncommitted here. Nothing
    # should ever edit code on this box, so a dirty tree means something is wrong and
    # is worth stopping for.
    if [ -n "$(sudo -u okf git status --porcelain)" ]; then
        echo "deploy: REFUSING — $REPO has local changes; resolve by hand"
        return 1
    fi

    # Already refused this exact revision: stay silent (the marker is the
    # signal, and a line here every 2 minutes is the journal nobody reads).
    if [ -f "$GATE_REFUSED" ] && [ "$(cat "$GATE_REFUSED")" = "$remote_rev" ]; then
        return 1
    fi

    echo "deploy: ${local_rev:0:7} -> ${remote_rev:0:7} — running the test gate first"
    local gate_rc=0
    run_gate "$remote_rev" || gate_rc=$?
    if [ "$gate_rc" -ne 0 ]; then
        refuse_revision "$remote_rev" "$local_rev" "$gate_rc"
        return 1
    fi
    # Green: a stale refusal (of an older revision) no longer describes master.
    sudo -u okf rm -f "$GATE_REFUSED" "$GATE_MARKER"
    echo "gate: ${remote_rev:0:7} passed — $(tail -n 1 "$GATE_LOG" 2>/dev/null)"

    if ! sudo -u okf git merge --ff-only "origin/$CODE_BRANCH"; then
        echo "deploy: REFUSING — not a fast-forward; the code checkout has diverged"
        return 1
    fi

    # Restart only after the code is actually in place.
    systemctl restart okf okf-web
    # Behavioural verdict, not unit state — probe_services retries within its
    # deadline, which also absorbs normal startup time.
    if ! probe_services; then
        echo "deploy: ${remote_rev:0:7} restarted but UNHEALTHY"
        return 1
    fi
    echo "deploy: ok, running ${remote_rev:0:7}"
    return 0
}


# Manual entry point: `okf-deploy.sh probe` runs only the probe — for
# operators, and for exercising both probe branches without a deploy.
if [ "${1:-}" = "probe" ]; then
    probe_services
    exit $?
fi

rc=0
check_corpus || rc=1
deploy_code  || rc=1
# Self-clearing: a past failure's marker is re-checked every cycle (unless
# this run already probed), so a manual fix clears the flag within 2 minutes
# without waiting for the next deploy.
if [ -f "$MARKER" ] && [ "$PROBED" -eq 0 ]; then
    probe_services || rc=1
fi
exit "$rc"
