---
description: Orchestrate a team of subagents to stage, commit, and push all pending repo changes — protects main session context
---

You are orchestrating a **multi-agent git sync workflow** for the Claudium repository. Your job is to delegate all git operations to subagents so the main session context is not consumed by heavy diff/log output. Follow the phases below exactly.

---

## Phase 1 — Parallel Analysis

Spawn **two agents in parallel** (single Agent tool call with two blocks, or two simultaneous calls). Do NOT act on Phase 2 until both complete.

### Agent A — Status & Diff
**Task:** Analyse what has changed. Research only, no writes.
- Run `git status` (never use `-uall` flag)
- Run `git diff HEAD` to see all unstaged changes
- Run `git diff --cached` to see staged changes
- Run `git log --oneline -8` to understand recent commit message style (prefix: feat/fix/docs/refactor/test/chore)
- Identify every changed/untracked file by name (avoid `git add -A`)
- Draft a concise commit message: one sentence, correct prefix, focus on the WHY
- **Return:** list of files to stage, proposed commit message

### Agent B — Safety Check
**Task:** Verify the repository is safe to commit and push.
- `git branch --show-current` — confirm branch name
- Check for any file containing real API key patterns: `grep -r "sk-ant-" . --include="*.py" --include="*.toml" --include="*.env" -l 2>/dev/null || echo "none"`
- Check `~/.claude/security-preferences.json` — if the pre-commit scanner is enabled, warn that it may block the push (workaround: set `"enabled": false` temporarily)
- Check whether a GH_TOKEN env var is available: `echo ${GH_TOKEN:+set} || echo "not set"`
- **Return:** safe/blocked signal, branch name, GH_TOKEN availability

---

## Phase 2 — Commit (after Phase 1 completes)

Spawn **one agent** with the findings from Agent A and Agent B.

**Task:** Stage and commit the changes.
- Stage only the specific files listed by Agent A — never use `git add -A` or `git add .`
- If Agent B flagged any secrets, STOP and report to user — do not commit
- Create the commit using a HEREDOC so formatting is preserved:
  ```
  git commit -m "$(cat <<'EOF'
  <commit message from Agent A>

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
  EOF
  )"
  ```
- Run `git status` after commit to verify working tree is clean
- **Return:** commit hash, commit message, git status output

---

## Phase 3 — Push (after Phase 2 completes)

Spawn **one agent** with the commit hash from Phase 2.

**Task:** Push to remote. The repo is `ai-craftsman404/claudium` on GitHub.

If GH_TOKEN is available (Agent B confirmed):
```bash
GH_REMOTE="https://${GH_TOKEN}@github.com/ai-craftsman404/claudium.git"
git remote set-url origin "$GH_REMOTE"
git push origin HEAD
git remote set-url origin "https://github.com/ai-craftsman404/claudium.git"
```
Do the URL set, push, and restore in a single compound command so the token is never left in the remote config.

If GH_TOKEN is NOT available: report to user that push cannot proceed — they should run `export GH_TOKEN=<token>` and re-invoke `/project:git-sync`.

- **Return:** push output (success or error), final remote URL (must be clean, no token)

---

## Phase 4 — Adversarial Review (after Phase 3 completes)

Spawn **one final agent** to independently verify the entire operation.

**Task:** Adversarial check — find anything wrong.
- `git log --oneline -3` — confirm latest commit looks correct
- `git show HEAD --stat` — verify no secrets or unintended files were committed
- `git remote get-url origin` — confirm remote URL is clean (no token embedded)
- `git status` — confirm working tree is clean
- **Return:** PASS or FAIL with specific findings

---

## After all phases complete

Report to the user:
- Commit hash and message
- Files staged
- Push result (branch → remote)
- Adversarial review verdict

If any phase fails, report the failure and the agent's exact output. Never silently skip a failure.

---

## Hard rules (never violate)

- NEVER use `git add -A` or `git add .`
- NEVER commit files matching `.env*` if they contain real API keys
- NEVER force push to main or master
- NEVER pass `--no-verify` to git commit or git push
- NEVER leave a GH_TOKEN embedded in the remote URL after the push
- Always use HEREDOC for commit messages to preserve formatting
- Co-Authored-By line is mandatory on every commit
