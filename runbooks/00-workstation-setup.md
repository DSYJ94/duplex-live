# Runbook 00 — Workstation & access setup (VS Code + agentic Kimi)

Goal: one-time setup so that (a) **you** get VS Code Remote-SSH on Isambard,
and (b) **Kimi** can execute agentically over SSH from this Mac after you
authenticate once. Allocation: Isambard-AI Phase 2, terminates 2026-10-21.

## 0. Accounts (you, interactive — one time)

- BriCS portal access confirmed (screenshot verified 2026-09-14).
- Check the portal docs for the exact SSH login host and whether key enrolment
  or MFA is required: https://docs.isambard.ac.uk

## 1. SSH key-based access (enables both VS Code and Kimi)

```bash
# on this Mac — generate a dedicated key if you don't have one
ssh-keygen -t ed25519 -f ~/.ssh/isambard_ed25519   # set a passphrase

# enrol the public key per the portal's instructions (or ssh-copy-id if supported)
cat ~/.ssh/isambard_ed25519.pub
```

Add to `~/.ssh/config`:

```
Host isambard
  HostName <login-host-from-portal>     # e.g. isambard.ai — verify
  User <your-username>
  IdentityFile ~/.ssh/isambard_ed25519
  # connection reuse: first login authenticates, later ones ride the socket
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 8h
```

Test: `ssh isambard 'hostname'` should log in with just the key passphrase
(use `ssh-add ~/.ssh/isambard_ed25519` to cache it in the agent).

**Why ControlMaster matters:** after your first authenticated login, subsequent
SSH calls (mine included) reuse the open connection — no repeated prompts.

## 2. VS Code (your interactive environment)

1. Install the **Remote-SSH** extension.
2. `Cmd+Shift+P → Remote-SSH: Connect to Host → isambard` (uses the config above).
3. Open the remote folder; clone the repo there:
   ```bash
   git clone https://github.com/DSYJ94/duplex-live   # private: use a PAT or SSH key
   ```
4. **Caveat — no Dev Containers on HPC:** Isambard uses Apptainer, not Docker.
   Instead: build/enter the container or venv per Runbook 01, then point VS
   Code's Python interpreter at it (`Python: Select Interpreter`).
5. Jupyter: start `jupyter lab --no-browser --port 8889` on a compute node,
   forward with `ssh -L 8889:<node>:8889 isambard`, connect from local VS Code.

## 3. Agentic Kimi access (how I'll work)

Once step 1 works, I can execute remotely from this chat. Ground rules I'll
follow (also recorded here for transparency):

- I run remote commands as `ssh isambard '<cmd>'` / `scp` / `sftp` from your
  Mac's shell; I never ask for or handle your password.
- Long jobs go through `sbatch` scripts or `tmux` on Isambard so they outlive
  any single command; I poll with `squeue`/log tails.
- Destructive or allocation-heavy actions (`rm -rf`, `scancel`, large `salloc`)
  are shown to you first.
- Generated code/docs get committed to the GitHub repo so the two environments
  never drift: repo = single source of truth.
- First agentic session checklist:
  - [ ] `ssh isambard 'hostname && nvidia-smi -L'`
  - [ ] `sinfo` / partition + limits discovery → append to Issue #1
  - [ ] `git clone` the repo on Isambard
  - [ ] Runbook 01 bring-up

## 4. If agentic SSH isn't possible

Some HPCs require per-login MFA that can't be reused. In that case the fallback
loop is: you run the runbook commands in VS Code's terminal and paste outputs
here; I update runbooks/code and you re-run. Slower, but everything in
`runbooks/` is written to work that way too.
