# Pinned third-party dependencies

## tau2-bench

- Upstream: https://github.com/sierra-research/tau2-bench
- Pinned commit: `a2c024725189473d2d7cea3a5cfdbcc67478e41f`
- Commit date: 2026-08-18
- Upstream version string: `tau2 1.0.1`

The working copy under `third_party/tau2-bench/` is a detached-HEAD checkout of
that commit and is **git-ignored** — it is not vendored into this repo's history.

Reproduce it with:

```bash
git clone https://github.com/sierra-research/tau2-bench.git third_party/tau2-bench
git -C third_party/tau2-bench checkout a2c024725189473d2d7cea3a5cfdbcc67478e41f
```

Rationale (see CLAUDE.md hard constraint #4): tau2 is under active development and
internal API refactors break custom agent implementations. We do not track `main`.

Bumping the pin is a deliberate act: update the commit here, re-run the harness
unit tests and the adapter contract tests, and re-run the retail subset baseline
before trusting any number produced against the new pin.

### Not yet a GitHub fork

CLAUDE.md says "fork + pin". Only the pin is done. Creating a fork under your own
GitHub account is an outward-facing action left to you; do it when you first need
to carry local patches against tau2. Once forked, change the clone URL above and
add your fork as `origin`, keeping upstream as a second remote.
