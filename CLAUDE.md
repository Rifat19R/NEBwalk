@AGENTS.md

# Claude Code — NEBwalk workflow

- Read the imported repository rules before changing files. For scientific or
  architecture work, consult the linked policy and relevant tests first.
- Use Plan Mode for multi-file, numerical-method, API, schema, QE recovery,
  active-learning, or release-impacting changes; make implementation scope
  explicit before editing. Small documentation fixes can be direct.
- Reuse the existing test suite and GitHub Actions. Do not add hooks, agents,
  skills, MCP servers, or CI workflows unless a demonstrated gap justifies them.
- Never assume a green unit test validates QE/MACE physics on untested systems.
- Before finishing, provide a compact evidence report: files changed,
  commands/results, scientific impact, and items needing maintainer approval.
- The maintainer uses ChatGPT for planning/review and Claude Code for
  implementation. Neither assistant may approve its own scientific claims.
