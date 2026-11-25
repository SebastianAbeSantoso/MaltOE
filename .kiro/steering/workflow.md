---
inclusion: always
---

# Workflow Rules

## Push at every checkpoint

- After every checkpoint (a finished spec task, a gate decision, a spike that is ready for the builder to run, or any point where work stops for the builder's review), commit and push to the remote before reporting back. Do not leave finished work only in the sandbox.
- Work on a branch and open or update a pull request. Do not commit directly to `main`.
- State where the work can be reviewed (branch or pull request link) in the report.

## Attribution

- Commits are authored and committed as the builder: `Sebastian Abe Santoso <185665220+SebastianAbeSantoso@users.noreply.github.com>`. Set it through `GIT_AUTHOR_*` and `GIT_COMMITTER_*` environment variables for the command; do not change git config.
- Pull requests and merges go through the GitHub API, which already acts as the builder's account.

## Merging

- Merge a pull request only when the builder asks.
