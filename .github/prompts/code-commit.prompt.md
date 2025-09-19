# Version Control — Short Prompt

Keep the git workflow tight, interactive, and fast. Only ask the developer when a decision affects commit scope or safety.

1. Status

- Check repo status (staged, modified, untracked). Summarize counts and key files.

2. Quick Review

- Show per-file change types (added/modified/deleted) and flag critical issues: secrets, large files (>10MB), failing tests.

3. Stage

- Suggest minimal staging: group related changes; prefer specific `git add file` over `git add .`.
- Ask only: "Stage these files?" with a short list.

4. Commit message

- Create 2 concise conventional-commit options: `<type>(scope): short summary` (<=50 chars).
- If change is complex, add a 1-paragraph body explaining why.

5. Commit

- Run `git commit` with chosen message. Report commit hash.

6. Push (optional)

- Confirm branch and remote. Run `git push` if approved. Report remote URL or instructions to open a PR.

Notes

- Always scan for secrets and .gitignore issues before commit.
- Prefer small, single-purpose commits.
- Ask developer only when necessary (staging, commit selection, push).

Response format

- Use short bullets, ✅ for done, ❌ for issues, and a final "Next steps" line.
