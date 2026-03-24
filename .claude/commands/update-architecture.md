You are updating ARCHITECTURE.md to reflect the current state of this repository.

## Steps

1. Run `git diff HEAD --name-only` to see which files have changed. Also run `git status --short` to catch untracked new files.

2. Read the current `ARCHITECTURE.md`.

3. For each changed or new file, read it and determine whether it affects any of these sections in ARCHITECTURE.md:
   - **Overview** — high-level diagram, what the system does
   - **Component Decisions** — why each major library/tool was chosen
   - **Data Flow** — how requests travel through the system
   - **File Map** — which files exist and what each one does
   - **Future Considerations** — table of scale/upgrade paths

4. Update ARCHITECTURE.md **in-place** with only the sections that need changing. Rules:
   - Preserve formatting style, tone, and diagram style of existing content.
   - Add new components/files if they appear for the first time.
   - Update descriptions if existing components were renamed, removed, or significantly changed.
   - Do NOT remove sections or restructure the document — only update the relevant parts.
   - Do NOT add a changelog or "last updated" timestamp.
   - Keep it concise — match the depth of the existing entries.

5. If nothing in ARCHITECTURE.md needs updating (the changes are cosmetic, config-only, or already documented), make no edits and say so briefly.
