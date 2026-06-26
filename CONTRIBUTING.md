# Contributing & version control guide

## Branch strategy

```
main
└── dev
    ├── stage/01-unity-mr-setup
    ├── stage/02-inference-pipeline
    ├── stage/03-spatial-annotations
    ├── stage/04-session-loop
    └── stage/05-rayban-migration
```

### Branch rules

| Branch | Purpose | Who merges |
|---|---|---|
| `main` | Stable, demo-ready only. Never commit directly. | PR from `dev` after stage complete |
| `dev` | Integration branch. All stages merge here first. | PR from stage branch |
| `stage/NN-name` | One branch per build stage. Cut from `dev`. | Author, after stage handoff verified |
| `fix/short-description` | Bug fixes against current stage. Cut from stage branch. | Author |
| `docs/short-description` | Documentation only. | Author |

### Commit message format

```
type(scope): short description

Body (optional) — what changed and why, not how.
```

**Types:** `feat` `fix` `docs` `refactor` `test` `chore`
**Scopes:** `unity` `backend` `bridge` `stt` `tts` `inference` `docs` `scripts`

**Examples:**
```
feat(unity): add passthrough camera rig and floating cube

fix(bridge): handle WebSocket disconnect without crashing server

docs(architecture): add data flow diagram for inference pipeline

chore(scripts): add launch.sh to start all backend services
```

### Per-stage workflow

```bash
# Start a new stage
git checkout dev
git pull origin dev
git checkout -b stage/02-inference-pipeline

# Work, commit often
git add .
git commit -m "feat(backend): add Whisper STT pipeline with audio chunking"

# When stage is complete — merge to dev
git checkout dev
git merge --no-ff stage/02-inference-pipeline
git push origin dev

# When stage is demo-verified — merge dev to main
git checkout main
git merge --no-ff dev
git tag -a v0.2.0 -m "Stage 2 complete: local inference pipeline working"
git push origin main --tags
```

### Tagging convention

```
v0.1.0  — Stage 1 complete (Unity + passthrough + WebSocket)
v0.2.0  — Stage 2 complete (local inference pipeline)
v0.3.0  — Stage 3 complete (spatial annotation system)
v0.4.0  — Stage 4 complete (full session loop)
v1.0.0  — Stage 5 complete (Ray-Ban Display migration)
```
