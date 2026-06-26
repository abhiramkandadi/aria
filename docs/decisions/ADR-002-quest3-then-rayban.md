# ADR-002: Build on Quest 3 first, migrate to Ray-Ban Display second

**Date:** 2025-06-26
**Status:** accepted

## Context
Target end-state is Ray-Ban Display glasses. However the Meta Wearables Device Access Toolkit display APIs only opened May 2026 and the ecosystem is very early.

## Decision
Build the full working system on Quest 3 as a proof of concept. The backend is designed frontend-agnostic from day one. Migrating to Ray-Ban Display is a frontend swap — no backend changes required.

## Consequences
- Can demo a working system now without waiting for SDK maturity
- Quest 3 gives richer MR demo (spatial anchors, 6DoF) for proof-of-concept
- Backend must stay strictly decoupled from Unity-specific code
- Creates compelling "Phase 2" story for investors and Meta partnership conversations

## Alternatives considered
- **Ray-Ban Display first**: rejected — display SDK too new, no spatial anchor support yet
- **Both simultaneously**: rejected — doubles complexity at proof-of-concept stage
