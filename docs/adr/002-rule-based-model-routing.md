# ADR-002 - Rule-based model routing instead of cost-based optimisation

**Date:** 2026-11-09
**Status:** Accepted

## Context

SY1-Gov needs to select between a cheaper, faster model (Haiku) and a more capable model
(Sonnet) for each tool call. The goal is to minimise cost while preserving quality for
calls that need it.

Three approaches were considered:

1. **Always Sonnet** - simplest, highest quality, highest cost
2. **Rule-based routing** - explicit rules on tier and input complexity
3. **Cost-optimisation loop** - track actual outcomes, tune thresholds over time

## Decision

Rule-based routing via `model_router.py`. Rules are:

- READ + simple inputs - Haiku
- READ + complex inputs - Sonnet
- REVERSIBLE - Sonnet (state-changing, needs reliability)
- IRREVERSIBLE - Sonnet (high stakes, needs reliability)
- Kill switch active - Haiku always (conservative fallback)

Complexity is assessed by input key count (threshold: 3 keys) and total string length
(threshold: 200 characters).

## Consequences

**Positive:**
- Deterministic and testable - routing decisions are fully predictable from inputs
- No feedback loop required - works from day one without historical data
- Kill switch integration is a natural fit for a rule layer
- Easy to audit - routing reason is recorded in every `TraceSpan`

**Negative:**
- Thresholds (3 keys, 200 chars) are heuristics, not empirically tuned
- Cannot adapt to actual model performance differences over time
- May over-route simple REVERSIBLE calls to Sonnet

## Alternatives considered

A cost-optimisation loop was deferred because it requires production traffic data that does
not yet exist. It can be layered on top of the rule-based router in a future phase without
changing the interface.

Always-Sonnet was rejected on cost grounds for the high-volume READ tier.
