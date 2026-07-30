# FinOps-R 0.4.0 release notes

FinOps-R 0.4.0 is the final release of the Stage 4 durable orchestration and
Stage 5 integrated qualification package. Release publication requires
exact-commit GitHub-hosted verification on Linux and Windows with Python 3.11
and 3.12, followed by verification of the complete four-cell E4 evidence set.

The release includes strict JSON boundary enforcement, bounded API request and
error handling, built-wheel verification, fail-closed artifact validation and
the corrected CodeQL stack-trace-exposure path. These changes do not alter the
locked scientific decision: C5 remains HOLD and C1 remains the production
reference.

The final tag must point to the exact commit that passes CI, Security and E4.
The retained E4 evidence and distribution checksums are release assets rather
than self-referential fields embedded in the tagged commit.

## Prior 0.4.0rc3 release

FinOps-R 0.4.0rc3 introduced the exact-commit E4 hosted-CI contract. It runs the
locked Stage 4 and Stage 5 independent verifiers in every Linux/Windows and
Python 3.11/3.12 matrix cell and emits one machine-readable evidence record per
successful hosted runner.

The E4 verifier requires exactly four distinct cells from one GitHub Actions
run and one immutable commit. Passing E4 does not alter the locked scientific
judgment: C5 remains HOLD and C1 remains the production reference.

## Prior 0.4.0rc2 release

FinOps-R 0.4.0rc2 adds the Stage 5 integrated qualification. It cryptographically binds a
fresh 12,000-case locked model-and-decision run to the independently verified Stage 4 durable
orchestration artifact. The integrated gate requires verified model artifacts, bundle identity,
the controlled-simulation claim boundary, zero unauthorized actions, Stage 4 core and LangGraph
parity, zero duplicate simulation effects, zero external executions, immutable released traces
and exactly-once crash recovery.

The qualification deliberately separates software status from model promotion. The integrated
software and safety contract may pass while the C5 model remains HOLD and C1 remains the
production reference. Hosted CI is still required before a final 0.4.0 release.

## Prior 0.4.0rc1 release

FinOps-R 0.4.0rc1 adds the Stage 4 orchestration qualification without changing
the simulation-only claim boundary. The release candidate includes a
framework-independent reference path, an optional LangGraph adapter, durable
hash-chained SQLite checkpoints, idempotent TradeOps-Sim effect recording,
human-review pause/resume, crash replay, immutable released traces and a frozen
veto-only learned risk verifier.

The retained Stage 4 qualification covers nine locked engineered scenarios.
It is a software and safety-contract qualification, not evidence of bank
production performance or general learned-verifier accuracy. The unrelated
FUNSD, EDGAR, Kleister, HMDA, DocVQA, CFPB, SEC financial statement, M5,
Instacart and Criteo attachments are not canonical Stage 4 inputs.

## Prior 0.3.0 release

FinOps-R 0.3.0 is the first GitHub-ready package of the controlled T+1 exception-resolution benchmark. It ships a Python wheel and source distribution, a fail-closed artifact protocol, a six-candidate experimental ladder, an ISO 20022-like snapshot adapter, Linux/Windows CI definitions and operational qualification scripts.

The scientific result remains deliberately negative: the guarded C5 candidate does not satisfy every locked promotion gate, so the verified C1 baseline remains the fallback. This release qualifies the framework and its safety behavior; it does not claim that C5 is better or that any bank-production outcome was improved.

Local Linux/Python 3.12 qualification covered two consecutive 70-test suites (36 subtests) with 86.81% branch coverage, a 70-test installed-wheel suite, two 1,200-case installed-wheel E2E runs, fresh network-resolved wheel and sdist installation, 59-file artifact reload, actual TCP API lifecycle checks, attached-data validation, failure injection, lint, format, static typing and a no-known-vulnerability dependency audit. It also exercises stable-byte model loading, review-threshold tie safety, bounded immutable ISO staging and graceful Windows service-stop semantics. GitHub-hosted, Windows, Python 3.11 and Docker execution require their own retained evidence and are not claimed here.

See `docs/release_qualification.md`, `qualification_manifest.json` and `release_bundle_manifest.json` for the exact evidence and conditional items.
