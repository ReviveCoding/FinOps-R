# Known limitations

## Release blockers for C5

- C5 is not promoted. Its normal-split residual harm is 83.2% worse than C1 under the locked cost model.
- Its critical false-greenlight upper bound is 0.03846, above the 0.03 gate.
- Its worst-slice Macro-F1 delta is -0.12711, below the -0.05 non-regression gate.
- On stress cases, its harm is 2.56 times C1 and mandatory manual intervention is 36.30%, above the 20% capacity contract.
- C1 remains the operational fallback. This is an intentional safety outcome, not a hidden failed experiment.

## Scientific limits

- All case labels, actions and outcomes are controlled synthetic data. No bank production impact, compliance outcome or monetary benefit is established.
- The decision-cost model is explicit but simulated. A real operations team must estimate review, hold, miss and correction costs.
- RSOP improves point estimates, but both Macro-F1 and harm confidence intervals include zero improvement.
- The stress generator is a controlled distribution shift, not a historically observed market crisis.
- One locked model seed plus paired entity-cluster bootstrap was evaluated. This corrects within-entity resampling dependence but does not quantify training-seed variability; a preregistered multi-seed study remains recommended.

## Operational and security limits

- Joblib uses pickle semantics. Only artifacts produced locally by this pipeline are in scope. Hash and runtime checks reduce accidental corruption and incompatibility, but they are not a cryptographic signature or sandbox.
- Exact core package versions are required for model loading. An incompatible environment fails closed and must be recreated from `requirements-lock.txt`.
- The exact lock pins versions but does not include distribution hashes. `pip-audit` reports no known vulnerabilities for the locked versions, but this is not equivalent to a fully hashed or signed software supply chain.
- The Uvicorn service passed three real TCP startup/shutdown cycles, 96 concurrent health requests, validation failures and port-conflict checks on Linux/Python 3.12. Windows graceful-stop logic has unit coverage but was not executed on Windows.
- Windows and Python 3.11 are represented in the GitHub Actions matrix but were not executed locally. No remote is configured, so GitHub evidence remains Level 1 (file inspection); Linux command parity is locally exercised but is not a hosted-run claim.
- The final wheel and sdist were installed non-editably in fresh virtual environments with dependencies resolved from the package index. The wheel also passed the complete 70-test installed-package suite against the latest dependency versions permitted by `pyproject.toml`.
- Docker is configured but was not executed because the Docker executable is absent.
- GPU, Qwen, LangGraph and external bank-system connectors are intentionally NOT_APPLICABLE. The implemented model study is CPU-first and simulator-only. Docker now packages the same local CLI; it does not broaden the trust or claim boundary.
- The supplied ISO 20022-like CSV snapshot has a validated staging adapter, but it is project-generated synthetic data, not official ISO conformance material or real bank traffic.
- BPI, CFPB, SEC FTD and FUNSD are governed proxy/stress/optional sources only. Phase 2 adds strict, provenance-preserving CFPB snapshot staging, and Phase 3 adds exact-snapshot BPI 2017 workflow staging. Neither phase establishes a model benchmark claim, and none of these sources is joined into TradeOps-Sim case labels.
- BPI Challenge 2017 describes one anonymized Dutch financial-institution loan process observed in 2016. It is not evidence about current workflows, U.S. institutions, securities post-trade operations, causal policy effects or production impact. Application-level chronological splits reduce leakage but do not remove temporal, organizational or process-selection bias.
- BPI `org:resource` values are deterministically pseudonymized and excluded in raw form from staged tables. Deterministic pseudonyms may remain linkable within this snapshot and are not claimed to provide irreversible anonymization.
- The exact qualification runtime uses NumPy 2.3.5 because the project supports Python 3.11; the package metadata prevents installation of NumPy 2.5 or newer.

## Recommended follow-up

1. Redesign C5 hard-gate coverage so stress HOLD demand stays within capacity without raising unsafe automation.
2. Repeat C1 and C3 over at least five training seeds and a larger independent stress generator family.
3. Replace rule-derived evidence with a versioned public policy corpus before evaluating an LLM or multi-agent challenger.
4. Execute the checked-in CI workflow on an exact Git commit and retain E4 logs.

## SEC Fails-to-Deliver adapter limitations (Data Phase 4)

- SEC FTD quantity is the aggregate net balance outstanding on a settlement date. It is not the number of new fails that arose that day, and fail age cannot be inferred.
- A missing security/date row must not be treated as zero without an explicit security universe. Post-2008 SEC files omit zero aggregate balances.
- FTD can arise from long or short sales for multiple reasons. The data are not evidence of abusive short selling or naked short selling.
- The SEC does not guarantee source accuracy or that its previous-day price matches another closing-price source. `fail_value_proxy_usd` therefore remains an analytical proxy.
- CUSIP and ticker mappings can change. This phase preserves the source identifiers and does not claim point-in-time issuer-master resolution.
- The SEC record-layout page states a 10-character maximum for `SYMBOL`, but official archives contain longer operational-event identifiers and some blank symbols. The adapter preserves those values, applies a defensive 32-character hard limit, and records exception counts rather than silently truncating or dropping rows.
- Two April 2021 source rows contain an unescaped pipe inside the 30-character description field. The adapter reconstructs that description only for the unambiguous seven-field pattern and records each repair in the manifest.
- Official archives use both `.txt` and extensionless member names. Two frozen archives have separately allowlisted member-name exceptions; the actual member and its contract class are recorded in the manifest.
- SEC publishes the files twice monthly and does not publish cryptographic checksums. Phase 4 restricts downloads to fixed HTTPS URLs on `www.sec.gov`, validates ZIP and record contracts, then records local SHA-256 values for reproducibility.
- SEC may temporarily limit an IP address or reject an undeclared automated client. A 403 response is not retried; after the SEC cooling period, rerunning with a declared contact reuses every valid archive already retained locally.
