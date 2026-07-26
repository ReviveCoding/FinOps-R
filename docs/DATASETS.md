# Dataset governance

FinOps-R keeps semantically different datasets behind explicit adapters. It does not concatenate unrelated public datasets into one training table.

| Dataset | Role | Used for | Not evidence of |
|---|---|---|---|
| TradeOps-Sim | Core | Case-level root cause, action and release-gate study | Bank production impact |
| Supplied ISO 20022-like snapshot | Proxy | Message pairing and data-contract smoke test | ISO conformance or real payments |
| BPI Challenge 2017 | Proxy | Workflow order, SLA and conformance research | Securities post-trade operations |
| CFPB complaints | Proxy | Narrative triage and retrieval | Representative consumer or bank performance |
| SEC Fails-to-Deliver | Stress | Aggregate external stress regimes | Case-level causes, counterparties or fail age |
| FUNSD | Optional | Document-layout smoke test | Financial evidence extraction |

`src/finops_r/dataset_registry.json` is the packaged machine-readable source of truth. A public dataset may inform a separate proxy or stress benchmark, but only TradeOps-Sim provides the complete case-level experimental contract.

## Stage the supplied ISO snapshot

From the repository root:

```bash
finops-r data verify-zip --archive /path/to/ISO20022_Payments_Analytics_Synthetic_v1.zip
finops-r data stage-iso \
  --archive /path/to/ISO20022_Payments_Analytics_Synthetic_v1.zip \
  --output data/external/iso20022_synthetic_v1
```

The staging command checks CRCs, traversal, symlinks, duplicate names, encrypted members, expansion bounds, exact schemas, one-to-one message pairing, status consistency, rejection-reason rules, timestamps and positive amounts. It then writes a source hash, validation results and chronological split assignments atomically.

The staged directory is ignored by Git. Commit only the registry, adapters and small deterministic fixtures—not redistributable or rolling source data.

## Stage the frozen CFPB narrative snapshot

The CFPB export supplied for Phase 2 contains 1,406,112 complete narrative records followed by
one physically truncated 90-byte record. The Phase 2 installer accepts only the exact known
archive, preserves those 90 bytes under `quarantine/`, and stages the complete prefix with both
the original and repaired hashes recorded. It does not silently skip malformed CSV rows.

The standalone staging command receives the repaired CSV and exact provenance evidence:

```bash
python scripts/stage_cfpb_snapshot.py \
  --source /path/to/repaired-complaints.csv \
  --output data/external/cfpb_complaints_20220514_20260424 \
  --expected-sha256 7e67e619c39baecc75d41091cb7a6d4b9f9e9fd700e0d406c9cc1f3a7156b8e1 \
  --expected-rows 1406112 \
  --source-archive-name CFPB-complaints-20220504-20260514.7z \
  --repair-report /path/to/source_repair.json \
  --quarantine /path/to/truncated_final_record.bin
```

Staging validates the exact 18-column schema, strict UTF-8 and CSV structure, critical non-empty
fields, parseable receive/send dates, causal date order, unique numeric complaint IDs, response
labels and a three-way chronological split. The full narrative CSV stays separate from
TradeOps-Sim and is never joined into its case labels.

## Stage the BPI Challenge 2017 workflow snapshot

Phase 3 accepts only the official `BPI Challenge 2017.xes.gz` published by 4TU.ResearchData:

- DOI: `10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b`
- bytes: `29,658,747`
- MD5: `10b37a2f78e870d78406198403ff13d2`
- declared coverage: applications filed in 2016 and subsequent events through 2017-02-01

The standalone command is:

```bash
python scripts/stage_bpi2017_snapshot.py \
  --source "/path/to/BPI Challenge 2017.xes.gz" \
  --output data/external/bpi_challenge_2017
python scripts/verify_bpi2017_stage.py \
  --stage data/external/bpi_challenge_2017
```

The adapter streams gzip and XML with fixed expansion bounds, requires nondecreasing event time
within each application, preserves application-to-offer relationships, and fails unless the log
contains exactly 31,509 applications, 1,202,267 events, 42,995 offers and 149 originators. It
publishes normalized `applications.csv`, `events.csv.gz`, `offers.csv` and a hashed manifest
atomically. Splits are chronological at application level, so events and offers from one
application can never cross partitions. Raw `org:resource` values are omitted and replaced by
deterministic pseudonyms. This transform is a data-minimization measure, not a claim of
irreversible anonymization.

The workflow benchmark remains separate from TradeOps-Sim labels. It supports process-order,
throughput, rework and multi-offer analysis, but it does not establish performance on securities
post-trade operations or a current financial institution.

## Recommended acquisition

- BPI Challenge 2017: obtain the raw `.xes.gz` from the [official 4TU record](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884). Preserve the source archive and record its hash.
- CFPB: download the official [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) CSV ZIP. Freeze a retrieval date and hash because the database generally changes daily. Its narratives are opt-in and the database is not a representative sample.
- SEC Fails-to-Deliver: select dated snapshots from the [official SEC page](https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data). The quantity is an aggregate outstanding balance, not a daily count; fail age and cause cannot be inferred.
- FUNSD: use the [official repository](https://github.com/guillaumejaume/FUNSD) only for the optional document adapter.

## Deliberately excluded from this repository

Instacart, M5, Criteo uplift, HMDA, EDGAR financial summaries, SEC financial statements and Kleister NDA do not measure this project's workflow or decision contract. They belong to other portfolio projects and are intentionally not wired into FinOps-R.

## SEC Fails-to-Deliver market-settlement stress snapshot (Data Phase 4)

- Official source: https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data
- Frozen source range: January 2020 through June 2026, 156 semi-monthly ZIP archives.
- Raw archive location: `C:/Users/bjw-0/Downloads/Project_Data/FinOps-R-data/raw/sec_ftd_202001_202606/`.
- Published location: `data/external/sec_fails_to_deliver_202001_202606/`.
- Published files: `fails.csv.gz`, `daily_stress.csv`, and `manifest.json`.
- Split contract: settlement-date chronological; train through 2023-12-31, validation during 2024, test from 2025-01-01 onward.
- `fail_value_proxy_usd` equals reported fail shares times the SEC previous-day price when a price is present. It is a stress proxy, not a production outcome label.
- Source archives are retained and independently reconciled by byte count and SHA-256 recorded in `manifest.json`.
- Every archive's trailer record count and total quantity are reconciled against parsed rows before publication.
- SEC source-format exceptions are preserved without truncation. `manifest.json` reports blank symbols, ticker values above the published 10-character limit, repaired unescaped description delimiters, and observed maximum symbol length.
- Automated retrieval requires `FINOPS_R_SEC_USER_AGENT` to identify the application and include a monitored contact email; requests are serialized with a 750 ms inter-request delay.
