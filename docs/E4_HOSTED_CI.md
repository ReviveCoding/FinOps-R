# E4 exact-commit hosted qualification

E4 qualifies one immutable Git commit on this required matrix:

| Runner | Python |
|---|---|
| GitHub-hosted Linux | 3.11 |
| GitHub-hosted Linux | 3.12 |
| GitHub-hosted Windows | 3.11 |
| GitHub-hosted Windows | 3.12 |

Each cell must install the development and orchestration extras, pass dependency
consistency, run the full test suite with at least 80% branch coverage, and independently
verify the locked Stage 4 and Stage 5 artifacts. Only then may it write an
`e4-cell-*.json` record.

For a manual dispatch, provide the full intended commit SHA in `expected_commit`.
The cell writer compares that value with `GITHUB_SHA` and fails closed on any mismatch.
Push and pull-request runs use the event commit automatically.

After the run, download all four `e4-cell-*` artifacts into one directory and verify them:

```shell
python scripts/verify_e4_hosted_evidence.py \
  --evidence-dir PATH_TO_DOWNLOADED_EVIDENCE \
  --expected-commit FULL_40_CHARACTER_COMMIT_SHA \
  --output e4_qualification.json
```

The verifier requires four unique matrix cells from one repository, run ID, and commit.
It also rechecks the locked judgments `model_promotion=HOLD` and
`production_champion=C1`. Workflow inspection or locally fabricated JSON is not E4
evidence.
