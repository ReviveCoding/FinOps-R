# FinOps-R 0.3.0 final iterative verification report

Verification date: 2026-07-20 (America/New_York)  
Verdict: **LOCAL SOFTWARE PASS / RELEASE CONDITIONALLY QUALIFIED / C5 HOLD**

## Method

Every round restarted from the complete candidate. A discovered defect was first reproduced by a regression test, then fixed, and then the source, package and runtime gates were repeated. “PASS” means no further defect was found in the exercised Linux/CPython 3.12 scope. It is not a proof over every input, platform or adversarial environment.

## Defects found and closed in the latest rounds

| Area | Reproduced defect | Resolution |
|---|---|---|
| Typed contracts | Booleans and unknown fields could be coerced or ignored in numeric configuration and case fields | Strict, finite, extra-forbidden Pydantic base contract and regression cases |
| Generated data | Fractional count values, absent required splits and missing training labels could survive early validation | Integer, split-coverage and full training-label-universe checks |
| Review capacity | A deployment threshold fitted on tied scores could review every tied online case and exceed capacity | Strict online cutoff; boundary ties are conservatively excluded |
| ISO source integrity | The source ZIP could be replaced between inspection and staging | Bounded private snapshot; inspection, parsing, hashing and staging use the same bytes |
| ISO identifiers | Blank identifiers and duplicate `EndToEndId` values were accepted | Trimmed non-empty identity contract and uniqueness check |
| Archive/filesystem safety | FIFO/device-like ZIP members and special artifact-tree objects were accepted | Regular-file/directory-only inventory checks |
| Production bundle | Valid checksums could coexist with semantically inconsistent flags, classes, policy or cross-file metadata | Exact schema, component and cross-artifact consistency validation |
| Model-load race | Verified files were reopened by path, allowing replacement between verification and deserialization | In-memory byte snapshots rechecked against the verified manifest before deserialization |
| Windows API smoke | Windows used hard termination but required a zero exit status | New process group plus `CTRL_BREAK_EVENT`; POSIX retains `SIGINT` |
| Output override | An external `--output-root` still wrote `latest.json` into the configuration repository | Release pointers now follow the override publication root |

Earlier v0.3.0 rounds also fixed run-ID collision, publication cleanup atomicity, portable ZIP/artifact paths, Ctrl+C exit 130, negative seed/entity bounds, non-finite ISO amounts and malformed registry handling.

## Final retained evidence

| Gate | Final result |
|---|---:|
| Full source suite with branch coverage | 70 passed, 36 subtests, 0 failed/skipped/xfailed |
| Branch coverage | 86.81% (gate 80%) |
| Consecutive complete source passes | 2 |
| Ruff lint / format | PASS / PASS |
| mypy | 17 source files, 0 issues |
| Wheel / sdist build | PASS / PASS |
| Wheel / sdist inventory | 24 / 80 entries |
| Fresh network-resolved wheel install | PASS, latest allowed dependency versions |
| Installed-wheel full suite | 70 passed, 36 subtests |
| Fresh network-resolved sdist install | PASS |
| `pip check` | PASS in wheel and sdist environments |
| Installed-wheel E2E | 2 × 1,200 cases, PASS |
| Published artifact verification | 59/59 files in both runs |
| TCP API lifecycle | 3 clean cycles, 96 concurrent health requests, 0 errors |
| Attached ISO snapshot | 5,000 paired rows, schema/identity/time/amount checks PASS |
| Attached FUNSD ZIP | 824 members, CRC and portable-path audit PASS |
| Locked dependency vulnerability audit | No known vulnerabilities found |

The two smoke runs produced distinct UUID run IDs and byte-identical release decisions. The decision was intentionally `HOLD`: C5 failed four locked promotion gates and C1 remained the production fallback. A negative model result is not rewritten into a superiority claim.

An attempted unconstrained parallel E2E run was interrupted after host CPU oversubscription made training abnormally slow. The bounded single-thread rerun completed normally twice. The interrupted attempt is not counted as a pass or product failure.

## Web-aligned implementation basis

- PyPA documents standards-based wheel and source-distribution build/install flows: <https://packaging.python.org/en/latest/flow/>.
- Python documents `os.replace` atomicity on a successful same-filesystem replacement: <https://docs.python.org/3/library/os.html#os.replace>.
- Python documents Windows `CTRL_BREAK_EVENT` delivery to processes created with `CREATE_NEW_PROCESS_GROUP`: <https://docs.python.org/3/library/subprocess.html>.
- GitHub recommends minimum workflow permissions and full-length commit SHA pinning: <https://docs.github.com/en/actions/reference/security/secure-use>.

## Unverified conditions

GitHub-hosted Actions, Windows, Python 3.11 and Docker were not executed on this host. GPU is not required by v0.3.0. Workflow inspection is not hosted execution evidence. Runtime artifacts use joblib/pickle semantics and are trusted-local, hash-verified objects rather than signed or sandboxed objects. Exact versions are locked, but dependency artifacts are not hash-pinned. All model evidence is controlled TradeOps-Sim evidence, not bank-production evidence.

After the final fixes, two complete source passes, an installed-wheel suite, fresh wheel/sdist installations, two E2E runs, artifact reloads, attached-data checks and three API cycles found no new defect in the exercised scope.
