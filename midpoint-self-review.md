# Midpoint Self-Review — Rishabh Kapur

**Alias:** rishabkk · **Team:** ACCS — Amazon Payments Services (Seller Wallet / Stanza)
**Mentor:** Sitikanta · **Period covered:** 6 July 2026 – 18 September 2026
**Prepared for:** Midpoint Discussion (manager + mentor, 60 min)

---

## Contents

1. [Scorecard](#1-scorecard)
2. [Complete work log](#2-complete-work-log)
3. [Deliverables in detail](#3-deliverables-in-detail)
4. [Roadblocks — cleared and live](#4-roadblocks--cleared-and-live)
5. [Where I fell short](#5-where-i-fell-short--self-assessed)
6. [Leadership Principles](#6-leadership-principles)
7. [Coachability — feedback received and what changed](#7-coachability--feedback-received-and-what-changed)
8. [What I've learned that isn't in a CR](#8-what-ive-learned-that-isnt-in-a-cr)
9. [Plan for the remainder](#9-plan-for-the-remainder)
10. [Feedback I'm asking for](#10-feedback-im-asking-for)
11. [Artifact index](#11-artifact-index)

---

## 1. Scorecard

**Five project workstreams. Four delivered end to end, one in progress. 13 code reviews merged
across 5 packages, one of them outside my own org. One COE action item closed. Zero production
incidents.**

| # | Workstream | Package | Status | Headline result |
|---|---|---|---|---|
| A | Integration-test reliability — NA + EU payee onboarding | `StanzaPlaywrightUITests` | ✅ **Delivered** | **7/9 → 9/9** (NA), **26 pass / 0 fail** (EU), **28/0/0** after hardening. Manual intervention weekly → quarterly |
| B | AWS Glue migration for FE and NA, incl. proactive alarming and prod validation | `FxAccsTieredPricingServiceCDK` | ✅ **Delivered to prod** | **42,048,310 rows** in 1,682s, 0 throttles, row-for-row provable. **COE-362540 closed** |
| C | Environment-supplied template variables (**cross-org contribution**) | `HorizonAITestCore` | ✅ **Delivered** | **239 → 285 tests** green on CPython310 + CPython312, zero consumer impact |
| D | DevOps Agent source stack migration **DUB → ZAZ** | `FxAccsTieredPricingServiceCDK` | ✅ **Delivered** | Migrated without breaking three data-path stacks — the export race I found and designed around |
| E | Data Pipeline / EMR decommission + Python unit tests for the S3 event handler | `FxAccsTieredPricingServiceCDK` | ✅ **Delivered** | **12 idle stacks removed**, wildcard `datapipeline:*` + `iam:PassRole` on `*` removed from **8 prod Lambdas**. First-ever tests on a handler that had caused a prod outage |
| F | HorizonAI Home suite (NA) + Hydra report integration | `StanzaHorizonAITests` | 🟡 **In progress** | **7/7 verified** on real Hydra/Fargate. "Horizon Console" report tab remaining |

---

## 2. Complete work log

Everything I have done, in order. Onboarding and learning included so the ramp is visible.

### 2.1 Onboarding and ramp — 6–16 July

| Date | Item | Status |
|---|---|---|
| 6 Jul | BT101 | ✅ Done |
| 7 Jul | Onboarding setup — Slack, Zoom, M365, Outlook | ✅ Done |
| 7 Jul | Pipeline stages & deployment architecture | ✅ Done |
| 7 Jul | Kiro / AgentSpaces usage | ✅ Done |
| 8 Jul | BT102 | ✅ Done |
| 8 Jul | Customer Payments 101 (CP Learning Series) | ✅ Done |
| 8 Jul | Cloud Desktop & DevSpaces setup | ✅ Done |
| 8 Jul | Brazil, brazil-build, packages, version sets | ✅ Done |
| 8 Jul | Dependency management — merge conflicts, major version replacement | ✅ Done |
| 9–10 Jul | BT111 — Coral in Java | ⏹ Cancelled (deprioritised for project work) |
| 13 Jul | **Builder Week Day 1** — authored AI-readiness artifacts (`AGENTS.md`, steering files); learned Kiro as a code assistant | ✅ Done |
| 14 Jul | **AWS Cloud Practitioner Essentials** | ✅ Done |
| 16 Jul | **Personal Stacks onboarding** for `AccsFuiouPayStationPluginCDK` — CR-289683361 | ✅ Done |
| 17 Jul | **Built a `unit-test-coverage` Kiro skill** with another intern — `SKILL.md`, `examples.md`, `reference.md`, deployed under `.kiro/skills/` and validated under the agent | ✅ Done |
| 23 Jul | **Learn AWS Glue** — from zero, ahead of the migration | ✅ Done |

### 2.2 Delivery — 27 July onward

| Completed | Deliverable | CR / artifact | Result |
|---|---|---|---|
| **27 Jul** | Payee limit & approved-payee pool depletion — **NA** resolution | [CR-291492143](https://code.amazon.com/reviews/CR-291492143) · SIM D332290620 | Pass rate **7/9 → 9/9**, 0 failed. Manual intervention **weekly → quarterly** |
| **30 Jul** | Payee tests — **EU** 50-payee cap + shared-session login collision | [CR-292612012](https://code.amazon.com/reviews/CR-292612012), [CR-293211042](https://code.amazon.com/reviews/CR-293211042) | **26 passed / 0 failed / 184 skipped**, zero post-login timeouts, verified over two consecutive runs |
| **3 Aug** | Environment-supplied template variables for HorizonAI test URLs — **cross-org** | [CR-294124372](https://code.amazon.com/reviews/CR-294124372) · commit `0669e03` | **239 → 285 tests** green on CPython310 + CPython312, mypy clean, ACCB unaffected. Caught a latent ordering bug pre-merge |
| **4 Aug** | EU integration-test hardening — scoped locators, readiness races, error propagation | [CR-294391747](https://code.amazon.com/reviews/CR-294391747) · commit `77f7e84` · 4 files, +126/−27 | Run 6: **28 passed / 0 failed / 0 skipped**. `brazil-build release` SUCCEEDED, AutoSDE **0 findings** |
| **19 Aug** | Proactive alarming for the Glue cutover — Lambda, DynamoDB, Glue + missed-run detection | [CR-297674889](https://code.amazon.com/reviews/CR-297674889) | Tests **35 → 106** across 12 suites, coverage **98.83% → 99.01%**, 4 new/touched files at 100% |
| **20 Aug** | AWS Glue onboarding for **FE and NA** — infra (switch off), then prod cutover | [CR-294675095](https://code.amazon.com/reviews/CR-294675095), [CR-298549826](https://code.amazon.com/reviews/CR-298549826) | NA first live run: **42,048,310 rows** in 1,682s, **0 throttles**, 75/75 keys exact-match, 0 malformed rows |
| **26 Aug** | MCM rollout for FE/NA | [MCM-156778182](https://mcm.amazon.com/cms/MCM-156778182) | **COE-362540 closed.** Alarms 38 → 26, SNS topics 28 → 14, subscriptions 28 → 14 |
| **2 Sep** | HorizonAI Home-NA suite — port of all 7 `HomePageTest.spec.ts` scenarios, realm-split structure | [CR-298129288](https://code.amazon.com/reviews/CR-298129288), [CR-301521524](https://code.amazon.com/reviews/CR-301521524) · SIM D486238649 | **5/7 genuinely passing (6/7 reported, one false) → 7/7 verified.** Validated on real Hydra/Fargate under SigV4 |
| **4 Sep** | Integration tests for NA and EU in the `stanzaHomeAssets` pipeline | — | Delivered |
| **Sep** | **DevOps Agent source stack migration DUB → ZAZ** | CR-296492313 (prerequisite) + migration CRs | Delivered. Prerequisite was byte-identical across **all 105 synthesized templates**, all **10 analyzers PASS**, Coverlay **98.85%** |
| **Sep** | **Data Pipeline / EMR decommission** | 4 CRs (FR-1..FR-9 / NFR-1..NFR-9) | **12 idle stacks removed**; dead Lambda branch removed; wildcard `datapipeline:*` + `iam:PassRole` on `resources: ['*']` removed from **all 8 prod handler Lambdas** |
| **Sep** | Python unit tests for `lambda-handler/s3_file_event_lambda.py` | Part of the decommission set | First tests ever on a 47-line file running in all 8 prod handler Lambdas. Import-contract tests **provably fail** against the commit that caused the outage |

### 2.3 In progress

| Item | Status | Note |
|---|---|---|
| Integrate HorizonAI test results with the Hydra test report — "Horizon Console" tab | 🟡 In progress | ToD discovers tabs dynamically from the TIM vault key listing, so one `index.html` under `HYDRA_TEST_ARTIFACT_UPLOADER_PATH` creates the tab. Trace opens in a CSS `:target` modal with **no JavaScript**; artifacts are snapshotted at run time because a static TIM-vault page cannot authenticate later |
| EU and UK realms for the HorizonAI Home suite | 🟡 Ready to start | **7 lines each** by design — the realm-split structure was built for exactly this |

---

## 3. Deliverables in detail

### A. Integration-test reliability — `StanzaPlaywrightUITests`

**What I was handed.** The payee-onboarding suites needed a human to log in and delete payees every
1–2 weeks, and 7 of 9 NA `PayeeManagement` tests failed intermittently. Worse, the tests that
happened to pass gave false confidence, because failure was timing-dependent on the ORCA approval
cycle rather than on the product.

**NA — CR-291492143.** Diagnosed that the 50-payee UI cap and the "needs an approved payee"
precondition are **fundamentally at odds on a single account**: addition tests consume slots,
management tests consume approved payees, and ORCA approval takes 24–72h. No amount of cleanup
tuning fixes that. Split it into a dual-account strategy with per-role pre-test cleanup, so the
addition account self-recycles (pending → approved → deleted → slots freed) and the management
account self-replenishes.

- Pass rate **7/9 → 9/9**, 0 failed
- Manual intervention **weekly → quarterly** — only if the management pool depletes after 15+
  consecutive days without ORCA approvals
- Tests now degrade gracefully with an actionable skip message instead of hard-failing when a
  precondition is temporarily unavailable
- Serial execution prevents same-account parallel contention; a `finally`-based cleanup eliminated a
  secondary modal-overlay timeout
- Backward compatible — EU/UK stayed on the legacy code path

**EU — CR-292612012 / CR-293211042.** The EU suite looked like the same capacity problem. It wasn't.
The actual blocking failure was authentication: `loginAndSaveState` cached one session in a single
shared `src/state.json` and short-circuited with *"Already logged in. Skipping login."* on **any**
non-empty file — ignoring which account was requested. Whichever account logged in first owned the
session and every other account rode it. The Playwright config compounded it by preloading the same
file into every context. Every EU test — including the one that used to pass — died identically at
the first post-login selector, so nothing reached any add or delete logic.

- Sharded the 8 EU countries deterministically across 4 provisioned accounts (~6 adds each against a
  50 cap), each in its own `test.describe` with `mode: 'serial'`
- Replaced the shared file with a **per-account in-memory cookie cache**, so a session is reused
  within a worker but never across accounts, and never survives a run
- Restored EU's management `beforeEach` (removed by the NA refactor) behind a `wireOwnBeforeEach`
  flag — default true for EU/UK, NA opts out
- **26 passed / 0 failed / 184 skipped**, zero post-login timeouts, verified across two consecutive
  runs. Logs confirm genuinely separate accounts with distinct per-account payee counts
- `brazil-build release` green, 0 tsc diagnostics. `src/state.json`, `playwright.config.ts` and
  `.gitignore` deliberately untouched

**EU hardening — CR-294391747** (commit `77f7e84`, 4 files, +126/−27, test-only). Three residual
defects: an unscoped `kat-alert` locator matching the success-page banner and the site header
(a Playwright strict-mode violation); readiness checks racing a spinner; and `openRecipient`
swallowing its own failure. Fixed with a scoped XPath, two purpose-built readiness helpers keyed on
real container elements rather than timeouts, and proper error propagation. Also made session reuse
**self-healing** — a stale cookie set is now detected, evicted from the cache, and re-authenticated,
so a retry can recover instead of inheriting the same bad session.

- Run 6: **28 passed / 0 failed / 0 skipped**; `brazil-build release` SUCCEEDED; AutoSDE **0 findings**

### B. AWS Glue migration for FE and NA — the largest deliverable

**Why it mattered.** NA and FE were the last two regions still loading cross-currency volume CSVs
into DynamoDB via **AWS Data Pipeline + EMR/Hive**, a deprecated service closed to new customers and
new regions. Migration was a COE action item (**COE-362540**), not optional cleanup. NA is the
highest-volume region at ~42M rows per monthly load, the data feeds downstream financial services,
and `TRAFFIC_CUTOVER_SWITCH` is strictly binary per region — no dual-run, no percentage dial-up.

**How I shaped it** — CR-294675095 (infra), CR-298549826 (cutover), MCM-156778182:

- Design and rollout plan up front with explicit **M1–M5 gates**, sequencing FE first as the
  lower-risk proving ground and giving NA a longer soak as the highest-volume region
- Rather than adding two more region conditionals, moved Glue behaviour into the existing per-region
  config as a `glue` block (`deployStack`, `lambdaJobAlias`, `lambdaJobRegion`, `cutover`, `workers`,
  `timeout`) — **one source of truth per region**, replacing the hardcoded `AwsRegion.ZAZ` stack gate
  and both inline DUB blocks. Made sizing props optional with `?? 5` / `?? 120` fallbacks so ES and
  ZAZ synthesized **byte-identical CloudFormation**
- Split into two reversible CRs: infrastructure with `cutover: false` (jobs deployed, zero traffic
  moved), then a one-line flip per region, with Data Pipeline kept warm so rollback stayed a
  single-line revert
- **Sized from measurement, not inheritance** — NA at 10 workers for its 2.1 GB monthly drop, FE at 5,
  reasoned from the job being write-bound at ~70% of table write capacity, so workers past that
  ceiling buy DPU cost and throttling risk rather than throughput. Asserted beta/prod sizing parity in
  a test, so a green beta run means something for prod
- Added `validateGlueRouting` so a wrong `lambdaJobAlias` fails **synth** instead of failing silently
  at runtime after cutover
- Validated in beta FE end-to-end against a real deployment: 5,000-row run SUCCEEDED in 54s, verified
  the one transformation the script performs, then reverted the switch and deleted the test object

**Prod validation — where I went past what was asked.** The MCM required a spot-check of one
customer. I ran:

- A forward check of **75 keys sampled five different ways**, including random byte-offset range
  reads across all 2.1 GB without downloading the file, and deliberately **non-zero-volume** samples,
  because most rows are `0.000000000000` and a sample of zeros cannot detect a shifted value
- A full-file pass over all **42,048,310 rows** and a 400-item reverse scan
- A completeness proof nobody asked for: each row is ~50 bytes and costs exactly 1 write unit, so
  metered `ConsumedWriteCapacityUnits` for the day should equal total rows.
  **42,065,095 vs 42,065,095** — matched to the digit
- Proof that the daily file is a full subset of the monthly file with identical values
  (**16,785 of 16,785**), establishing that the two same-day jobs cannot conflict regardless of write
  order — and I told the team the daily load is therefore redundant on the 26th rather than quietly
  leaving it
- FE day-two validation: **879/879 row parity** via 9 `BatchGetItem` calls, deliberately avoiding a
  scan of a 5.2M-item prod table; **0 identity mismatches**, **0 throttles over 52 hours**
- **Every AWS call in every validation was read-only**

**Results:**

- NA's first live monthly run: **1,682s for 42M rows** on 10 workers (~25,000 rows/sec), zero
  DynamoDB throttles, 75/75 sampled keys exact-match, 0 malformed rows
- Full 19-significant-digit precision preserved (`1209880.424589950474` round-trips
  character-for-character), which documented **why** `disbursementVolume` must stay a string — a float
  would silently truncate it
- Operational surface shrank: **38 alarms → 26, 28 SNS topics → 14, 28 subscriptions → 14**, and no
  more EMR clusters, Hive scripts, bootstrap scripts, or per-run VPC subnet lookups
- **No outage, no data loss, no customer impact.** COE-362540 closed
- Left a written engineering record: measured thresholds (daily 1–3 min, investigate >8 min; monthly
  20–40 min, investigate >60 min), the account/profile map, the seven environment traps that cost me
  time, and the exact commands to repeat every check. Also corrected the runbook's monthly trigger
  rule, because 131s of that latency sits in S3 event delivery *before* our Lambda

**Monitoring shipped before the cutover, not after — CR-297674889.** Before flipping the
highest-volume region I found four ways the load could fail with nobody finding out. The existing 16
Lambda alarms were **mathematically unable to fire** — `evaluationPeriods: 5` over 5-minute periods
needs 25 minutes of continuous errors from a handler invoked once a day by an S3 event. All 16 also
had **zero AlarmActions**, so they notified nobody, and their names derived from CFN-generated
function names carrying a random suffix that changes on every Lambda replacement. Nothing at all
watched DynamoDB write throttles, Glue job failures, or a Glue job that simply stopped running. A
green dashboard over an unmonitored pipeline, right before the riskiest deployment of the project.

- Made `alarmName` a **required** prop fed from a new `lib/common/alarm-names.ts`, so omitting it is a
  compile error rather than a silent regression
- Manufactured a Glue failure metric via EventBridge on *Glue Job State Change*, deliberately chosen
  over a log metric filter on the script's own error line — because an OOM, timeout or driver crash
  never reaches the exception handler, and those are exactly what a write-bound job hits first
- Handled the failure mode the failure alarms structurally cannot see: a job that never starts emits
  no event, publishes no datapoint, and leaves every alarm at OK while the table goes stale. Noticed
  the legacy monitor being replaced was named `…DataPipelineJobSkipped`, so the old system had covered
  this and the new one would have **regressed** it
- Worked around a hard CloudWatch constraint rather than accepting a gap:
  `Period × EvaluationPeriods ≤ 86400` means an absence-based alarm can't express a window longer than
  24h. Encoded staleness in the metric **value** — `SecondsSinceLastSuccessfulRun` — moving the window
  out of alarm configuration entirely, and read it from Glue's own run history so it covers runs
  predating the alarm
- **Gated the missed-run alarms on the same switch as the traffic**, so turning cutover on turns on
  the alarm proving the cutover took, with no second step to forget. Added a sweep test asserting
  *every* alarm in the stack has an action, including ones built inside the `if`
- Thresholds from measured baselines: confirmed prod FE, NA and ES each published zero
  `WriteThrottleEvents` over the 30 days to 17 Aug on ~0.6k/15k/3k WCU daily. Predicted in advance
  that NA's first Glue run was the one place it might fire, and argued why that would be capacity
  tuning rather than an outage. The live run came in at **zero throttles**
- Scoped IAM tightly and made it enforced: `glue:GetJobRuns` pinned to the stack's two job ARNs, with
  a test asserting the probe role holds exactly those two actions and specifically **no**
  `glue:StartJobRun`, so a probe bug cannot double-load the table
- Verified read-only across all six accounts before shipping, including that all 12 Glue jobs exist
  under exactly the names the `JobName` dimension uses — the assumption whose failure would have left
  the alarm at OK forever
- Tests **35 → 106** across 12 suites, coverage **98.83% → 99.01%**, four new/touched files at 100%

### C. Cross-org contribution — `HorizonAITestCore` (CR-294124372, commit `0669e03`)

Every HorizonAI test hardcoded its environment URL — fourteen occurrences across seven tests in the
home suite. Retargeting the suite from integ to beta or prod meant editing every file, and shared
templates couldn't be reused because each one carried a hardcoded host. My mentor asked me to make
the URL come from outside the test files.

- Traced the full request path first — HorizonAITestCore → API Gateway → HorizonAIBackendLambda →
  HorizonAIAgentCore — to establish where substitution belonged and confirm nothing downstream needed
  to know about placeholders
- Worked through **three candidate designs with my mentor** before implementing. Template-holds-the-URL
  was rejected because it pins an environment into a supposedly shared file; per-test declaration was
  rejected because seven tests means seven declarations. Settled on environment-supplied variables,
  matching the pattern `config.py` already used for six other settings
- Implemented a generic `HORIZONAI_VAR_<NAME>` → `{{<name>}}` mapping rather than a single-purpose
  `HORIZONAI_BASE_URL`, so future needs cost a config line rather than another CR. Extended
  substitution from `instructions` to `starting_url` and `success_criteria`, and added a `variables:`
  block that merges template → test with the test winning
- **Found a latent ordering bug while implementing.** Currency derivation ran against the raw
  `starting_url`, so any test written as `{{base_url}}/dp/B123` would have found no hostname and
  **silently stopped resolving** `{{marketplace_currency}}` — producing literal
  `{{marketplace_currency}}` text in browser instructions with no error, and breaking ACCB-UIAITests
  templates for whoever adopted the feature first. Restructured resolution so `starting_url` resolves
  before derivation reads it
- Added fail-fast guards for unresolved placeholders and scheme-less URLs, each naming the missing
  variable, so a misconfiguration surfaces in seconds instead of as a fifteen-minute browser timeout.
  Chose to log variable **names but never values**, since a `variables:` block can hold credentials
- **239 → 285 tests**, green on CPython310 and CPython312, mypy clean, fully backward compatible. Ran
  the suite with and without a polluted environment variable to confirm no test reads ambient state —
  that would have behaved differently in Hydra than locally
- For end-to-end proof I ran against the real API (session `456edb05`, fully substituted URL, no
  placeholders anywhere in the payload). That run then failed on a pre-existing MFA issue, so I ran a
  **control against unmodified code** (session `687d6a32`) which failed identically — same error, same
  instruction count, same recorded URL. That separated my change from the pre-existing defect
- Kept the MFA problem out of the CR but filed it properly: reading `mfa_handler.py`,
  `login_handler.py` and `strands_orchestrator.py` produced **three separate defects** worth filing
  against HorizonAIAgentCore, with session evidence and a proposed fix that needs no per-account secret
- Also corrected a standing README error found along the way — `account_id` documented as Required,
  optional in code

**Impact.** Test authors no longer write environment URLs. Fourteen hardcoded hosts collapsed to one
pipeline config value, and shared templates became portable across environments instead of
integ-only. All 239 pre-existing tests pass unmodified and ACCB-UIAITests is unaffected.

This is the work I would point to if asked what I'm most proud of: it was outside my own org, it
required convincing senior developers on another team, and the bug I caught during implementation
would have been invisible in production.

### D. DevOps Agent source stack migration DUB → ZAZ — ✅ delivered

**The judgement call that mattered.** I had already written a two-CR plan. Before executing it I
discovered that `deleteStack` is **asynchronous**, which meant it would race the producer updates
dropping three `AWS::StackId` Outputs — producing `Cannot delete export … in use` and failing three
data-path stacks. Rather than defend my own plan, I **superseded it**, rewrote the sequencing, and
wrote a separate runbook so the execution steps were reviewable independently of the design.

- Also dropped the planned beta MCM after checking 315 team records showed it wasn't required, rather
  than filing it because the template said to
- Required coverage to **strictly increase** via the union of all non-skipped region configs' stacks
  plus `GluePipelineStack`, so the migration could not quietly reduce what was tested
- Prerequisite CR-296492313 (adding `skipDevOpsAgentSource` and `electDevOpsAgentSourceRegions`) was
  **byte-identical across all 105 synthesized templates**, all **10 analyzers PASS**, Coverlay
  **98.85%** — a genuinely zero-behaviour-change change, which is what made the rest safe
- Prod deletion was held until explicitly authorized, per the production-safety rules

**Honest note:** this project slipped its original date, and the reason was a review that sat with
zero approvals for five weeks while I waited instead of chasing it. See §5.

### E. Data Pipeline / EMR decommission — ✅ delivered

Scoped as FR-1..FR-9 / NFR-1..NFR-9 across 4 CRs.

- **12 idle stacks removed**, plus the dead Lambda branch left behind by the Glue cutover
- Removed a wildcard `datapipeline:*` **and** `iam:PassRole` on `resources: ['*']` from **all 8 prod
  handler Lambdas** — a standing over-permission, not just dead infrastructure
- Deletion was blocked by **three CloudFormation export/import couplings**, all released by CR-2, so
  the CRs had to land in a specific order rather than as independent cleanups
- Sequenced so that **revert-a-CR stayed a valid rollback** throughout, deliberately holding stack
  deletion relative to the monthly load rather than deleting as soon as it was possible
- Flagged the `Required<Pick<…>>` cast in my own design as a soft spot for the reviewer to push on,
  rather than hoping it wouldn't come up

**Python unit tests for `lambda-handler/s3_file_event_lambda.py`.** 47 lines, running in all 8 prod
handler Lambdas, **no tests at all** before this. The motivating outage is specific and worth stating
plainly: commit `e187f4d` removed a module-scope `boto3` client as "dead code". It was actually paying
botocore start-up cost during Lambda **init**, where the timeout does not apply. Moved into the
handler, it blew the 3s timeout and **lost the prod EU and ES daily load**; `c2ee51d` reverted it.

- Build constraints forced stdlib `unittest` — the build chroot is `basic` with
  `network-access = blocked` and no pip
- Tests live in `test/lambda-handler/`, not `lambda-handler/`, because
  `lib/stacks/eventHandlerLambda.ts` ships that directory via `Code.fromAsset` — test files there
  would be deployed into the Lambda
- Build wiring is **one line** appended to the existing `test` script, not a new harness
- The validation check I care most about: the import-contract tests **provably fail** against
  `git show e187f4d:lambda-handler/s3_file_event_lambda.py`. A test that passes against the broken
  version would have been worthless
- Found **two real bugs** while writing them — the object key is never URL-decoded, and
  `MaxResults=20` is used with no paging — and **deferred both to a follow-up CR** rather than
  widening the test CR

### F. HorizonAI Home suite (NA) — 🟡 in progress

**CR-298129288 + CR-301521524** (SIM D486238649). Ported **all 7** Home scenarios from
`HomePageTest.spec.ts` into HorizonAI YAML rather than cherry-picking the easy ones.

Three instructions were resolving to the wrong element, and I found them by **reading the agent's
trace screenshots to attribute each failure**, rather than rewording instructions until they went
green:

- A bare `footer` sent the agent to the **page** footer, away from the balance card it had just found
- `"Payments"` (plural) matched the **left nav**, not the balance card's footer button, which reads
  `"Payment"` (singular). I verified the on-screen label against a trace screenshot rather than
  assuming it from the DOM or the Playwright test
- `each non-empty` on five modal fields is unsatisfiable — Seller Wallet renders an unset money field
  as `-`, which the agent correctly reports as empty

**The one I'm most glad I caught:** `balance_card` was **passing while checking the left nav** instead
of the balance card. Fixing only the two red tests would have shipped a test that asserts nothing.
Two of the three fixes closed silent holes rather than visible failures.

- Also found that the Strands orchestrator **does not evaluate `success_criteria`**, so any test
  relying on that field passes unconditionally. Moved every verification inline
- Structured as one realm-agnostic template per scenario plus a **7-line per-realm binding**, so all
  three wording fixes landed as one-line template edits every realm inherits, and EU/UK are 7 lines
  each rather than a second copy of the suite
- Reused the existing frozen test accounts as-is rather than inventing new ones, because each is
  frozen at a specific onboarding stage and swapping them silently changes what the test proves
- **5/7 genuinely passing (6/7 reported, one false) → 7/7 verified, 0 failed.** Validated on **real
  Hydra/Fargate under SigV4** — run `…Hydra-4af2eb98-75Qth4A6lD-1788259875`, 7 passed in 23:50 — not
  just locally, because the local path differs in auth and would have hidden an auth-only failure
- Landed the package as proper `brazilpython` rather than a loose directory of YAML
- Wrote the three failure modes into `AGENTS.md` as standing rules, and documented the one
  intentional coverage reduction inline in the template, so the next person adding a surface doesn't
  rediscover them by watching a test fail

**Remaining:** the "Horizon Console" tab on the Hydra test page, and adding the EU/UK realms.

---

## 4. Roadblocks — cleared and live

### Cleared

| Roadblock | Impact it had | How it was cleared |
|---|---|---|
| **CR-296492313 pending with zero approvals for 5 weeks** | Held the entire DUB→ZAZ critical path and pushed the project past its original date | Escalated to get a reviewer engaged, then landed the migration. **This one was on me — I should have escalated in week one, not week five** |
| **Expired AWS credentials** (`InvalidClientTokenId`) blocking Data Pipeline validation checks | Stalled the §7 verification steps | Correct role identified; checks completed and the decommission shipped |
| **Three CloudFormation export/import couplings** blocking stack deletion | Deletion could not proceed at all in the obvious order | Released by CR-2, with the four CRs sequenced so revert stayed a valid rollback |
| **Async `deleteStack` racing producer updates** | Would have failed three data-path stacks in prod | Found during planning, not execution. Superseded my own plan and re-sequenced |

### Live

| Roadblock | Impact | Owner | What I need |
|---|---|---|---|
| **3 MFA / OTP-seed defects in `HorizonAIAgentCore`** — MFA attempted with no resolved OTP; step-count detection heuristic false-positives; a credential-config failure retried 3× at session scope instead of treated as permanent | Blocks the HorizonAI home suite running fully end to end. Diagnosed with session evidence and a proposed fix needing no per-account secret | Platform team (acc-qa-sdet) | **Sponsorship to get these prioritised.** Not mine to fix, and I've taken it as far as I can alone |
| **Hydra "Horizon Console" tab** — artifacts must be snapshotted at run time, because a static TIM-vault page cannot authenticate later | In progress; slipped its original 28 Aug target | Me | A review turnaround commitment so this doesn't repeat the CR-296492313 pattern |

---

## 5. Where I fell short — self-assessed

The brief says be realistic. These are the four things I would do differently, ordered by what they
cost.

**1. I didn't chase reviews fast enough.** I waited on the reviews for the CRs that detailed my
project instead of taking the initiative to pursue them. CR-296492313 sitting five weeks with zero
approvals is the direct consequence, and it is the reason the DUB→ZAZ project missed its original
date. The fix isn't "ping harder" — it's recognising that a review which hasn't moved in 48 hours is
**my** problem to solve, and that there were obvious ways to solve it (message the reviewer directly,
ask my mentor to help find one, raise it in standup) that I simply wasn't reaching for. When I finally
did escalate, it moved.

**2. I didn't know the manual-intervention side of the test estate, and learned it the hard way.**
That the pipeline needed a human to log in and delete payees every 1–2 weeks was something I
discovered by running into it during testing. Had I asked earlier how the suite was actually
*operated*, rather than only how it was written, I'd have reached the dual-account design a week
sooner.

**3. My early code wasn't clean enough.** My first CRs carried a lot of comments and weren't tightly
written, and I collected a run of `nit: write cleaner code` comments as a result. I've since changed
how I write — the code says what it does, comments explain only *why*, and I do a pass over my own
diff before sending it. Comparing my earliest CRs against CR-297674889 or CR-301521524 is the
clearest visible improvement in my work over this period, and I'd like a reviewer's read on whether
they agree.

**4. I didn't know about follow-up CRs, so my revision counts escalated.** I kept reworking a single
review instead of landing what was agreed and taking the rest as a follow-up. That made reviews
longer for my reviewers than they needed to be — a cost I imposed on other people, not just on
myself. Now visible in the opposite direction: the two handler bugs I found are explicitly deferred
to a follow-up CR.

---

## 6. Leadership Principles

### Strengths

**Dive Deep** — the strongest signal in my work. The Glue write-unit completeness proof
(**42,065,095 = 42,065,095**) was not requested by anyone; I looked for a property that would make
completeness provable rather than sampled. Reading agent trace screenshots to attribute each
HorizonAI failure, instead of rewording until green, is what surfaced the false pass. Running a
control session against unmodified code to separate my change from a pre-existing MFA defect is the
same instinct. On the alarms, I diagnosed *why* they were mathematically unable to fire rather than
adjusting a threshold.

**Insist on the Highest Standards** — I went past the MCM's required one-customer spot-check to a
75-key, five-way, full-file, 42M-row validation. I refused to ship known-broken monitoring alongside
the cutover and fixed three alarm defects the cutover itself exposed. I fixed the test that was
passing for the wrong reason, not only the tests that were red. And I made correctness
**structurally enforced** where I could: `alarmName` as a required prop, `validateGlueRouting`
failing synth, a sweep test blocking any unwired alarm, an IAM test asserting the probe role
specifically lacks `glue:StartJobRun`.

**Ownership** — I wrote the three HorizonAI failure modes into `AGENTS.md`, left the Glue validation
as a written engineering record with measured thresholds and repeatable commands, and corrected the
runbook's monthly trigger rule because 131s of that latency sits in S3 event delivery before our
Lambda. I also corrected a README error and a scenario table that weren't my ticket. The next person
does not redo my work.

**Bias for Action** — the LP that has moved most for me over this period. Concretely: shipping
monitoring **before** the cutover rather than after; splitting the Glue work so a zero-traffic infra
CR could land while the cutover decision was still open; filing the three AgentCore defects instead
of waiting for someone else to hit them; telling the team the daily load is redundant on the 26th
rather than leaving it. Early on I would have waited to be asked.

**Learn and Be Curious** — AWS Glue from zero to a prod migration of a 42M-row load in about five
weeks, plus AWS Cloud Practitioner Essentials, plus Kiro/AgentSpaces and a shared
`unit-test-coverage` skill built with another intern during Builder Week.

**Have Backbone; Disagree and Commit** — I brought three candidate designs for the template-variable
work and argued against two of them. I **superseded my own** DUB→ZAZ plan once I found the async
`deleteStack` race, rather than defending what I'd already written. On the alarms I took reviewer
feedback and removed our Lambda from the failure path entirely, even though my version worked, and
moved 133 lines of Python out of a CDK template literal — taking `glue-pipeline.ts` from 563 to 390
lines.

**Earn Trust** — when I missed my original delivery date I went to my manager with a **new committed
date** rather than a status update, and then finished before it. I flag my own gaps in writing: the
`SuccessfulRequestLatency` query that came back empty because I omitted the `Operation` dimension is
recorded as a self-reported limitation in my own validation doc, and I flagged the `Required<Pick<…>>`
cast in my own design as something the reviewer should push on. I'd rather be the one who names the
weak spot.

### Growth opportunities

**Bias for Action, on the unblocking side.** I'm good at acting on the work. I'm not yet good at
acting on the *people* part of the work. A stalled review, an expired credential, a defect owned by
another team — I've been too willing to treat those as facts of the environment rather than tasks
with my name on them. Same root cause as §5 #1, and the single thing I most want a concrete habit for.

**Deliver Results.** One project missed its original date. I recovered it and beat my revised date,
but the first estimate was wrong — and it was wrong because I didn't price in review latency or the
possibility of discovering something like the manual-intervention requirement. I want to get better
at estimating with the unknowns included.

**Insist on the Highest Standards, applied to my own first draft.** The `nit: write cleaner code`
feedback was fair. I've changed, but the change came from being told rather than from reviewing my own
diff critically before sending it. That's now a habit, but it started as a correction.

**Learn and Be Curious, about process and not only technology.** I picked up Glue, CDK, CloudWatch and
Playwright quickly. I was much slower on the mechanics of *how we work* — follow-up CRs, how suites
are operated day to day, when to escalate a review. Those cost me more time than any technical
unknown did.

---

## 7. Coachability — feedback received and what changed

| Feedback | Source | What I changed |
|---|---|---|
| `nit: write cleaner code`, repeatedly, early on | CR reviewers | Rewrote how I write. Comments now explain *why* only; I do a comment pass over my own diff before sending. Visible from CR-297674889 onward |
| "Make the URL supplied from outside the test files" | Mentor | Went further than asked — a generic `HORIZONAI_VAR_*` mechanism rather than a single `BASE_URL`, so the next variable is a config line, not a CR |
| Three designs discussed before implementation | Mentor | Learned to bring options with stated trade-offs instead of one answer, and to let the reviewer kill the ones that pin an environment into a shared file |
| "Take our Lambda out of the failure path" | Reviewer, CR-297674889 | Rebuilt as rule → alarm instead of four hops, and moved 133 lines of Python out of a CDK template literal into `glue-monitoring/` |
| Use follow-up CRs | Reviewers | Now landing what's agreed and deferring the rest — the two S3-handler bugs are explicitly deferred rather than widening the test CR |
| Review latency is partly the author's job | Learned the hard way, CR-296492313 | Escalation is now something I do at 48 hours, not five weeks. §9 makes it a tracked rule rather than something I remember |

---

## 8. What I've learned that isn't in a CR

- **How to communicate with senior engineers.** Bring the trace, the session ID, the control run —
  not the conclusion. Senior people engage with evidence far more readily than with claims.
- **How to operate in a corporate workspace** and put my own thinking forward inside it.
- **Putting an idea forward is not wrong even when the idea turns out to be inaccurate.** This was the
  biggest shift for me. Two of the three designs I proposed for the template-variable work were
  rejected — and proposing them is *how* we reached the right one. I was holding back early because I
  thought being wrong in front of senior developers was the risk. The actual risk was staying quiet.
- **How to contribute outside my own org.** The HorizonAI work meant working with senior developers on
  another team, to their standards, in their package, without breaking their existing 239 tests.
- **How to recover a missed date honestly.** Not delivering on time was uncomfortable. Going to my
  manager with a new committed date and then beating it taught me that a revised commitment people can
  plan around is worth far more than an optimistic one they can't.

---

## 9. Plan for the remainder

| # | Action | By |
|---|---|---|
| 1 | Finish the **Hydra "Horizon Console"** tab so HorizonAI traces are visible from the Hydra test page | Next 2 weeks |
| 2 | Land the **two deferred S3-handler bugs** (object key URL-decode, `MaxResults` paging) as the follow-up CR | Sept |
| 3 | Get the **3 AgentCore MFA defects** sponsored with the platform team, then run the HorizonAI home suite fully end to end | Needs manager help |
| 4 | Add **EU and UK realms** to the HorizonAI home suite — 7 lines each by design | Oct |
| 5 | **Standing rule, already in effect:** any CR of mine with no reviewer activity in **48h** gets a direct follow-up; at **5 days** it gets raised in standup. Tracked, not remembered | Immediately |
| 6 | Estimate every remaining item **with review latency and unknown-discovery priced in**, and review those estimates with my mentor rather than committing alone | Next 1:1 |
| 7 | Write up the Glue and decommission work as a durable team record, so the legacy-ETL knowledge doesn't leave with me | Before end of internship |

---

## 10. Feedback I'm asking for

**For my manager:**

1. Against a same-level student hire, how does the **scope and volume** of the last ten weeks read to
   you — and where would you have wanted me to go deeper rather than broader?
2. Is my read on the review-chasing gap the right diagnosis, or is there a different root cause you
   see from the outside?
3. Which of the LPs in §6 would you **not** have listed as a strength for me, and why? I'd rather hear
   the disagreement than the agreement.
4. Now that D and E have landed, what's the highest-value thing I could pick up for the remainder?

**For my mentor (Sitikanta):**

1. Where in our design conversations was I still bringing you a decision to make rather than a
   recommendation to challenge?
2. Are my CR descriptions and validation write-ups now at the level you'd expect, or still longer than
   they need to be?
3. What would you have done differently on the DUB→ZAZ sequencing, given the export-race discovery?

**For reviewers and stakeholders:**

1. Has the code-cleanliness feedback actually stopped being needed, or have you just stopped flagging
   it?
2. Is the volume of validation evidence I produce useful to you, or is it more than a reviewer wants to
   read?

---

## 11. Artifact index

**Code reviews (13)**

| CR | Subject |
|---|---|
| CR-289683361 | `AccsFuiouPayStationPluginCDK` — Personal Stacks onboarding |
| CR-291492143 | NA payee limit & approved-payee pool depletion |
| CR-292612012 | EU payee tests — 50-payee cap + shared-session collision |
| CR-293211042 | EU payee tests — follow-on |
| CR-294124372 | `HorizonAITestCore` — environment-supplied template variables (**cross-org**) |
| CR-294391747 | EU integration-test hardening — scoped locators, readiness, error propagation |
| CR-294675095 | Glue infrastructure for FE/NA (cutover off) |
| CR-296492313 | DevOps Agent source — `skipDevOpsAgentSource` prerequisite |
| CR-297674889 | Proactive alarming for the Glue cutover |
| CR-298129288 | `StanzaHorizonAITests` — Home-NA suite + realm-split structure |
| CR-298549826 | Glue prod cutover for FE/NA |
| CR-301521524 | HorizonAI instruction-wording fixes (false pass eliminated) |
| + Data Pipeline decommission set | 4 CRs — stack removal, export decoupling, IAM narrowing, handler unit tests |

**Other artifacts**

- **MCM-156778182** — FE/NA Glue rollout
- **COE-362540** — closed by the Glue migration
- **SIM D332290620** — NA payee limit · **SIM D486238649** — HorizonAI Home suite
- Design docs and runbooks for the Glue rollout, the Data Pipeline decommission, and the DUB→ZAZ
  migration

**Packages touched (5)**
`StanzaPlaywrightUITests` · `StanzaHorizonAITests` · `HorizonAITestCore` ·
`FxAccsTieredPricingServiceCDK` · `AccsFuiouPayStationPluginCDK`
