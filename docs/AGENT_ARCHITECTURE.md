# Agent architecture

Written for: the project team.

This document covers the agent layer that sits on top of the encoder. It answers one question:

> Can an agent improve its decisions and guidance from interaction, outcomes, and feedback,
> without continuously retraining its underlying foundation model?

Everything here exists to make that question answerable. That is the filter used below to decide
what to build.

## 1. What we adopt, and what we leave out

The proposed design had five layers and three domains. Most of it is genuinely useful; some of it
would add structure without adding evidence. Being explicit about which is which:

| Proposed | Decision | Why |
|---|---|---|
| Experience store (state, action, outcome, reward) | **Adopt** | This *is* the experiment. Without it there is nothing to learn from and nothing to measure. |
| Feedback → learning system → updated policy | **Adopt** | The loop the research question names. |
| User-specific memory changing the answer | **Adopt** | Distinguishes a learning agent from a predictor. Also gives us the per-user isolation test. |
| Knowledge retrieved, not memorised | **Adopt** | Load-bearing: it is how the policy improves while the encoder stays frozen. |
| Simulated historical environment | **Adopt** | Required. The agent never trades real money. |
| Frozen foundation model, adapters on top | **Adopt** | The "without retraining" half of the question. |
| Generic core → domain adapter → finance / education / healthcare | **Reduce to one seam** | We keep the state/action/reward interface domain-agnostic and implement finance only. Two unused adapters are two untested code paths, and a second domain proves nothing we cannot claim from an interface. |
| Agent Interface and Context Manager as separate layers | **Fold in** | Real work, but plumbing. They become one `Context` object assembled per request, not two layers. |
| Education and healthcare domains | **Drop** | No data, no evaluator, no way to test. Mentioning them as future work costs nothing; building them costs the timeline. |

Dropping these is a scope decision, not a claim that they are wrong. If the finance arm works, the
interface is where a second domain would attach.

## 2. The separation that matters

Four things must stay separate, because the experiment collapses if any two are entangled:

```
Knowledge   frozen encoder + retrieved documents    never updated by feedback
Reasoning   how a recommendation is assembled       code, not learned weights
Policy      what to recommend, given state          the only thing that learns
Memory      who this user is, what happened before  per-user, never shared
```

The reason is direct. If feedback were allowed to change the encoder, any improvement could be
explained by "we fine-tuned the model on more data" — which is the thing we are trying to show is
unnecessary. Keeping the encoder frozen makes the improvement attributable to the policy and the
memory, and nothing else.

Concretely: **the encoder's parameters are loaded once and never appear in an optimizer.** That is
a testable statement, and it gets a test.

## 3. Request path

```
user request
   │
   ├── user memory        risk tolerance, horizon, portfolio, past interactions
   ├── retrieved context  documents from the corpus, selected for this request
   └── market state        features from the historical window being simulated
   │
   ▼
Context  ──▶  frozen encoder  ──▶  state vector  ──▶  policy head  ──▶  action
                                                                          │
                                                       recommendation ◀───┘
                                                                          │
                                simulated market + user feedback  ◀───────┘
                                                                          │
                                                        experience store ◀┘
                                                                          │
                                                   policy / memory update ◀
```

Only the last arrow writes to parameters, and it writes only to the policy head, the adapters, and
the user's memory.

## 4. Experience store

One record per interaction. This schema is the contract between the agent and every learning
method we compare, so it is fixed before any of them are written.

| Field | Meaning | Why it is here |
|---|---|---|
| `interaction_id` | Unique id | Deduplication, and joining feedback that arrives later |
| `user_id` | Which user | Per-user isolation tests |
| `as_of` | Simulated timestamp | Leakage checks. Every feature must be derivable from data before this instant |
| `state` | Encoded state vector | What the policy actually saw |
| `state_sources` | Which documents and windows produced it | Reproducibility, and auditing for look-ahead |
| `action` | The recommendation made | — |
| `action_probability` | Probability under the policy that chose it | Required for off-policy correction and for a REINFORCE baseline |
| `outcome` | Realised market result over the horizon | The environment's reward signal |
| `feedback` | User's response, when given | The human signal; often absent |
| `reward` | Scalar combining outcome and feedback | Written once, by one function, so the definition cannot drift |
| `policy_version` | Which policy produced the action | Lets us attribute improvement to a specific update |

Two rules on this table:

- `as_of` is not decoration. A record whose `state` used data after `as_of` is a leak, and a leaked
  record makes every number computed from it meaningless. This gets an automated check.
- `reward` is derived, never hand-written per call site. One function, one definition.

## 5. Per-user memory

A user is a risk tolerance, a horizon, a portfolio, a set of goals, and a history. The same market
state must produce different recommendations for a retiree and for a 25-year-old — otherwise the
"adapts to each user" claim is empty.

Implementation: a small per-user adapter over the frozen encoder plus a user profile vector, and
that user's slice of the experience store. Users never share adapter parameters.

This yields the sharpest test in the project: **swap two users' memories and the recommendations
must swap too.** If they do not, the personalisation is cosmetic.

## 6. Simulated historical environment

The agent never places a real order. It acts inside a replay of history:

- The environment exposes only data with a timestamp at or before `as_of`.
- Advancing time reveals the next period, which produces the outcome for the previous action.
- Splits are walk-forward with purging and an embargo, so a training window never touches the
  evaluation window through overlapping return horizons.
- Costs and slippage are modelled, because a policy that ignores them looks profitable and is not.

The environment is also where the honest baselines live. Any claimed improvement is measured
against buy-and-hold and against a fixed non-learning policy. If the learning agent does not beat
those, we report that.

## 7. What counts as "the agent improved"

The headline result is not accuracy and not return. It is a curve: performance as a function of
accumulated interactions, with the encoder frozen throughout.

| Arm | What it isolates |
|---|---|
| A0 frozen policy, no learning | Control. Everything is measured against this |
| A1 learns from random queries | Does interaction volume alone help? |
| A2 learns from uncertainty-selected queries | Does *choosing* what to ask help beyond volume? |
| A3 reward model + policy gradient | Does learning from preferences help? |
| A4 direct preference optimisation | Same signal, simpler method — a fair comparison |
| A5 no replay, no EWC | Ablation: does the agent forget without them? |

The claim we can make if this works is narrow and worth stating precisely: *a frozen foundation
encoder plus a learned policy, per-user memory, and retrieved knowledge improves its
recommendations with interaction, and the improvement does not come from updating the encoder.*

The claim we cannot make: that it beats the market, that it generalises to domains we did not
build, or that it would work at production scale. `docs/RESEARCH.md` §4 keeps that list.

## 8. Foundation encoder: what to do about the library change

**Status: proposal, not adopted.** Nothing in this section has been implemented. The import ban on
TensorFlow and Keras stays in force until it is explicitly lifted, and `selfagent/` is unchanged.

TensorFlow and Keras being permitted would change what the from-scratch NumPy stack is for.

The recommendation is to **make the encoder swappable and keep both**:

- The research question is about the policy and memory layers. The encoder is a fixed input to it.
  Whether that encoder was hand-written or loaded from a checkpoint does not change the finding.
- A pretrained encoder largely dissolves the corpus problem recorded as R11 in `docs/RESEARCH.md`.
  We measured ~3 tokens per parameter against BERT-base's ~30; a pretrained encoder starts from
  the other side of that gap.
- The from-scratch stack is 114 passing tests, including 44 finite-difference gradient checks. It
  is the strongest evidence in the repository that the team understands the architecture rather
  than importing it. Deleting that to save a directory would be a bad trade.

So: one `Encoder` interface, two implementations behind it — the tested NumPy stack, and a
pretrained checkpoint. The experiment runs against whichever is loaded, and reports which one it
used. This also makes an ablation available for free: *does the conclusion hold for both encoders?*
If it does, the finding is about the agent design and not about one model.

Cost of adopting it, so the decision is priced: `tests/test_import_guard.py` lists `tensorflow`,
`keras`, and `transformers` in `BANNED_EVERYWHERE`. Lifting the ban means narrowing that test to
the thing that would still matter — that `selfagent/` core modules import no deep learning
framework, so the from-scratch path stays genuinely independent — and confining the pretrained
encoder to its own module. Until the ban is lifted, that test stays exactly as it is.
