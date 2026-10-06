#!/usr/bin/env python3
"""Let the generator write, keep only what its own evidence supports, and train it on that.

The third and last way this project has to teach the model that writes. Supervised training gave it one
target per question and the targets are unfaithful -- 56.9% of them state a figure the retrieved passages do
not contain -- so it learned to invent numbers. Preference training ranked two answers drawn from the same
corpus, which cannot express grounding, and two full runs moved nothing. This asks the model for k answers,
scores each against the passages with the rule the server screens with, and fine-tunes on the winners. The
labels are the model's own output and the judgement is a check, so the loop needs no corpus, no reward model
and no second network.

That is RAFT, and the reason for RAFT over PPO or GRPO is the autograd this project is built on: a clipped
surrogate needs an elementwise minimum, which `selfagent.autograd` does not have. It costs nothing to give
up. RAFT is best-of-k followed by ordinary supervised fine-tuning, so it reuses `train_generator.batch_loss`
unchanged, holds one model in memory instead of PPO's four, and is insensitive to how the reward is scaled --
only the ranking inside a group matters.

**Read the group tally before the loss.** `winner` drops a question whose k samples all invented a figure,
because the best of those is still wrong, and drops one whose k samples all scored the same, because then the
pick is not a judgement and the step is just the model copying itself. On this model that second case is
expected to dominate: `learn/rank.py` measured 1.77 distinct answers out of 8 at the served sampler. If the
tally says almost nothing is teachable then sampling is the bottleneck and no reward will fix it, which is
worth knowing before spending a round. `--temperature` is the only knob that changes it.

The winning answer is re-encoded from its text rather than carried as ids, which reuses `answer_rows`
untouched. Measured on 300 held-out answers, 299 survive decode-then-encode unchanged.

**Measured, and it did not pay.** Four rounds of 192 questions x 4 samples from the grounded generator found
57-61 teachable groups a round -- 129-132 drew four samples that all stated no figure and so scored alike --
which is 8 steps of training per round. Round 1 read 65.6% -> 54.9% unsupported on the 200 selection answers,
but on 300 independent rows it scored 63.0% against the frozen model's 53.8%, with 44 supported figures
against 61. That was selection noise, and rounds 3-4 were refused for saying less. What would change it is
more signal per round: groups where the samples actually state figures.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_answers import load_model  # noqa: E402
from rerank import sample_candidates  # noqa: E402
from selfagent import pretrained  # noqa: E402
from selfagent.data import advisory  # noqa: E402
from selfagent.learn.verified import reward, winner  # noqa: E402
from selfagent.models.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P  # noqa: E402
from selfagent.optim import AdamW, clip_grad_norm  # noqa: E402
from selfagent.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID  # noqa: E402
from train_generator import batch_loss, load_split  # noqa: E402
from train_preference import writes  # noqa: E402


def target_row(tokenizer, answer, width):
    """One kept answer as a target row: the markers the decoder was trained on, then padding.

    Built to the dataset's own width so the kept samples go through `batch_loss` as the gold answers did.
    An answer longer than the width is cut and still closed with SEP, because a target that never ends
    teaches the model not to stop.
    """
    ids = list(tokenizer.encode(answer))[: width - 2]
    row = np.full(width, PAD_ID, dtype=np.uint16)
    row[: len(ids) + 2] = [CLS_ID, *ids, SEP_ID]
    return row


def harvest(model, tokenizer, split, chosen, rng, candidates, batch_size, temperature, top_p):
    """(source rows, keep rows, targets, tally) -- the winning samples, and why the rest were dropped.

    The tally is the feasibility number: `taught` is how many of these questions carry a signal at all.
    """
    drawn = sample_candidates(model, tokenizer, split, chosen, rng, candidates, batch_size,
                              temperature, top_p)
    width = split[2].shape[-1]
    rows, targets = [], []
    tally = dict(taught=0, all_invented=0, all_alike=0, distinct=0)
    for offset, row in enumerate(chosen):
        answers = [trial[offset][3] for trial in drawn]
        evidence = drawn[0][offset][1]
        rewards = [reward(answer, evidence, advisory.UNSUPPORTED) for answer in answers]
        tally["distinct"] += len(set(answers)) / len(chosen)
        best = winner(rewards)
        if best is None:
            tally["all_invented" if not max(rewards) else "all_alike"] += 1
            continue
        tally["taught"] += 1
        rows.append(row)
        targets.append(target_row(tokenizer, answers[best], width))
    if not rows:
        return None, None, None, tally
    taken = np.array(rows)
    return split[0][taken], split[1][taken], np.stack(targets), tally


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="generator", help="prefix of the .npy split files")
    parser.add_argument("--checkpoint", default="generator.npz",
                        help="the trained generator this starts from; the pretrained encoder has never "
                             "written an answer to check")
    parser.add_argument("--run-name", default="verified")
    parser.add_argument("--split", default="train", help="where the questions come from")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--rows", type=int, default=256, help="questions sampled per round")
    # Acceptance is 1/k, so k sets both how much exploration each question gets and how much of the
    # sampling is thrown away. RAFT measured 16 and 32 converging in 10-12 rounds against 15-18 for 8.
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="an order of magnitude below the supervised 1e-4: the targets are the "
                             "model's own samples, so a large step amplifies whatever it already does")
    # Exploration is the whole risk here. At the served pair this model draws 1.77 distinct answers out
    # of 8, and a group of identical answers is dropped, so a run at the served temperature may find
    # almost nothing to train on. Raising it is the only lever this architecture has.
    parser.add_argument("--temperature", type=float, default=ANSWER_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=ANSWER_TOP_P)
    parser.add_argument("--eval-rows", type=int, default=200,
                        help="held-out questions the model answers itself at each round, which is what "
                             "selects the checkpoint; 64 was too few to resolve a figure rate")
    parser.add_argument("--max-repeat", type=float, default=0.05,
                        help="a checkpoint repeating more of its own 4-grams than this is not eligible")
    parser.add_argument("--dry-run", action="store_true",
                        help="harvest one round and report the tally without training, which is what "
                             "says whether the loop has any signal to learn from")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer, model, config = load_model(artifacts, args.dataset, args.checkpoint)
    split = load_split(artifacts, args.dataset, args.split)
    validation = load_split(artifacts, args.dataset, "validation")
    rng = np.random.default_rng(config.seed)
    optimizer = AdamW(model.trainable_parameters(), lr=args.learning_rate)

    print(f"starting from {artifacts / args.checkpoint}, {len(split[0]):,} questions in {args.split}")
    print(f"{args.rounds} rounds x {args.rows} questions x {args.candidates} candidates at "
          f"temperature {args.temperature} top-p {args.top_p}, lr {args.learning_rate}")
    before = writes(model, tokenizer, validation, args.eval_rows, args.batch_size, config.seed)
    print(f"frozen writing: {before[0]:.1%} of figures unsupported, {before[1]:.1%} repeated 4-grams, "
          f"{before[2]:.2f} supported figures per answer on {args.eval_rows} answers")

    best, best_round = before[0], 0
    started = time.monotonic()
    for round_number in range(1, args.rounds + 1):
        chosen = rng.permutation(len(split[0]))[: args.rows]
        source, keep, targets, tally = harvest(model, tokenizer, split, chosen, rng, args.candidates,
                                               args.batch_size, args.temperature, args.top_p)
        print(f"\nround {round_number}/{args.rounds}  {tally['taught']}/{len(chosen)} questions "
              f"teachable  {tally['all_alike']} all alike  {tally['all_invented']} all invented  "
              f"{tally['distinct']:.2f} distinct answers of {args.candidates}")
        if source is None:
            raise SystemExit(
                f"no question in round {round_number} produced a group worth training on, out of "
                f"{len(chosen)}: {tally['all_alike']} drew the same answer every time and "
                f"{tally['all_invented']} invented a figure in every sample. The reward is not the "
                f"problem -- the model is not exploring. Raise --temperature or --candidates."
            )
        if args.dry_run:
            print("dry run, so nothing was trained")
            return 0

        model.train()
        losses = []
        for start in range(0, len(source), args.batch_size):
            stop = start + args.batch_size
            optimizer.zero_grad()
            loss = batch_loss(model, source[start:stop], keep[start:stop], targets[start:stop],
                              config.vocab_size)
            loss.backward()
            clip_grad_norm(optimizer.parameters, 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()

        unsupported, repeated, supported = writes(model, tokenizer, validation, args.eval_rows, args.batch_size,
                                       config.seed)
        elapsed = (time.monotonic() - started) / 60
        print(f"  loss {np.mean(losses):.4f} over {len(losses)} steps  "
              f"writing: {unsupported:.1%} of figures unsupported (from {before[0]:.1%})  "
              f"{repeated:.1%} repeated (from {before[1]:.1%})  {supported:.2f} supported figures per answer "
              f"(from {before[2]:.2f})  {elapsed:.0f} min in")
        pretrained.save(artifacts / f"{args.run_name}.npz", model, config)
        if repeated > args.max_repeat:
            print(f"  not eligible: repeating {repeated:.1%} of its own 4-grams, over the "
                  f"{args.max_repeat:.1%} ceiling")
        elif supported < before[2]:
            print(f"  not eligible: {supported:.2f} supported figures per answer, below the frozen "
                  f"{before[2]:.2f} -- a lower rate from saying less is not learning")
        elif unsupported < best:
            best, best_round = unsupported, round_number
            pretrained.save(artifacts / f"{args.run_name}.best.npz", model, config)

    if best_round:
        print(f"\nsaved {artifacts}/{args.run_name}.best.npz from round {best_round}, "
              f"{best:.1%} of figures unsupported against {before[0]:.1%} frozen")
    else:
        print(f"\nno round beat the frozen {before[0]:.1%} unsupported-figure rate, so this run has "
              f"nothing to promote")
    return 0


if __name__ == "__main__":
    sys.exit(main())
