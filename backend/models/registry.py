"""Which checkpoint serves, and the measurement that earned it the job.

`advisory_combined.npz` is a default string in four scripts, and `scripts/ask.py` -- the one that
serves rather than measures -- now takes it from here instead: a role points at one artifact, at the
digest of that artifact's bytes, and at the tally `scripts/check_answers.py` produced on the reworded
split. A role with no entry serves nothing, rather than serving the newest file it can find.

Two standards decide a promotion, and both are the margin rule the price head already serves under --
`price.beats`, one standard error -- so nothing here invents arithmetic of its own:

  **S14, faithfulness.** The unsupported-figure rate on the reworded split has to sit below 2% by more
  than the noise in a measurement at 2%, which also stops a record that states fifty figures from
  clearing the bar by measuring almost nothing. It is the rate of *figures*, which is what S14 says;
  the rate of *answers* is recorded beside it because this project quotes that one too.

  **Generation, not loss.** A candidate has to beat the promoted checkpoint's exact match on that same
  split by more than one standard error. A1 and A2 both measured the loss-picked checkpoint generating
  worse -- 17.2% unsupported figures against 4.1%, 6.5 points less exact match -- so loss is not a
  number a promotion can be decided on, and a candidate that only looks better on it is refused.

Both tallies must come from the same dataset and split. A2's gate compared its unseen loss against a
figure from another dataset's unseen split, and no amount of care after the fact could settle it.

The digest is checked again at serving, so an artifact overwritten after its promotion is refused
rather than answering under a record that describes the bytes it replaced.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .price import beats

DEFAULT_PATH = Path("artifacts/registry.json")

# The one role so far. Named because a caller asking for "the model" would get whichever the registry
# happened to hold, and the price head is promoted by its own recorded numbers in `price.load`.
GENERATOR = "generator"

# S14 is judged on the reworded phrasings and nowhere else: the trained split cannot tell a model that
# reads its evidence from one that memorised the wording, and it reads 1.7 points kinder.
STANDARD_SPLIT = "unseen"
FAITHFUL = 0.02

REQUIRED = ("artifact", "digest", "sampler", "measured")
TALLY = ("dataset", "rows", "matched", "answerable", "figures", "unsupported_figures",
         "unsupported_answers")


class Refused(ValueError):
    """Raised when a candidate's own record does not meet the standards."""


def digest(path):
    """What the artifact hashes to, which is what a record is pinned to rather than a file name."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def promoted(path=DEFAULT_PATH):
    """Every role and the entry promoted to it, or `{}` when nothing has been promoted yet."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def failures(entry, incumbent=None):
    """Every reason this entry may not be promoted. Empty means it may."""
    absent = [name for name in REQUIRED if name not in entry]
    if absent:
        return [f"the record states no {', '.join(absent)}"]
    tally = entry["measured"].get(STANDARD_SPLIT)
    if tally is None:
        return [f"nothing was measured on the {STANDARD_SPLIT} split, which is the reworded one"]
    absent = [name for name in TALLY if name not in tally]
    if absent:
        return [f"the {STANDARD_SPLIT} tally states no {', '.join(absent)}"]

    reasons = []
    if not tally["figures"]:
        reasons.append(f"its answers state no figures on {STANDARD_SPLIT}, so S14 is unmeasured")
    else:
        rate = tally["unsupported_figures"] / tally["figures"]
        # The bar clearing the rate, not the reverse: the standard error is the one a measurement at 2%
        # carries, so a record built on too few figures cannot pass by having found none.
        if not beats(FAITHFUL, rate, tally["figures"]):
            reasons.append(f"{rate:.1%} of the {tally['figures']} figures it states on "
                           f"{STANDARD_SPLIT} are unsupported, which S14's {FAITHFUL:.0%} does not "
                           f"clear by more than the measurement's own noise")
    if incumbent:
        reasons += _worse_than(tally, incumbent["measured"][STANDARD_SPLIT], incumbent["artifact"])
    return reasons


def promote(role, artifact, measured, sampler, path=DEFAULT_PATH):
    """Records `artifact` as what `role` serves, or raises `Refused` naming every reason it may not."""
    path = Path(path)
    entry = {"artifact": str(artifact), "digest": digest(artifact), "sampler": sampler,
             "measured": measured,
             "promoted": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    entries = promoted(path)
    reasons = failures(entry, entries.get(role))
    if reasons:
        raise Refused(f"{artifact} is not promoted to {role}: " + "; ".join(reasons))
    entries[role] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entry


def serving(role, path=DEFAULT_PATH):
    """The artifact promoted to `role`, checked against the digest it was promoted under."""
    entry = promoted(path).get(role)
    if entry is None:
        raise LookupError(f"nothing is promoted to {role} in {path}; scripts/promote.py measures a "
                          f"candidate and records what it measured")
    artifact = Path(entry["artifact"])
    if not artifact.exists():
        raise LookupError(f"{role} is promoted to {artifact}, which is no longer on disk")
    found = digest(artifact)
    if found != entry["digest"]:
        raise ValueError(f"{artifact} is not the file promoted to {role}: it hashes to {found} and the "
                         f"record says {entry['digest']}; re-measure it with scripts/promote.py")
    return artifact


def _worse_than(tally, incumbent, name):
    """Whether this candidate's exact match clears the promoted one's by more than one standard error."""
    if (tally["dataset"], tally["rows"]) != (incumbent["dataset"], incumbent["rows"]):
        return [f"it was measured on {tally['rows']} rows of {tally['dataset']} and {name} on "
                f"{incumbent['rows']} of {incumbent['dataset']}, which are not the same rows"]
    match, held = tally["matched"] / tally["rows"], incumbent["matched"] / incumbent["rows"]
    if not beats(match, held, tally["rows"]):
        return [f"it matches {match:.1%} of {STANDARD_SPLIT} against {name}'s {held:.1%}, which is not "
                f"more than the {tally['rows']} rows can distinguish"]
    return []
