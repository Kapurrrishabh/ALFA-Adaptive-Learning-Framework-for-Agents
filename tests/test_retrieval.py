"""C2: what the retriever is allowed to show, and the two rules that make that honest.

The measurement that decided C2 lives in `scripts/eval_retrieval.py` and needs the built index. What is
pinned here is everything the measurement assumed: a document with no publication date is dropped rather
than dated by its download, the as-of mask is applied before ranking rather than to the ranked list, and
the two arms abstain differently because a BM25 zero means something a cosine of zero does not.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backend.retrieval import (ARMS, HYBRID, LEXICAL, SERVED, VECTOR, Chunk, Document, Hybrid,  # noqa: E402
                               chunks, deduplicate, documents, fuse, load, published_on, save)
from backend.retrieval.index import RRF_K, centred, without  # noqa: E402
from eval_retrieval import queries  # noqa: E402
from selfagent.tokenizer.wordpiece import pretokenize  # noqa: E402

EARLY, LATE = "2020-01-01", "2024-01-01"


class FakeTokenizer:
    """One id per word, so a chunk's token budget reads as its word count."""

    def __init__(self):
        self.ids = {}

    def encode(self, text):
        return [self.ids.setdefault(word, len(self.ids) + 5) for word in text.split()]


def document(tmp_path, text, key="k0", day=EARLY, source="fed_press"):
    path = tmp_path / f"{key}.txt"
    path.write_text(text, encoding="utf-8")
    return Document(key, "https://example.test/x", "public domain", source, day, path)


def corpus(tmp_path, entries):
    """A manifest and its extracted text, laid out the way `dataforge` leaves them."""
    manifest = tmp_path / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
            if entry.pop("extracted", True):
                source = tmp_path / "text" / entry["source"]
                source.mkdir(parents=True, exist_ok=True)
                (source / f"{entry['sha256'][:16]}.txt").write_text("body", encoding="utf-8")
    return manifest, tmp_path / "text"


def entry(name, source="fed_press", sha256="0" * 64, extracted=True):
    return {"source": source, "path": f"data/raw/{source}/{name}", "sha256": sha256,
            "url": f"https://example.test/{name}", "licence": "public domain",
            "fetched_at": "2026-09-22T00:00:00Z", "extracted": extracted}


# --- what may be retrieved at all -------------------------------------------

def test_both_dated_naming_schemes_parse_and_nothing_else_does():
    assert published_on("1770787_2024-02-15_10-K.txt") == "2024-02-15"
    assert published_on("enforcement20260918b.txt") == "2026-09-18"
    assert published_on("monetary-policy-principles.txt") is None


def test_an_undated_document_is_dropped_rather_than_dated_by_its_download(tmp_path):
    """The look-ahead the project forbids. `fetched_at` says 2026 for every file we hold, so dating by
    it would let a release we downloaded today answer a question asked in 2020."""
    manifest, text = corpus(tmp_path, [entry("enforcement20200103a.txt", sha256="a" * 64),
                                       entry("monetary-policy-principles.txt", sha256="b" * 64)])
    found = documents(manifest, text)
    assert [held.key for held in found] == ["a" * 16]
    assert found[0].day == "2020-01-03"


def test_a_document_whose_text_was_never_extracted_is_skipped(tmp_path):
    # An unreadable PDF leaves a manifest row and no cache file. One of 18,000 must not fail the index.
    manifest, text = corpus(tmp_path, [entry("enforcement20200103a.txt", sha256="a" * 64),
                                       entry("enforcement20200104a.txt", sha256="b" * 64,
                                             extracted=False)])
    assert [held.key for held in documents(manifest, text)] == ["a" * 16]


def test_an_undated_source_is_not_silently_indexed_as_empty(tmp_path):
    manifest, text = corpus(tmp_path, [entry("policy.txt", source="wikipedia", sha256="c" * 64)])
    with pytest.raises(ValueError, match="no dated documents"):
        documents(manifest, text, sources=("wikipedia",))


def test_a_missing_manifest_names_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataforge"):
        documents(tmp_path / "absent.jsonl", tmp_path)


# --- cutting a document into passages ---------------------------------------

def test_a_chunk_never_exceeds_the_window_and_overlaps_by_a_whole_sentence(tmp_path):
    """Overlap is what keeps a figure and the thing it measures in one chunk. Carried as a sentence
    because a chunk starting mid-sentence reads as noise to an encoder trained on whole ones."""
    text = " ".join(f"sentence {n} holds four." for n in range(4))
    built = chunks(document(tmp_path, text), FakeTokenizer(), 10, min_tokens=1)
    assert [len(chunk.ids) for chunk in built] == [8, 8, 8]
    assert built[1].text.startswith("sentence 1") and built[0].text.endswith("holds four.")


def test_a_passage_too_short_to_be_a_passage_is_not_indexed(tmp_path):
    assert chunks(document(tmp_path, "a heading."), FakeTokenizer(), 10, min_tokens=4) == []


def test_a_sentence_longer_than_the_window_is_truncated_not_dropped(tmp_path):
    # In a filing that is a run-together list of figures, and its first window still matches a question
    # about it. Dropped instead, the whole document would vanish from the index.
    built = chunks(document(tmp_path, " ".join(["figure"] * 20)), FakeTokenizer(), 10, min_tokens=1)
    assert len(built) == 1 and len(built[0].ids) == 10


def test_a_chunk_inherits_its_documents_key_and_date(tmp_path):
    built = chunks(document(tmp_path, "the committee met and agreed to hold.", key="abc", day=LATE),
                  FakeTokenizer(), 10, min_tokens=1)
    assert built[0].document == "abc" and built[0].day == LATE


def test_a_zero_window_is_an_error_not_an_empty_index(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        chunks(document(tmp_path, "a sentence here."), FakeTokenizer(), 0)


def test_deduplicate_keeps_the_first_copy_which_is_the_earliest_dated():
    """A standing paragraph appears in 43 press releases. Retrieving it five times fills the generator's
    window with one passage, and the copy worth keeping is the one that was published first."""
    first = Chunk("The minutes are published three weeks later.", [], "a", EARLY)
    kept = deduplicate([first, Chunk("the minutes are published three weeks later!", [], "b", LATE),
                        Chunk("Something else entirely.", [], "c", LATE)])
    assert [chunk.document for chunk in kept] == ["a", "c"] and kept[0] is first


# --- the index on disk -------------------------------------------------------

def test_an_index_round_trips_through_disk(tmp_path):
    held = [Chunk("inflation eased in June.", [1, 2], "a", EARLY),
            Chunk("the committee held rates.", [3], "b", LATE)]
    vectors = np.eye(2, dtype=np.float32)
    save(tmp_path / "index.npz", held, vectors)
    loaded, read = load(tmp_path / "index.npz")
    assert [(chunk.text, chunk.document, chunk.day) for chunk in loaded] == [
        (chunk.text, chunk.document, chunk.day) for chunk in held]
    assert np.array_equal(read, vectors)


def test_a_lexical_only_index_saves_and_loads_without_vectors(tmp_path):
    save(tmp_path / "index.npz", [Chunk("inflation eased in June.", [], "a", EARLY)])
    assert load(tmp_path / "index.npz")[1] is None


def test_a_newline_in_a_chunk_is_refused_rather_than_splitting_a_row(tmp_path):
    # The texts are stored as one newline-joined blob, so a newline inside one would load as two chunks
    # and silently shift every document key after it.
    with pytest.raises(ValueError, match="newline"):
        save(tmp_path / "index.npz", [Chunk("two\nlines", [], "a", EARLY)])


# --- searching ---------------------------------------------------------------

def dated_chunks():
    """Five early chunks and five later ones that score higher, so filtering order is observable."""
    return ([Chunk(f"inflation report number {n}", [], f"e{n}", EARLY) for n in range(5)]
            + [Chunk(f"inflation inflation inflation report {n}", [], f"l{n}", LATE) for n in range(5)])


def unit(rows, seed=0):
    vectors = np.random.default_rng(seed).normal(size=(rows, 8))
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def test_the_as_of_mask_runs_before_ranking_not_after(tmp_path):
    """Filtering a ranked top-5 would return nothing here, because the five best chunks are all later
    than the question. Masking the scores instead returns the best five that existed."""
    index = Hybrid(dated_chunks(), pretokenize)
    assert all(index.cite(found).day == LATE for found in index.search("inflation report", 5))
    visible = index.search("inflation report", 5, as_of=EARLY)
    assert len(visible) == 5 and all(index.cite(found).day == EARLY for found in visible)


def test_a_question_older_than_every_chunk_returns_nothing(tmp_path):
    assert Hybrid(dated_chunks(), pretokenize).search("inflation", 5, as_of="2001-01-01") == []


def test_the_lexical_arm_abstains_where_no_query_term_appears():
    """A BM25 zero means the passage holds none of the query's words. Padding the list out to top_k
    would hand the generator text that has nothing to do with the question."""
    assert Hybrid(dated_chunks(), pretokenize).search("zzzz qqqq", 5) == []


def test_the_vector_arm_returns_a_negative_cosine_rather_than_abstaining():
    """A cosine has no value meaning "matched nothing", and after centring half of them are negative.
    So the floor is below -1, and the row pointing the other way is still the third best of three."""
    wanted = unit(1)[0]
    vectors = np.concatenate([unit(2), -wanted[None, :], unit(3, seed=1)]).astype(np.float32)
    index = Hybrid(dated_chunks()[:3] + dated_chunks()[5:8], pretokenize, vectors,
                   lambda _: wanted, centre=False)
    found = index.search("inflation", 5, as_of=EARLY, arm=VECTOR)
    assert len(found) == 3 and found[-1] == 2


def test_a_masked_chunk_is_never_returned_by_the_vector_arm():
    index = Hybrid(dated_chunks(), pretokenize, unit(10), lambda _: unit(1)[0], centre=False)
    assert all(index.cite(found).day == EARLY
               for found in index.search("inflation", 10, as_of=EARLY, arm=VECTOR))


def test_fusion_prefers_a_chunk_both_halves_ranked_second():
    # Why rank fusion and not a weighted sum of scores: agreement is what it rewards, and it needs no
    # scale constant between an unbounded BM25 score and a cosine.
    fused = fuse([[7, 3], [9, 3]], RRF_K)
    assert max(fused, key=lambda index: fused[index]) == 3


def test_the_hybrid_arm_honours_the_date_filter_too():
    index = Hybrid(dated_chunks(), pretokenize, unit(10), lambda _: unit(1)[0])
    found = index.search("inflation report", 5, as_of=EARLY, arm=HYBRID)
    assert found and all(index.cite(result).day == EARLY for result in found)


def test_the_served_arm_is_the_lexical_one_and_needs_no_vectors():
    """C2's gate, answered in the negative: lexical 15.5% recall@5 against hybrid 12.0% and vector
    1.5%. So the default arm must work on an index that was never embedded."""
    assert SERVED == LEXICAL
    assert Hybrid(dated_chunks(), pretokenize).search("inflation report", 3)


def test_asking_for_a_vector_arm_on_a_lexical_index_is_an_error():
    index = Hybrid(dated_chunks(), pretokenize)
    for arm in (VECTOR, HYBRID):
        with pytest.raises(ValueError, match="needs vectors"):
            index.search("inflation", 5, arm=arm)


def test_an_unknown_arm_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="unknown arm"):
        Hybrid(dated_chunks(), pretokenize).search("inflation", 5, arm="semantic")
    assert ARMS == (LEXICAL, VECTOR, HYBRID)


def test_vectors_without_a_way_to_embed_a_query_fail_at_construction():
    # Not at the first vector search, which is when a served question would be the thing that failed.
    with pytest.raises(ValueError, match="must be given together"):
        Hybrid(dated_chunks(), pretokenize, unit(10))


def test_a_vector_count_that_does_not_match_the_chunks_fails_at_construction():
    with pytest.raises(ValueError, match="they must match"):
        Hybrid(dated_chunks(), pretokenize, unit(9), lambda _: unit(1)[0])


def test_a_non_finite_vector_is_caught_at_construction():
    """One NaN row wins every comparison it is in, so it would look like a retrieval result rather than
    a half-written file."""
    vectors = unit(10)
    vectors[4, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Hybrid(dated_chunks(), pretokenize, vectors, lambda _: unit(1)[0], centre=False)


# --- the geometry the vector arm lost to ------------------------------------

def test_centring_removes_the_direction_every_embedding_shares():
    """Measured on the real index: unrelated chunks sit at cosine 0.925 raw and 0.142 centred, and the
    mean embedding has norm 0.942 out of 1. Renormalised after, or a dot product is no longer a cosine."""
    shared = unit(1)[0]
    vectors = unit(20, seed=2) * 0.2 + shared * 3.0
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    moved = without(vectors, centred(vectors))
    off = ~np.eye(len(vectors), dtype=bool)
    assert (vectors @ vectors.T)[off].mean() > 0.9 > (moved @ moved.T)[off].mean()
    assert np.allclose(np.linalg.norm(moved, axis=1), 1.0, atol=1e-6)


# --- the measurement's own assumptions --------------------------------------

def test_the_query_sentence_is_excluded_from_its_own_targets():
    """Otherwise BM25 scores a passage against words lifted straight out of it, which is string search,
    and the vector half loses the comparison to an artefact."""
    held = [Chunk("The committee voted to raise the target range today.", [], "a", EARLY),
            Chunk("Inflation has eased since the previous meeting of this committee.", [], "a", EARLY)]
    asked = queries(held, np.random.default_rng(0), 1, len)
    (query, targets), = asked
    assert not any(query in held[index].text for index in targets)


def test_the_most_distinctive_sentence_is_the_query_not_a_random_one():
    """A random sentence out of a Fed release is its standing boilerplate, and asking any retriever to
    find one release from a line printed in four thousand of them measures nothing."""
    boilerplate = "For media inquiries please call the office on this number."
    held = [Chunk(boilerplate, [], "a", EARLY),
            Chunk(f"{boilerplate} The repo corridor widened by twelve basis points overnight.",
                  [], "a", EARLY)]
    rare = Hybrid(held, pretokenize).lexical.inverse_document_frequency
    asked = queries(held, np.random.default_rng(0), 1,
                    lambda text: np.mean([rare.get(term, 0.0) for term in pretokenize(text)] or [0.0]))
    assert asked[0][0].startswith("The repo corridor")
