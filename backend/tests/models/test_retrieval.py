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


from backend.knowledge_base.retrieval import ARMS, HYBRID, LEXICAL, SERVED, VECTOR, Chunk, Document, Hybrid, answers, chunks, deduplicate, documents, fuse, load, published_on, save
from backend.knowledge_base.retrieval.index import RRF_K, centred, without  # noqa: E402
from backend.models.agent.finance import Subject  # noqa: E402
from backend.database.live import BlockedByHost, Encyclopedia, Filings, Live  # noqa: E402
from backend.database.live.sources import EDGAR_EPOCH, EDGAR_SEARCH, WIKIPEDIA  # noqa: E402
from backend.models.training.eval_retrieval import queries  # noqa: E402
from backend.models.core.tokenizer.wordpiece import pretokenize  # noqa: E402

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


def test_a_truncated_sentence_cuts_its_text_too_and_not_only_its_ids(tmp_path):
    """7% of a 77-token index overflowed its own window because only the ids were cut. The text is what
    gets stored, re-encoded and read by the decoder, so a longer one is refused there as too big."""
    tokenizer = FakeTokenizer()
    built = chunks(document(tmp_path, " ".join(f"word{n}" for n in range(20))), tokenizer, 10,
                   min_tokens=1)
    assert len(tokenizer.encode(built[0].text)) == 10


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


def test_an_undated_chunk_is_visible_only_when_no_as_of_is_in_play():
    """What makes indexing the undated sources safe. An empty day sorts below every real date, so a plain
    `<=` would let a textbook page we cannot date answer a question asked in 2020 -- the look-ahead the
    corpus rule forbids. Excluded from a dated question, and retrievable for one that names no date."""
    held = dated_chunks() + [Chunk("inflation report from a textbook", [], "u0", "")]
    index = Hybrid(held, pretokenize)
    assert 10 in index.search("inflation report", 11)
    assert 10 not in index.search("inflation report", 11, as_of=LATE)


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


# --- the answers, indexed instead of the threads holding them ----------------


def qa(tmp_path, questions, answers):
    """A `data/qa/<site>` tree in the shape `dataforge` leaves it, one site."""
    site = tmp_path / "qa" / "money"
    site.mkdir(parents=True, exist_ok=True)
    for name, records in (("questions", questions), ("answers", answers)):
        with (site / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    return tmp_path / "qa"


def thread(question_id, accepted=None):
    """Ids are strings and the site is its full domain, which is how the collector writes them."""
    return {"site": "money.stackexchange.com", "id": str(question_id), "title": "what is a stop loss ?",
            "text": "asked badly", "accepted_answer_id": accepted}


def reply(answer_id, question_id, text, score=0):
    return {"site": "money.stackexchange.com", "id": str(answer_id), "question_id": str(question_id),
            "text": text, "score": score}


def threaded(sha256, question_id):
    """A manifest row for a collected thread, whose URL is what the key is looked up by."""
    return {"source": "stackexchange", "path": "data/raw/stackexchange/t.json", "sha256": sha256,
            "url": f"https://money.stackexchange.com/questions/{question_id}",
            "licence": "CC BY-SA 4.0", "fetched_at": "2026-09-22T00:00:00Z"}


def test_an_indexed_answer_holds_the_answers_words_and_not_the_questions(tmp_path):
    """The whole reason this path exists. Indexing the flattened thread put the asker's words in the
    corpus, so a question retrieved a question: measured, `what is a stop loss order ?` served a
    stranger's wash-sale question while the passages below it were all on topic."""
    manifest, _ = corpus(tmp_path, [threaded("a" * 64, 11)])
    root = qa(tmp_path, [thread(11, accepted=99)], [reply(99, 11, "it sells at the stop price.")])
    built, = answers(manifest, root)
    assert built.text == "it sells at the stop price."
    assert "asked badly" not in built.text and "stop loss ?" not in built.text


def test_an_indexed_answer_is_keyed_by_the_manifest_row_that_licences_it(tmp_path):
    """A citation resolves through the document key, so a key this corpus invented would attribute a
    CC BY-SA answer to nothing."""
    manifest, _ = corpus(tmp_path, [threaded("b" * 64, 11)])
    built, = answers(manifest, qa(tmp_path, [thread(11, accepted=99)],
                                  [reply(99, 11, "it sells at the stop price.")]))
    assert built.key == "b" * 16 and built.source == "qa_money"


def test_an_answer_nobody_endorsed_is_not_indexed(tmp_path):
    """Retrieving the wrong answer is the failure this path can still have, so the corpus holds only the
    ones the asker accepted or the site voted up."""
    manifest, _ = corpus(tmp_path, [threaded("a" * 64, 11), threaded("b" * 64, 12)])
    root = qa(tmp_path, [thread(11, accepted=99), thread(12)],
              [reply(99, 11, "accepted."), reply(98, 12, "downvoted.", score=-2)])
    assert [built.text for built in answers(manifest, root)] == ["accepted."]


def test_a_thread_the_manifest_never_recorded_is_skipped_not_indexed_unattributable(tmp_path):
    manifest, _ = corpus(tmp_path, [threaded("a" * 64, 11)])
    root = qa(tmp_path, [thread(11, accepted=99), thread(12, accepted=98)],
              [reply(99, 11, "recorded."), reply(98, 12, "collected in another run.")])
    assert [built.key for built in answers(manifest, root)] == ["a" * 16]


def test_no_thread_matching_the_manifest_at_all_is_an_error_not_an_empty_corpus(tmp_path):
    """Silently indexing nothing would look like a corpus with no endorsed answers in it."""
    manifest, _ = corpus(tmp_path, [threaded("a" * 64, 11)])
    root = qa(tmp_path, [thread(12, accepted=98)], [reply(98, 12, "another run.")])
    with pytest.raises(ValueError, match="collected from different runs"):
        answers(manifest, root)


def test_an_answer_carries_no_date_so_it_answers_only_undated_questions(tmp_path):
    """The records hold no creation date. An answer dated by its collection would be evidence the index
    served for a day nobody could show it was true on."""
    manifest, _ = corpus(tmp_path, [threaded("a" * 64, 11)])
    built, = answers(manifest, qa(tmp_path, [thread(11, accepted=99)], [reply(99, 11, "no date.")]))
    assert built.day == ""
    held = chunks(built, FakeTokenizer(), 10, min_tokens=1)
    assert Hybrid(held, pretokenize).search("no date", 1, as_of=LATE) == []


# --- passages fetched while the question waits -------------------------------

# Long enough to survive the chunker's minimum, which drops a page too short to be evidence.
FILED = "the company commissioned the unit and expects it to run at capacity within the quarter . " * 3


class FakeSession:
    """The two calls a source makes, answered from a dict, with every request recorded."""

    def __init__(self, json_by_url=None, text=f"<p>{FILED}</p>", fails=()):
        self.json_by_url = json_by_url or {}
        self.text = text
        self.fails = set(fails)
        self.asked = []

    def get_json(self, url, params=None):
        self.asked.append((url, dict(params or {})))
        if url in self.fails:
            raise BlockedByHost(f"{url} answered 403")
        return self.json_by_url[url]

    def get_text(self, url, params=None):
        self.asked.append((url, dict(params or {})))
        return self.text


REGISTRANT = Subject("AAPL", ("apple inc",), "0000320193")
UNREGISTERED = Subject("RELIANCE.NS", ("reliance industries",), None)

SEARCH = f"{WIKIPEDIA}/search/page"
HISTORY = f"{WIKIPEDIA}/page/Reliance_Industries/history"


def _hit(accession, day, cik="0000320193"):
    return {"_id": f"{accession}:ex99.htm", "_source": {"file_date": day, "ciks": [cik]}}


def filings_search(*hits):
    return {EDGAR_SEARCH: {"hits": {"hits": list(hits)}}}


def _edit(revision, day):
    return {"id": revision, "timestamp": f"{day}T04:22:27Z"}


def articles(*edits):
    """One article found by search, with the revisions it has, newest first as the endpoint returns them."""
    return {SEARCH: {"pages": [{"key": "Reliance_Industries"}]}, HISTORY: {"revisions": list(edits)}}


def test_filings_are_taken_newest_first_because_the_search_ranks_by_relevance(tmp_path):
    """The bug this pins was measured against EDGAR itself: `q=earnings` for Apple returns 2006, 2004 and
    2015 in that order, so taking the search's own top two serves a twenty-year-old filing to a question
    about now."""
    session = FakeSession(filings_search(_hit("0000-06", "2006-07-19"), _hit("0000-04", "2004-04-14"),
                                         _hit("0000-15", "2015-07-31")))
    fetched = Filings(session, filings=2).documents("what was announced", ["apple"])
    assert [page.day for page in fetched] == ["2015-07-31", "2006-07-19"]


def test_a_filing_search_asks_for_the_company_by_number_and_stops_at_the_as_of_date():
    """Two claims in one request. A name alone returns filings by whoever used the word, and the date bound
    is pushed into the search because filtering a top-two list down to the day returns nothing."""
    session = FakeSession(filings_search(_hit("0000-24", "2024-05-02")))
    Filings(session).documents("anything", [], as_of="2024-06-30", subject=REGISTRANT)
    _, params = session.asked[0]
    assert params["ciks"] == "0000320193" and params["q"] == "apple inc"
    assert (params["startdt"], params["enddt"]) == (EDGAR_EPOCH, "2024-06-30")


def test_an_instrument_that_files_nothing_here_is_not_searched_for_by_name():
    """RELIANCE.NS is not an SEC registrant, and a search for the word returns other companies' filings
    that merely say it -- which the gate would pass, because they do say it."""
    session = FakeSession(filings_search(_hit("0000-24", "2024-05-02")))
    assert Filings(session).documents("what has it announced", [], subject=UNREGISTERED) == []
    assert session.asked == []


def test_an_article_is_searched_for_by_the_companys_name_and_not_by_the_question():
    """Measured against the endpoint: "Reliance Industries" returns that article first, and "Reliance
    Industries recent news" returns Jio Platforms, a green-energy complex and a businessman."""
    session = FakeSession(articles(_edit(137, "2026-09-22")))
    Encyclopedia(session).documents("what has it announced recently ?", [], subject=UNREGISTERED)
    assert session.asked[0] == (SEARCH, {"q": "reliance industries", "limit": 2})


def test_an_article_is_read_at_the_revision_that_was_current_on_the_as_of_date():
    """Today's article carries edits made after the question's date, so serving it would be look-ahead by
    the plainest route there is. The citation names the revision so a reader checks what was read."""
    session = FakeSession(articles(_edit(137, "2026-09-22"), _edit(99, "2024-05-02")))
    page, = Encyclopedia(session, articles=1).documents("what is it", [], as_of="2024-12-31",
                                                       subject=UNREGISTERED)
    assert page.day == "2024-05-02" and page.key.endswith("?oldid=99")
    assert f"{WIKIPEDIA}/revision/99/html" in [url for url, _ in session.asked]


def test_an_article_edited_only_after_the_as_of_date_refuses_instead_of_serving_todays_text():
    """One page of history is 20 revisions, which reaches back two months on a page edited as often as
    Apple's. Past that this says so, because the alternative is answering 2010 out of 2026."""
    session = FakeSession(articles(_edit(137, "2026-09-22")))
    with pytest.raises(RuntimeError, match="all postdate 2010-01-01"):
        Encyclopedia(session).documents("what is it", [], as_of="2010-01-01", subject=UNREGISTERED)


def test_a_templates_wikitext_rides_in_an_attribute_and_must_not_reach_the_reader():
    """Measured on the Tesla article, which came back as `}}</ref>"},"module":{"wt":"{{infobox network serv`
    -- template source served as prose, because a `>` inside a quoted attribute value ended the tag."""
    leaky = ('<p><span typeof="mw:Transclusion" data-mw=\'{"target":{"wt":"cite web"},"params":{"url":'
             '{"wt":"https://www.sec.gov/ix?doc=/x.htm"}}}</ref>","module":{"wt":"{{infobox network serv"}\'>'
             f"{FILED}</span></p>")
    session = FakeSession(articles(_edit(137, "2026-09-22")), text=leaky)
    page, = Encyclopedia(session, articles=1).documents("what is it", [], subject=UNREGISTERED)
    assert page.text == FILED.strip()


def test_a_filings_xbrl_taxonomy_is_not_the_answer_to_what_the_company_does():
    """Measured on Amazon's 8-K, which came back as `false 0001018724 AMAZON COM INC 2026-07-09` -- the
    cover-page facts, which a tag strip keeps because they are element text and not attributes."""
    filing = ("<ix:header><ix:hidden>false 0001018724 AMAZON COM INC 2026-07-09</ix:hidden></ix:header>"
              f"<p>{FILED}</p>")
    session = FakeSession(filings_search(_hit("0000-26", "2026-07-09")), text=filing)
    page, = Filings(session, filings=1).documents("what does it do", [], subject=REGISTRANT)
    assert page.text == FILED.strip()


def test_an_article_is_not_answered_out_of_its_own_footnotes():
    """One element holds the whole reference list, and it is a page dense with the company's own name, so
    the lexical arm ranked it first: "who runs TCS" was answered with `Retrieved 16 August 2024 . |^ ...`."""
    article = (f"<p>{FILED}</p><ol class=\"mw-references references\" typeof=\"mw:Extension/references\">"
               "<li><cite>\"Tata Sons repays Rs 20,000-crore debt\" . The Times of India .</cite> Retrieved "
               "16 August 2024 .</li></ol>")
    session = FakeSession(articles(_edit(137, "2026-09-22")), text=article)
    page, = Encyclopedia(session, articles=1).documents("who runs it", [], subject=UNREGISTERED)
    assert page.text == FILED.strip()


def test_one_source_being_blocked_costs_its_passages_and_not_the_turn():
    """The other source may still hold the answer, so a refusal from one is not a refusal from the index."""
    session = FakeSession({**filings_search(_hit("0000-24", "2024-05-02")), SEARCH: {}}, fails=[SEARCH])
    found = Live([Encyclopedia(session), Filings(session)], FakeTokenizer(), chunk_tokens=64).passages(
        "what was announced", [], subject=REGISTRANT)
    assert [chunk.text for chunk in found] == [FILED.strip()]


def test_a_source_that_refused_and_found_nothing_travels_rather_than_looking_like_an_empty_corpus():
    """RELIANCE.NS is the case this was measured on: EDGAR is asked nothing because it files nothing there,
    so a blocked encyclopedia leaves no evidence either way -- and "nothing is known about this" would be a
    claim about the documents rather than about the network."""
    session = FakeSession({SEARCH: {}}, fails=[SEARCH])
    live = Live([Filings(session), Encyclopedia(session)], FakeTokenizer())
    with pytest.raises(BlockedByHost, match="no source could be searched"):
        live.passages("what has it announced", [], subject=UNREGISTERED)
