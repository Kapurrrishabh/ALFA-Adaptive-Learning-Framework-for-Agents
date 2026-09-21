"""Stack Exchange finance Q&A: money, quant, economics.

This is the only informal register in the corpus. Everything else — filings, circulars, textbooks,
speeches — is written by institutions for institutions. These are real people asking "should I
prepay my home loan?" and being answered in ordinary language, which is the register an agent
talking to a user actually has to produce.

Stack Exchange publishes complete dumps on archive.org under CC BY-SA, so this needs no key and no
crawling: one file per site. Only Posts.xml is read, and only questions and answers from it — a
question and its answers become one document, because the thread is the conversation. Comments.xml
is in the dump too and is more informal still, but it is not read here.

Dumps are refreshed quarterly. A site already holding both its threads and its question-and-answer
records is skipped rather than re-downloaded, because re-fetching 130 MB to discover nothing changed
is not worth it.
"""

import io
import json
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import py7zr

from .. import config
from ..common.http import BlockedByHost
from ..extract import html_text

LICENCE = "CC BY-SA 4.0 (Stack Exchange contributors)"
SOURCE = "stackexchange"

_QUESTION, _ANSWER = "1", "2"
_POSTS = "Posts.xml"


def _thread_text(title, body):
    return f"{title}\n\n{html_text.extract(body)}" if title else html_text.extract(body)


def _write_threads(posts_path, destination_root, records_root, site):
    """One file per question with its answers appended, plus the same posts as records.

    Ids in the dump ascend and an answer is always created after its question, so a single pass
    can append an answer to a file the same pass already wrote.

    The flattened thread is what pretraining reads; the records keep the structure pretraining
    throws away. Which answer the asker accepted, and how the site voted, is the only signal in
    the corpus that says a finance answer was any good — it is what a question-answering model
    trains on and what preference pairs are drawn from later. Questions and answers stream into
    separate files so nothing has to be held in memory until its thread is complete.
    """
    written = set()
    questions_path = records_root / "questions.jsonl"
    answers_path = records_root / "answers.jsonl"
    with questions_path.open("w", encoding="utf-8") as questions, answers_path.open(
        "w", encoding="utf-8"
    ) as answers:
        for _, element in ElementTree.iterparse(posts_path):
            if element.tag != "row":
                continue
            kind = element.get("PostTypeId")
            body = element.get("Body") or ""

            if kind == _QUESTION and body:
                post_id = element.get("Id")
                title = element.get("Title") or ""
                path = destination_root / f"{post_id}.txt"
                path.write_text(_thread_text(title, body), encoding="utf-8")
                written.add(post_id)
                _write_record(
                    questions,
                    site=site,
                    id=post_id,
                    title=title,
                    text=html_text.extract(body),
                    accepted_answer_id=element.get("AcceptedAnswerId"),
                    score=element.get("Score"),
                    tags=element.get("Tags") or "",
                )
            elif kind == _ANSWER and body:
                parent = element.get("ParentId")
                if parent in written:
                    with (destination_root / f"{parent}.txt").open("a", encoding="utf-8") as handle:
                        handle.write("\n\n" + html_text.extract(body))
                    _write_record(
                        answers,
                        site=site,
                        id=element.get("Id"),
                        question_id=parent,
                        text=html_text.extract(body),
                        score=element.get("Score"),
                    )
            element.clear()
    return written


def _write_record(handle, score, **fields):
    # Score is absent on a handful of rows in the older dumps, and a missing vote count is not the
    # same as zero votes: one is unknown, the other is a real signal about the answer.
    fields["score"] = int(score) if score is not None else None
    handle.write(json.dumps(fields, ensure_ascii=False) + "\n")


def _collect_site(session, manifest, site, log):
    destination_root = config.RAW_DIR / SOURCE / site.split(".")[0]
    records_root = config.QA_DIR / site.split(".")[0]
    has_threads = destination_root.exists() and any(destination_root.iterdir())
    if has_threads and (records_root / "answers.jsonl").exists():
        log(f"  {site}: already held, skipping the dump download")
        return 0
    if has_threads:
        log(f"  {site}: threads held but no answer records, re-reading the dump")
    destination_root.mkdir(parents=True, exist_ok=True)
    records_root.mkdir(parents=True, exist_ok=True)

    archive = session.get(config.STACKEXCHANGE_DUMP.format(site=site), stream=True).content
    log(f"  {site}: {len(archive) // (1024 * 1024)} MB archive, extracting {_POSTS}")

    with tempfile.TemporaryDirectory() as workspace:
        py7zr.SevenZipFile(io.BytesIO(archive)).extract(path=workspace, targets=[_POSTS])
        written = _write_threads(Path(workspace) / _POSTS, destination_root, records_root, site)

    fetched, dropped_short = 0, 0
    for post_id in sorted(written):
        path = destination_root / f"{post_id}.txt"
        if path.stat().st_size < config.STACKEXCHANGE_MINIMUM_CHARACTERS:
            path.unlink()
            dropped_short += 1
            continue
        manifest.record(f"https://{site}/questions/{post_id}", path, SOURCE, LICENCE)
        fetched += 1

    log(f"  {site}: {fetched} threads kept, {dropped_short} too short")
    return fetched


def collect(session, manifest, settings, log):
    sites = config.STACKEXCHANGE_SITES
    limit = settings["stackexchange_sites"]
    if limit is not None:
        sites = sites[:limit]

    fetched = 0
    for site in sites:
        try:
            fetched += _collect_site(session, manifest, site, log)
        except BlockedByHost:
            raise
        except RuntimeError as error:
            log(f"  {site}: {error}")
    return fetched
