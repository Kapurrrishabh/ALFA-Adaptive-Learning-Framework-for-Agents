"""The learning loop's state: what the agent was told, by whom, and in what order."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from selfagent.data import advisory
from selfagent.learn import AGENT, ORACLE, FeedbackLog
from selfagent.learn.abstain import Abstainer
from selfagent.learn.calibrate import (Calibrator, expected_calibration_error, ranking_auc)
from selfagent.learn.rank import best, consensus, rank
from selfagent.learn.teacher import AgentTeacher, OracleTeacher, verdict_key

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

TURN = dict(asked_at="2026-09-22T10:00:00", question="is AAPL overbought ?",
            evidence="ticker AAPL ; rsi_14 71", answer="its 14 day rsi is 71 .", confidence=0.98)

REFUSAL = advisory.UNSUPPORTED


def log(tmp_path):
    return FeedbackLog(tmp_path / "feedback.sqlite")


def oracle():
    return OracleTeacher(REFUSAL)


def confident_but_often_wrong(rows=400, seed=0):
    """Confidences crammed against 1.0, a third of them right, like the real feedback log."""
    rng = np.random.default_rng(seed)
    confidence = 1.0 - rng.beta(1.0, 300.0, size=rows)
    likely = 0.15 + 0.5 * (confidence - confidence.min()) / np.ptp(confidence)
    return confidence, rng.random(rows) < likely


def test_a_turn_comes_back_out_the_way_it_went_in(tmp_path):
    with log(tmp_path) as feedback:
        feedback.append(**TURN, is_right=True, labeller=ORACLE, why="matches the gold answer")
        row = feedback.rows()[0]
    assert row["question"] == TURN["question"]
    assert row["is_right"] is True
    assert row["labeller"] == ORACLE
    assert row["confidence"] == pytest.approx(0.98)


def test_the_log_survives_being_reopened(tmp_path):
    """The loop runs across sessions, and a store that only held while the process lived would lose the
    history the whole experiment is measured over."""
    with log(tmp_path) as feedback:
        feedback.append(**TURN, is_right=False, labeller=AGENT)
    with log(tmp_path) as reopened:
        assert len(reopened) == 1
        assert reopened.rows()[0]["is_right"] is False


def test_replaying_twice_gives_the_same_training_set(tmp_path):
    """A learning curve is a sequence of fits over growing prefixes of this log. If a replay came back
    in a different order the curve would be measuring the shuffle, not the learning."""
    with log(tmp_path) as feedback:
        for index in range(5):
            feedback.append(**{**TURN, "question": f"question {index}"}, is_right=index % 2 == 0,
                            labeller=ORACLE)
        first = [row["question"] for row in feedback.rows()]
        second = [row["question"] for row in feedback.rows()]
    assert first == second == [f"question {index}" for index in range(5)]


def test_an_unjudged_turn_is_kept_but_left_out_of_a_replay(tmp_path):
    """Serving logs every turn; only some get judged. Counting an unjudged turn as wrong would punish
    the model for feedback that never arrived, which is the silent way to fake a learning curve."""
    with log(tmp_path) as feedback:
        feedback.append(**TURN, is_right=None, labeller=AGENT)
        assert len(feedback) == 1
        assert feedback.rows(judged_only=True) == []
        assert len(feedback.rows(judged_only=False)) == 1


def test_a_row_with_no_named_judge_is_refused(tmp_path):
    """The one field that cannot be defaulted. Without it a result cannot be split by who judged it,
    and 'the agent agreed with itself' would be indistinguishable from a measurement."""
    with log(tmp_path) as feedback:
        with pytest.raises(ValueError, match="unknown labeller"):
            feedback.append(**TURN, is_right=True, labeller="whoever")


def test_agreement_counts_only_answers_both_judges_saw(tmp_path):
    """Agreement over rows only one judge labelled would be invented. Here the oracle and the agent
    overlap on one answer and disagree on it, so agreement is 0 of 1, not 0 of 3."""
    with log(tmp_path) as feedback:
        feedback.append(**TURN, is_right=True, labeller=ORACLE)
        feedback.append(**TURN, is_right=False, labeller=AGENT)
        feedback.append(**{**TURN, "question": "only the agent saw this"}, is_right=True, labeller=AGENT)
        assert feedback.agreement() == (0, 1)


def test_a_later_label_does_not_erase_an_earlier_one(tmp_path):
    """Append-only, and the reason is the thesis: what the agent believed and when is the data. An
    update in place would leave a log that cannot explain its own learning curve."""
    with log(tmp_path) as feedback:
        feedback.append(**TURN, is_right=False, labeller=AGENT)
        feedback.append(**TURN, is_right=True, labeller=ORACLE)
        assert [row["is_right"] for row in feedback.rows()] == [False, True]


def test_the_oracle_blames_the_invented_figure_and_not_the_wording():
    """Both checks fail at once on an invented figure, and which one is reported decides what the loop
    tries to fix. An unsupported number is the failure the design exists to prevent, so it is named
    first — 'differs from the gold answer' would send the next pass after the phrasing instead."""
    is_right, why = oracle().judge(TURN["question"], TURN["evidence"], "its 14 day rsi is 62 .",
                                   "its 14 day rsi is 71 .")
    assert is_right is False
    assert "62" in why and "figures" in why


def test_the_oracle_judges_a_word_at_a_time_like_the_scorer_does():
    """It has to agree with check_answers.py's exact match. A labeller that called a row right where
    the scorer called it wrong would leave the log and the reported accuracy permanently out of step,
    and neither number would be checkable against the other."""
    graded = oracle().judge(TURN["question"], TURN["evidence"], "its  14 day  rsi is 71 .",
                            "its 14 day rsi is 71 .")
    assert graded[0] is True


def test_the_oracle_separates_a_false_refusal_from_a_wrong_answer():
    """Two different failures with different fixes: refusing what the evidence answers costs coverage,
    answering what it does not costs trust. A single 'wrong' label would merge them. The invention here
    states no figure on purpose — with one, the figure check fires first and says so instead."""
    refused, why = oracle().judge(TURN["question"], TURN["evidence"], REFUSAL, "its 14 day rsi is 71 .")
    assert refused is False and "refused" in why
    answered, why = oracle().judge(TURN["question"], "ticker AAPL", "it is overbought .", REFUSAL)
    assert answered is False and "without evidence" in why


def test_the_oracle_refuses_to_judge_without_a_gold_answer():
    """The silent-fallback failure. Defaulting to 'right' or 'wrong' here would put a label in the log
    that nothing ever produced, and it would be indistinguishable from a real one."""
    with pytest.raises(ValueError, match="needs a gold answer"):
        oracle().judge(TURN["question"], TURN["evidence"], "its 14 day rsi is 71 .")


def test_the_agent_says_it_does_not_know_rather_than_guessing(tmp_path):
    """The whole loop rests on this. A teacher that returned a default verdict on a cache miss would
    manufacture agreement and every number downstream would be flattered by it."""
    agent = AgentTeacher(tmp_path / "verdicts.jsonl")
    is_right, why = agent.judge(TURN["question"], TURN["evidence"], TURN["answer"])
    assert is_right is None
    assert why == "no verdict yet"
    assert len(agent.asked) == 1


def test_an_agent_verdict_replays_in_a_later_run(tmp_path):
    """A learning curve has to be reproducible, so the second run must see the first run's feedback
    without the agent being asked again."""
    first = AgentTeacher(tmp_path / "verdicts.jsonl")
    first.judge(TURN["question"], TURN["evidence"], TURN["answer"])
    first.record(verdict_key(TURN["question"], TURN["answer"]), True, "right fact, different words")

    replayed = AgentTeacher(tmp_path / "verdicts.jsonl")
    assert replayed.judge(TURN["question"], TURN["evidence"], TURN["answer"]) == (
        True, "right fact, different words")
    assert replayed.asked == []


def test_two_answers_to_one_question_get_their_own_verdicts(tmp_path):
    """Keying on the question alone would let one verdict label a second, different answer — the model
    would be told its wrong answer was fine because an earlier good one had been approved."""
    agent = AgentTeacher(tmp_path / "verdicts.jsonl")
    agent.record(verdict_key(TURN["question"], TURN["answer"]), True)
    assert agent.judge(TURN["question"], TURN["evidence"], "its 14 day rsi is 62 .")[0] is None


def test_the_requests_file_carries_what_a_judge_needs_and_no_gold(tmp_path):
    """Showing the gold would turn judging into string comparison, and agreement would be 100% by
    construction rather than measured."""
    agent = AgentTeacher(tmp_path / "verdicts.jsonl")
    agent.judge(TURN["question"], TURN["evidence"], TURN["answer"])
    written = tmp_path / "wanted.jsonl"
    assert agent.write_requests(written) == 1
    row = json.loads(written.read_text())
    assert set(row) == {"key", "question", "evidence", "answer"}


def test_reading_the_verdicts_back_twice_logs_them_once(tmp_path):
    """--tell gets re-run: after an interruption, or because the operator is not sure it took. A second
    append would give one judgement two votes in every fit made from this log, and nothing would show it
    — the agreement rate is over distinct answers, so it stays put while the row count doubles."""
    from label_feedback import tell

    (tmp_path / "verdicts_given.jsonl").write_text(json.dumps(
        {"key": verdict_key(TURN["question"], TURN["answer"]), "is_right": True, "why": "right"}) + "\n")
    with log(tmp_path) as feedback:
        feedback.append(**TURN, is_right=False, labeller=ORACLE, why="differs from the gold answer")
        tell(tmp_path, feedback)
        tell(tmp_path, feedback)
        assert len(feedback.rows(labeller=AGENT)) == 1
        assert feedback.agreement() == (0, 1)


def test_calibration_pulls_a_confident_model_down_to_the_rate_it_is_right():
    """The failure this exists for: the model states 0.99 on answers that are right a third of the time.
    An abstention threshold set on a number that never moves is a threshold on nothing."""
    confidence, right = confident_but_often_wrong()
    calibrated = Calibrator().fit(confidence, right)(confidence)
    assert confidence.mean() > 0.98
    assert calibrated.mean() == pytest.approx(right.mean(), abs=0.05)
    assert expected_calibration_error(calibrated, right) < \
        expected_calibration_error(confidence, right) / 3


def test_calibration_never_reorders_two_answers():
    """B4 ranks on this score and B5 picks the top of it. A calibrator that changed the order would make
    the AUC reported beside it a fiction, and would quietly move which answers get abstained on."""
    confidence, right = confident_but_often_wrong()
    calibrated = Calibrator().fit(confidence, right)(confidence)
    assert ranking_auc(calibrated, right) == pytest.approx(ranking_auc(confidence, right))


def test_calibration_error_can_see_an_error_in_a_crowded_score():
    """Equal-width bins would drop all 400 of these into the top bin and report one bin's average miss,
    near zero. The measure has to survive every score sitting above 0.98."""
    confidence, right = confident_but_often_wrong()
    overconfident_by = confidence.mean() - right.mean()
    assert expected_calibration_error(confidence, right) > 0.9 * overconfident_by


def test_a_calibrator_will_not_fit_on_rows_that_are_all_right():
    """Silently returning a constant here would ship a mapping that says 1.0 to everything, and the
    first genuinely wrong answer would be served with full confidence."""
    with pytest.raises(ValueError, match="no difference to learn"):
        Calibrator().fit(np.array([0.9, 0.99]), np.array([True, True]))


def test_a_saved_calibrator_predicts_the_same_after_loading(tmp_path):
    """It is fit in one process and served in another. A mapping that drifted across that boundary would
    move every abstention decision downstream of it, with nothing to show the two disagreed."""
    confidence, right = confident_but_often_wrong()
    fitted = Calibrator().fit(confidence, right)
    fitted.save(tmp_path / "calibrator.json")
    assert Calibrator.load(tmp_path / "calibrator.json")(confidence) == \
        pytest.approx(fitted(confidence))


def graded_by_confidence():
    """100 rows: the top fifth nearly all right, the next third a coin toss, the bottom half wrong.

    One wrong answer sits inside the top fifth so that no cut is perfect, which is also what a real
    precision-coverage curve looks like — a fixture without it makes any bar reachable.
    """
    confidence = np.linspace(0.90, 0.999, 100)
    right = np.zeros(100, dtype=bool)
    right[80:] = True
    right[95] = False
    right[50:80:2] = True
    return confidence, right


def test_the_cut_is_the_lowest_that_clears_the_bar_not_the_most_precise():
    """The most precise cut is nearly always the one answering a handful of rows at 100%, and a threshold
    fitted to a handful is a coincidence quoted as a policy. Asking for 60% has to buy coverage."""
    abstainer = Abstainer.fit(*graded_by_confidence(), wanted=0.6)
    assert abstainer.expected >= 0.6
    assert abstainer.coverage >= 0.5


def test_a_bar_nothing_reaches_is_reported_as_missed_not_claimed():
    """Returning the bar that was asked for would put a precision in the served config that no row ever
    demonstrated, and the first person to trust it would be reading a wish."""
    confidence, right = graded_by_confidence()
    abstainer = Abstainer.fit(confidence, right, wanted=0.99)
    assert abstainer.wanted == 0.99
    assert abstainer.expected < 0.99


def test_no_feedback_yet_refuses_to_produce_a_cut():
    """A fresh install has an empty log. Defaulting to a cut of zero there would answer every question
    at full confidence, which is the opposite of what abstention is for."""
    with pytest.raises(ValueError, match="collect more feedback"):
        Abstainer.fit(np.array([]), np.array([], dtype=bool), wanted=0.6)


def test_a_saved_cut_answers_the_same_rows_after_loading(tmp_path):
    confidence, right = graded_by_confidence()
    fitted = Abstainer.fit(confidence, right, wanted=0.6).save(tmp_path / "abstain.json")
    assert (Abstainer.load(tmp_path / "abstain.json").answers(confidence)
            == fitted.answers(confidence)).all()


GROUNDED_ANSWER = "its 14 day rsi is 71 ."
INVENTED_ANSWER = "its 14 day rsi is 62 ."


def test_an_invented_figure_loses_however_confident_it_is():
    """The failure the tiers exist for. The model is most fluent exactly when it invents, so ranking on
    confidence alone would systematically serve the made-up figure — a reranker doing harm, not nothing."""
    picked = best([INVENTED_ANSWER, GROUNDED_ANSWER], TURN["evidence"], [1.0, 0.5], REFUSAL)
    assert picked == GROUNDED_ANSWER


def test_a_refusal_beats_an_invention_and_loses_to_a_grounded_answer():
    """Both halves matter. Putting refusals last would serve invented figures whenever nothing else is
    grounded; putting them first would make the model refuse whenever any sample happened to."""
    candidates = [INVENTED_ANSWER, REFUSAL, GROUNDED_ANSWER]
    order = rank(candidates, TURN["evidence"], [1.0, 0.2, 0.3], REFUSAL)
    assert [candidates[position] for position in order] == [GROUNDED_ANSWER, REFUSAL, INVENTED_ANSWER]


def test_confidence_only_decides_between_candidates_of_the_same_kind():
    """Within a tier the confidence is all there is to go on, and B3 measured that it ranks right above
    wrong at AUC 0.74. Ignoring it would throw away the one signal the log says is real."""
    grounded = [GROUNDED_ANSWER, "the rsi is 71 ."]
    assert best(grounded, TURN["evidence"], [0.4, 0.9], REFUSAL) == grounded[1]


def test_the_same_candidates_always_produce_the_same_pick():
    """A learning curve is a sequence of these picks. A tie broken by dict order or hash would make the
    curve move between runs, and the movement would be indistinguishable from learning."""
    tied = [GROUNDED_ANSWER, "the rsi is 71 .", "rsi 71 ."]
    assert rank(tied, TURN["evidence"], [0.8, 0.8, 0.8], REFUSAL) == [0, 1, 2]


def test_consensus_takes_the_repeated_answer_over_the_confident_one():
    """The whole point of the second policy. Within one question the confidence prefers the short
    refusal, so a picker that ignores it and counts repeats has to be able to disagree with it."""
    candidates = ["the rsi is 71 .", GROUNDED_ANSWER, GROUNDED_ANSWER]
    assert consensus(candidates, TURN["evidence"], REFUSAL) == 1


def test_consensus_will_not_repeat_its_way_past_the_guardrail():
    """A wrong figure is more repeatable than a right one when the model is confidently wrong. Counting
    votes without the tiers would serve the invented number seven times out of eight."""
    candidates = [INVENTED_ANSWER] * 7 + [GROUNDED_ANSWER]
    assert consensus(candidates, TURN["evidence"], REFUSAL) == 7


def test_ranking_nothing_is_refused_rather_than_answered():
    """A generation step that produced no candidates is a bug upstream. Returning a default answer here
    would hide it behind a served response."""
    with pytest.raises(ValueError, match="at least one"):
        rank([], TURN["evidence"], [], REFUSAL)
    with pytest.raises(ValueError, match="against"):
        rank([GROUNDED_ANSWER], TURN["evidence"], [0.9, 0.9], REFUSAL)
