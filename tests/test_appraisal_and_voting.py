from erc_engine.strategies.appraisal import AppraisalSanitizer, UnconditionalAssignmentPolicy
from erc_engine.strategies.voting import EqualAgentWeightedVoting


def test_appraisal_defaults_and_assignment_policy():
    clean = AppraisalSanitizer.sanitize({"appraisals": [{"utt_key": "utt1", "appraisal": {}}]}, ["utt1"])
    assigned = UnconditionalAssignmentPolicy.assign(clean)
    assert assigned["stage1_appraisal_gate"][0]["appraisal"]["goal_relevance"] == "low"
    assert len(assigned["stage1_appraisal_gate"][0]["gate"]["modules"]) == 6


def test_equal_expert_vote_can_outvote_baseline():
    labels = ("neutral", "joy", "anger")
    normalize = lambda value, default="neutral": value if value in labels else default
    voting = EqualAgentWeightedVoting(labels, normalize)
    baseline = {"predictions": [{"utt_key": "utt1", "emotion": "neutral", "confidence": 0.9}]}
    prior = {"stage2_5_appraisal_prototype_prior": []}
    experts = {"stage2": [{"utt_key": "utt1", "selected_modules": {
        "a": {"emotion_hint": "anger"}, "b": {"emotion_hint": "anger"}
    }}]}
    assert voting.fuse(baseline, prior, experts)["predictions"][0]["emotion"] == "anger"

