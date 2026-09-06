MODULES = (
    "diagnosis_of_thought",
    "cognitive_distortion",
    "social_appraisal",
    "visual_agent",
    "speech_agent",
    "personality_traits",
)

APPRAISAL_FIELDS = {
    "goal_relevance": ({"low", "medium", "high", "unclear"}, "low"),
    "goal_congruence": ({"supports_goal", "blocks_goal", "mixed", "unclear"}, "unclear"),
    "agency": ({"self", "other", "shared", "environment", "unclear"}, "unclear"),
    "control": ({"low", "medium", "high", "unclear"}, "unclear"),
    "certainty": ({"low", "medium", "high", "unclear"}, "medium"),
    "novelty": ({"low", "medium", "high", "unclear"}, "low"),
    "social_threat": ({"low", "medium", "high", "unclear"}, "low"),
    "moral_violation": ({"low", "medium", "high", "unclear"}, "low"),
    "affect_intensity": ({"low", "medium", "high", "unclear"}, "low"),
    "loss_or_separation": ({"low", "medium", "high", "unclear"}, "low"),
    "blame_responsibility": ({"low", "medium", "high", "unclear"}, "low"),
    "expectancy_violation": ({"low", "medium", "high", "unclear"}, "low"),
    "appraisal_target": ({"self", "other", "relationship", "event", "unclear"}, "unclear"),
}

