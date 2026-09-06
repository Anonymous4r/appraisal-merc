from erc_engine.domain.constants import MODULES

COGNITIVE_DISTORTIONS = {
    "emotional_reasoning": "Feelings overrule contrary facts.",
    "overgeneralization": "Broad conclusions from limited evidence.",
    "mental_filter": "Attention is restricted to negative evidence.",
    "should_statements": "Rigid rules dictate how people must behave.",
    "all_or_nothing_thinking": "Only extreme outcomes are considered.",
    "mind_reading": "Another person's thoughts are assumed without evidence.",
    "fortune_telling": "A fixed future outcome is assumed.",
    "magnification": "Negative evidence is exaggerated or positive evidence minimized.",
    "personalization": "Broad causes are taken as personal blame.",
    "labeling": "A fixed label is applied without enough evidence.",
    "no_distortion": "No clear distortion is present.",
}

FRIENDS_PERSONA = {
    "Chandler": "Witty and sarcastic; often uses humor defensively.",
    "Ross": "Intellectual and earnest; strongly affected by family and relationship events.",
    "Joey": "Direct, loyal, warm and socially uncomplicated.",
    "Phoebe": "Eccentric, independent and empathetic.",
    "Monica": "Organized, competitive, hospitable and group-oriented.",
    "Rachel": "Socially perceptive and increasingly independent.",
}

MODULE_CONTRACTS = {
    "diagnosis_of_thought": "Separate objective facts, subjective belief and reasoning; infer one natural emotion.",
    "cognitive_distortion": f"Classify one thought pattern from {COGNITIVE_DISTORTIONS} and infer its emotional implication.",
    "social_appraisal": "Analyze relationship, face threat, blame, support, norms and social evaluation.",
    "visual_agent": "Use only supplied visual evidence; never invent facial or body cues.",
    "speech_agent": "Use only supplied audio analysis; never invent pitch, energy, tempo or voice quality.",
    "personality_traits": f"Use stable traits only as a weak tie-breaker. Persona context: {FRIENDS_PERSONA}",
}

assert set(MODULE_CONTRACTS) == set(MODULES)

