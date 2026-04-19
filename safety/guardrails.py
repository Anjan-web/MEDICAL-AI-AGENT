BLOCKED_TOPICS = [
    "suicide", "self-harm", "how to kill", "overdose on purpose",
    "illegal drugs", "drug abuse", "how to make drugs",
    "bomb", "weapon", "poison someone"
]

MEDICAL_DISCLAIMER = (
    "Note: This information is for educational purposes only "
    "and does not constitute medical advice. "
    "Please consult a qualified healthcare professional for diagnosis and treatment."
)

def check_safety(question: str) -> dict:
    question_lower = question.lower()

    for topic in BLOCKED_TOPICS:
        if topic in question_lower:
            return {
                "safe": False,
                "reason": f"Query contains sensitive topic: '{topic}'",
                "answer": (
                    "I'm sorry, I can't help with that. "
                    "If you're in crisis, please contact a healthcare professional "
                    "or call emergency services."
                )
            }

    return {"safe": True, "reason": None, "answer": None}


def add_disclaimer(answer: str) -> str:
    return f"{answer}\n\n{MEDICAL_DISCLAIMER}"