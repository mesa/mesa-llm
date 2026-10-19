"""
Court case definition for the jury deliberation simulation.

The case is deliberately ambiguous — strong evidence exists on both sides,
forcing jurors to genuinely debate rather than converge immediately.
"""

CASE_TITLE = "State v. Marcus Rivera"

CASE_SUMMARY = (
    "Marcus Rivera, 34, is charged with second-degree burglary of a jewelry store "
    "on the night of November 15th. The prosecution alleges Rivera broke into "
    "Eastside Fine Jewelers through a rear window and stole approximately $45,000 "
    "worth of merchandise. Rivera maintains he was at a friend's apartment that "
    "evening and had no involvement in the crime."
)

PROSECUTION_NARRATIVE = (
    "On the night of November 15th, a silent alarm was triggered at Eastside Fine "
    "Jewelers at 11:47 PM. Police arrived within 8 minutes to find the rear window "
    "smashed and display cases emptied. Security footage from a neighboring business "
    "captured a figure matching Rivera's build and clothing near the alley at 11:32 PM. "
    "Rivera's fingerprint was found on a shard of glass from the broken window. "
    "A pawn shop owner identified Rivera as the man who attempted to sell jewelry "
    "matching the stolen items three days after the burglary. Rivera has a prior "
    "conviction for petty theft from 2019."
)

DEFENSE_NARRATIVE = (
    "Marcus Rivera was at his friend David Chen's apartment from 9 PM until past "
    "midnight on November 15th, watching a basketball game. Mr. Chen will testify "
    "to this. Rivera's fingerprint on the glass can be explained by the fact that "
    "he had visited the jewelry store legitimately two days prior, on November 13th, "
    "to look at engagement rings for his girlfriend — the store's sales log confirms "
    "a visit by 'Marcus R.' on that date. The security footage is grainy and shows "
    "only a silhouette. The pawn shop identification occurred after the owner had "
    "already seen Rivera's photo in a news report about the arrest. Rivera's 2019 "
    "theft conviction was for shoplifting at age 27 — a nonviolent misdemeanor "
    "that he completed probation for."
)

EVIDENCE = {
    "E1": {
        "name": "Security Footage",
        "description": (
            "Grainy footage from a camera across the street shows a figure in dark "
            "clothing and a hoodie entering the alley behind the jewelry store at "
            "11:32 PM. The figure's height and build are consistent with Rivera "
            "(5'10\", 175 lbs), but facial features are not visible. The footage "
            "quality is low and taken from approximately 40 feet away."
        ),
        "favors": "prosecution",
        "strength": "moderate",
    },
    "E2": {
        "name": "Fingerprint on Window Glass",
        "description": (
            "A single fingerprint matching Rivera's right index finger was found on "
            "a glass shard from the broken rear window. However, Rivera visited the "
            "store two days earlier on November 13th and may have touched the window "
            "during that visit. The defense notes the fingerprint was on the exterior "
            "surface of the glass."
        ),
        "favors": "ambiguous",
        "strength": "strong",
    },
    "E3": {
        "name": "Alibi Witness — David Chen",
        "description": (
            "David Chen, Rivera's friend of 12 years, testifies that Rivera was at "
            "his apartment from approximately 9 PM to 12:30 AM on November 15th. "
            "They watched the Lakers vs. Celtics game together. Chen's testimony is "
            "consistent with the game schedule. However, the prosecution notes that "
            "Chen is a close friend and may be biased."
        ),
        "favors": "defense",
        "strength": "moderate",
    },
    "E4": {
        "name": "Pawn Shop Identification",
        "description": (
            "Pawn shop owner Gerald Hayes identified Rivera as the man who brought "
            "in jewelry matching some of the stolen items on November 18th. However, "
            "Hayes admitted during cross-examination that he had seen Rivera's photo "
            "in a local news report about the arrest before making the identification. "
            "The jewelry was never recovered from the pawn shop — Hayes says the man "
            "left when asked for ID."
        ),
        "favors": "prosecution",
        "strength": "weak",
    },
    # Note: prior record admissibility was debated at trial; judge allowed it
    "E5": {
        "name": "Prior Criminal Record",
        "description": (
            "Rivera has one prior conviction: petty theft (shoplifting) from 2019, "
            "classified as a nonviolent misdemeanor. He completed probation and had "
            "no further incidents until this charge. The defense argues this is "
            "irrelevant to a burglary charge."
        ),
        "favors": "prosecution",
        "strength": "weak",
    },
    "E6": {
        "name": "Store Visit Log — November 13",
        "description": (
            "The store's sales log shows a customer named 'Marcus R.' visited on "
            "November 13th at 2:15 PM and inquired about engagement rings. The sales "
            "associate, Maria Torres, confirms a man fitting Rivera's description "
            "spent about 20 minutes browsing. This supports Rivera's explanation for "
            "how his fingerprint ended up on the store's window."
        ),
        "favors": "defense",
        "strength": "strong",
    },
    "E7": {
        "name": "Cell Phone Location Data",
        "description": (
            "Rivera's cell phone pinged a tower near David Chen's apartment at "
            "10:05 PM and 11:58 PM. However, the tower also covers a radius that "
            "includes the jewelry store, which is 0.8 miles from Chen's apartment. "
            "The data neither confirms nor eliminates either location."
        ),
        "favors": "ambiguous",
        "strength": "moderate",
    },
}

MAX_EVIDENCE_ITEMS = len(EVIDENCE)

JUDGE_INSTRUCTIONS = (
    "Members of the jury, you must determine whether the prosecution has proven "
    "beyond a reasonable doubt that the defendant, Marcus Rivera, committed the "
    "crime of second-degree burglary. You must consider all evidence presented "
    "and evaluate the credibility of witnesses. If you have a reasonable doubt "
    "about the defendant's guilt, you must find him not guilty. A unanimous "
    "verdict is required."
)


def get_case_brief():
    """Return a compact case brief for inclusion in juror prompts."""
    evidence_summary = "\n".join(
        f"- [{eid}] {e['name']} ({e['strength']}, favors {e['favors']})"
        for eid, e in EVIDENCE.items()
    )
    return (
        f"CASE: {CASE_TITLE}\n\n"
        f"{CASE_SUMMARY}\n\n"
        f"PROSECUTION: {PROSECUTION_NARRATIVE}\n\n"
        f"DEFENSE: {DEFENSE_NARRATIVE}\n\n"
        f"EVIDENCE OVERVIEW:\n{evidence_summary}\n\n"
        f"JUDGE'S INSTRUCTIONS: {JUDGE_INSTRUCTIONS}"
    )


def get_evidence_detail(evidence_id: str) -> str:
    """Return full details for a specific piece of evidence."""
    evidence = EVIDENCE.get(evidence_id)
    if not evidence:
        available = ", ".join(EVIDENCE.keys())
        return f"Evidence '{evidence_id}' not found. Available: {available}"
    return (
        f"[{evidence_id}] {evidence['name']}\n"
        f"Favors: {evidence['favors']} | Strength: {evidence['strength']}\n"
        f"{evidence['description']}"
    )
