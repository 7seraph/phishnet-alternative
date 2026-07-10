"""Curated phishing-avoidance tips. Tips are keyed by the heuristic reason id
that triggered them (see heuristics.py) so the API can surface advice that's
actually relevant to what was found in the message, on top of a couple of
tips that are always useful.
"""

GENERAL_TIPS = [
    "Hover over a link (or long-press on mobile) to preview its real destination before clicking.",
    "When in doubt, contact the sender through a phone number or website you already trust - not by replying to the email or using contact info it provides.",
    "Legitimate companies will never ask you to send a password, SSN, or full card number by email.",
]

TIP_BY_REASON_ID = {
    "urgency": (
        "Scammers manufacture urgency so you act before you think it through. "
        "A real deadline will survive a five-minute phone call to verify it."
    ),
    "generic_greeting": (
        "A company that genuinely has your account usually greets you by name. "
        "A generic greeting like \"Dear Customer\" is typical of mass-sent scams."
    ),
    "sensitive_info": (
        "Never provide passwords, SSNs, or one-time codes by replying to an email "
        "or filling out a form linked from one."
    ),
    "link_mismatch": (
        "Copy a link's real address (right-click > copy link, or long-press on mobile) "
        "and compare it to what's displayed before clicking."
    ),
    "risky_link": (
        "Type the company's website address directly into your browser instead of "
        "clicking the link in the email."
    ),
    "sender_link_mismatch": (
        "Check that the sender's email domain (the part after the @) actually matches "
        "the company it claims to be from."
    ),
}

TIP_FOR_PHISHING_VERDICT = (
    "If you weren't expecting this email, don't reply - report it as phishing/spam "
    "using your email provider's built-in tool and delete it."
)


def build_tips(reason_ids, prediction):
    tips = []
    seen = set()

    for reason_id in reason_ids:
        tip = TIP_BY_REASON_ID.get(reason_id)
        if tip and tip not in seen:
            tips.append(tip)
            seen.add(tip)

    if prediction == "fake" and TIP_FOR_PHISHING_VERDICT not in seen:
        tips.append(TIP_FOR_PHISHING_VERDICT)
        seen.add(TIP_FOR_PHISHING_VERDICT)

    for tip in GENERAL_TIPS:
        if tip not in seen:
            tips.append(tip)
            seen.add(tip)

    return tips
