"""Intent taxonomy for SpotifyCares, derived from the data.

Method: TF-IDF + KMeans (k=14) over 28k customer opening messages, then manual
merge of clusters into 7 actionable intents + `other`. Merges were driven by
"would a support agent do the same thing for these?", not by cluster geometry.
See reports/DECISIONS.md.
"""

INTENTS = {
    "playback_error": (
        "App, player or device is not working: won't play, crashes, skips, "
        "shuffle/repeat broken, offline downloads failing, web player broken, "
        "Spotify Connect / car / speaker issues, sound quality, battery, sync."
    ),
    "account_access": (
        "Cannot get into or control the account: login fails, forgotten or "
        "incorrect password, email change, account hacked/compromised, "
        "unauthorised devices, deleting the account."
    ),
    "billing_subscription": (
        "Money or plan state: charged wrongly or twice, refund, cancel, "
        "upgrade/downgrade, payment method declined, promo or student discount, "
        "trial, receipts, subscription still showing as free."
    ),
    "family_plan": (
        "Spotify Family / Duo specifically: invites, address verification, "
        "adding or removing members, plan owner issues."
    ),
    "content_request": (
        "About the music catalogue itself: song/album/artist missing or removed, "
        "not available in my country, wrong metadata or wrong artist tagged, "
        "'please add X to Spotify', lyrics or podcast availability."
    ),
    "feature_feedback": (
        "Product opinion rather than a fault: feature request, UX complaint, "
        "recommendation/Discover Weekly quality, design change gripes."
    ),
    "praise_chatter": (
        "No support action needed: thanks, compliments, jokes, memes, "
        "fandom talk, replies to Spotify's own marketing."
    ),
    "other": (
        "Anything that fits none of the above, or is too vague to classify: "
        "empty text, pure links, unrelated brands, press/business enquiries."
    ),
}

# Escalation policy. Derived from what SpotifyCares itself actually did:
# every intent below that needs an account lookup was answered in the data with
# "DM us your account email" -- i.e. a human working backstage.
AUTO_OK = {"content_request", "feature_feedback", "praise_chatter"}
ALWAYS_ESCALATE = {"account_access", "billing_subscription", "family_plan"}

ESCALATION_POLICY = """\
Escalate to a human when ANY of these hold:
1. Resolving it needs account-specific data (billing, subscription state,
   family plan membership, login/identity). The agent has no account access.
2. Security or fraud: hacked account, unauthorised charges, unknown devices.
3. The customer has posted personal data (email, card, phone) in public.
4. Legal threat, press enquiry, self-harm, or abuse of staff.
5. The customer is already angry after a previous unresolved reply.
6. The agent's own confidence in the intent is low, or the retrieved history
   contains no similar resolved case.

Otherwise auto-handle: catalogue questions, feature feedback, praise, and
playback problems that map to standard, publicly documented troubleshooting.
"""


def policy_escalates(intent: str) -> bool:
    """Intent-only prior. The agent may still escalate an AUTO_OK intent."""
    return intent not in AUTO_OK


LABELS = list(INTENTS)

if __name__ == "__main__":
    assert set(ALWAYS_ESCALATE) | AUTO_OK <= set(LABELS)
    assert not (ALWAYS_ESCALATE & AUTO_OK)
    for k, v in INTENTS.items():
        print(f"{k:22} {v[:70]}...")
