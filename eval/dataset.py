"""Labeled justification pairs for calibrating duplicate detection.

The duplicate service compares the *free-text business justification* of a new
request against existing requests **that already share the same categorical key**
(request_type + environment). So the unit that actually decides a duplicate is a
pair of justifications drawn from the same category. This dataset mirrors exactly
that: each row is two justifications plus a human label.

The set is deliberately *hard*. The negatives are same-category requests that
share vocabulary ("provision a Kafka cluster for X" vs "...for Y") — precisely
the false-positive risk that matters for a bank, where wrongly telling an
engineer "you already requested this" can block legitimate work. A backend that
only beats a trivial bag-of-words baseline on the easy paraphrases but collapses
on these negatives is not production-ready, and this dataset surfaces that.

It is a small, illustrative seed (not a production benchmark): enough to *measure*
precision/recall and pick a threshold with evidence instead of a guess. The
honest next step before production is a larger, domain-sampled, multiply-annotated
set — see eval/README.md.
"""

from __future__ import annotations

# (justification_a, justification_b, is_duplicate)
PAIRS: list[tuple[str, str, bool]] = [
    # ---------------------------------------------------------------------
    # Positives: same underlying request, worded differently (paraphrase,
    # synonyms, reordering, abbreviation). These SHOULD be flagged.
    # ---------------------------------------------------------------------
    (
        "Need extra compute capacity for the Q3 product launch",
        "Require additional capacity ahead of the Q3 launch",
        True,
    ),
    (
        "Provision a new Kafka cluster for the payments platform",
        "Set up a Kafka cluster to support the payments platform",
        True,
    ),
    (
        "Grant read access to the analytics datalake for the BI team",
        "BI team needs read permissions on the analytics data lake",
        True,
    ),
    (
        "Roll out the new fraud-detection service to handle peak traffic",
        "Deploy the fraud detection service so it can cope with peak load",
        True,
    ),
    (
        "Rotate the expired TLS certificates on the payments gateway",
        "The payments gateway TLS certs have expired and must be rotated",
        True,
    ),
    (
        "Increase the database connection pool to fix timeouts under load",
        "Raise the DB connection pool size because we hit timeouts at load",
        True,
    ),
    (
        "Onboard the new notifications microservice into the platform",
        "Add the notifications microservice onto the platform",
        True,
    ),
    (
        "Open firewall access from the app tier to the new Redis instance",
        "Allow the application tier to reach the new Redis through the firewall",
        True,
    ),
    (
        "Urgent incident fix: checkout service returning 500s in production",
        "Production checkout service is throwing 500 errors and needs a fix urgently",
        True,
    ),
    (
        "Add a staging environment for the recommendations pipeline",
        "We need a staging setup for the recommendations pipeline",
        True,
    ),
    (
        "Provision additional GPU nodes for model training",
        "Need more GPU nodes to run the model training jobs",
        True,
    ),
    (
        "Update the CI pipeline to run integration tests on every merge",
        "Change the CI to execute integration tests for each merge",
        True,
    ),
    (
        "Scale up the search cluster ahead of the marketing campaign",
        "Add capacity to the search cluster before the marketing push",
        True,
    ),
    (
        "Grant the data team write access to the reporting bucket",
        "Give the data team permission to write to the reporting bucket",
        True,
    ),
    # ---------------------------------------------------------------------
    # Negatives: same category, often overlapping vocabulary, but a
    # genuinely different request. These must NOT be flagged (precision).
    # ---------------------------------------------------------------------
    (
        "Provision a new Kafka cluster for the payments platform",
        "Provision a new Kafka cluster for the fraud-detection platform",
        False,
    ),
    (
        "Grant read access to the analytics datalake for the BI team",
        "Grant read access to the analytics datalake for the audit team",
        False,
    ),
    (
        "Deploy the payments service to handle higher throughput",
        "Deploy the notifications service to handle higher throughput",
        False,
    ),
    (
        "Need extra capacity for the Q3 product launch",
        "Need extra capacity for the year-end financial close",
        False,
    ),
    (
        "Rotate the expired TLS certificates on the payments gateway",
        "Rotate the database credentials on the payments gateway",
        False,
    ),
    (
        "Open firewall access from the app tier to the new Redis instance",
        "Open firewall access from the app tier to the new Postgres instance",
        False,
    ),
    (
        "Provision additional GPU nodes for model training",
        "Provision additional CPU nodes for the batch ETL jobs",
        False,
    ),
    (
        "Update the CI pipeline to run integration tests on every merge",
        "Update the CI pipeline to publish container images to the registry",
        False,
    ),
    (
        "Add a staging environment for the recommendations pipeline",
        "Add a staging environment for the billing pipeline",
        False,
    ),
    (
        "Urgent incident fix: checkout service returning 500s in production",
        "Urgent incident fix: login service returning 500s in production",
        False,
    ),
    (
        "Grant the data team write access to the reporting bucket",
        "Grant the data team read access to the raw events bucket",
        False,
    ),
    (
        "Scale up the search cluster ahead of the marketing campaign",
        "Scale up the recommendations cluster ahead of the marketing campaign",
        False,
    ),
    (
        "Increase the database connection pool to fix timeouts under load",
        "Increase the request timeout to reduce errors under load",
        False,
    ),
    (
        "Onboard the new notifications microservice into the platform",
        "Decommission the legacy notifications microservice from the platform",
        False,
    ),
]


def positives() -> list[tuple[str, str]]:
    return [(a, b) for a, b, dup in PAIRS if dup]


def negatives() -> list[tuple[str, str]]:
    return [(a, b) for a, b, dup in PAIRS if not dup]
