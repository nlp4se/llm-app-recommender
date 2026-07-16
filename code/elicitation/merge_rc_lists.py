"""Semantic alignment and overlap statistics across RC lists (open/proprietary × small/large).

Criteria are grouped by human/LLM semantic judgment (not embedding similarity).
Sub-criteria are treated as equivalent to their parent concept (e.g. In-App Purchases → Cost).
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from code.experiments.criteria import load_rq3_criteria

METHODS = (
    "open_small",
    "open_large",
    "proprietary_small",
    "proprietary_large",
)

DEFAULT_RC_PATHS: dict[str, str] = {
    "open_small": "data/output/features/rq1/rc/open_small/rc_wo_id_open_small.csv",
    "open_large": "data/output/features/rq1/rc/open_large/rc_wo_id_open_large.csv",
    # Knowledge-only proprietary small (config A); web-augmented legacy is proprietary_small/
    "proprietary_small": (
        "data/output/features/rq1/rc/proprietary_small_wo_websearch/"
        "rc_wo_id_proprietary_small_wo_websearch.csv"
    ),
    "proprietary_large": "data/output/features/rq1/rc/proprietary_large/rc_wo_id_proprietary_large.csv",
}

# Final taxonomy used for RQ3 (16 criteria). Storage Efficiency is tracked in alignment
# statistics only when present in a method list; it is excluded from the unified export.
CANONICAL_UNIFIED_NAMES: tuple[str, ...] = (
    "Cost and Pricing",
    "Customer Support",
    "Customization Options",
    "Platform and Device Compatibility",
    "Popularity and Market Reach",
    "Regular Updates and Maintenance",
    "Security and Privacy",
    "User Interface and Experience",
    "User Ratings",
    "Integration with Other Services",
    "Performance and Stability",
    "Regional and Geographical Availability",
    "User Engagement",
    "Accessibility Features",
    "Feature Breadth and Richness",
    "Cross-Platform Synchronization",
)

# Curated semantic equivalence groups. Each entry maps method → criterion name(s) plus a merged
# name/description synthesised from the commonalities across methods.
# Multiple names in one method (semicolon-separated) denote sub-criteria folded into the same concept.
SEMANTIC_GROUPS: list[dict[str, str]] = [
    {
        "merged_name": "Cost and Pricing",
        "merged_description": (
            "The pricing model of the app, including free tiers, one-time purchases, subscriptions, "
            "in-app purchases, and overall value for money."
        ),
        "open_small": "Cost",
        "open_large": "Cost & Subscription Model",
        "proprietary_small": "Pricing and Value for Money",
        "proprietary_large": "Pricing and Value",
    },
    {
        "merged_name": "Security and Privacy",
        "merged_description": (
            "Protection of user data and communications, including encryption, access controls, "
            "privacy policies, and compliance with privacy regulations."
        ),
        "open_small": "Security",
        "open_large": "Data Privacy & Security",
        "proprietary_small": "Security and Privacy",
        "proprietary_large": "Security and Privacy",
    },
    {
        "merged_name": "Customer Support",
        "merged_description": (
            "The availability, responsiveness, and quality of customer support for user queries, "
            "troubleshooting, and issue resolution."
        ),
        "open_small": "Customer Support",
        "open_large": "Customer Support",
        "proprietary_small": "Customer Support",
        "proprietary_large": "Customer Support",
    },
    {
        "merged_name": "Customization Options",
        "merged_description": (
            "The extent to which users can personalize the app's appearance, themes, layout, "
            "and behavior to suit their preferences."
        ),
        "open_small": "Customization Options",
        "open_large": "Customization Options",
        "proprietary_large": "Customization Options",
    },
    {
        "merged_name": "User Interface and Experience",
        "merged_description": (
            "The ease of use, navigation, and overall user experience of the app, including "
            "interface design and intuitiveness."
        ),
        "open_small": "User Interface",
        "open_large": "User Interface",
        "proprietary_small": "User Experience and Accessibility",
        "proprietary_large": "User Experience (UX)",
    },
    {
        "merged_name": "Platform and Device Compatibility",
        "merged_description": (
            "Support for the app across operating systems, devices, and platforms "
            "(e.g., iOS, Android, web), affecting user accessibility."
        ),
        "open_small": "Device Compatibility",
        "open_large": "Platform Availability",
        "proprietary_small": "Cross-Platform Availability",
        "proprietary_large": "Cross-Platform Availability",
    },
    {
        "merged_name": "Regional and Geographical Availability",
        "merged_description": (
            "The countries, regions, and languages in which the app is available and offers "
            "localized content or services."
        ),
        "open_small": "Regional Availability",
        "open_large": "Regional Availability",
        "proprietary_small": "Global Availability",
        "proprietary_large": "Geographic Coverage",
    },
    {
        "merged_name": "Integration with Other Services",
        "merged_description": (
            "Compatibility and seamless integration with other apps and platforms, such as email, "
            "messaging, cloud storage, and social media."
        ),
        "open_small": "Integration with Other Services",
        "open_large": "Integration",

    },
    {
        "merged_name": "Performance and Stability",
        "merged_description": (
            "The app's speed, responsiveness, and reliability, including stable operation without "
            "crashes, lag, or performance degradation under load."
        ),
        "open_small": "Mobile App Performance",
        "open_large": "Performance & Stability",
        "proprietary_small": "App Performance and Stability",
        "proprietary_large": "Performance and Stability",
    },
    {
        "merged_name": "User Ratings",
        "merged_description": (
            "Average rating from app store reviews, reflecting overall user satisfaction and "
            "perceived quality of the app."
        ),
        "open_small": "User Ratings",
        "open_large": "User Rating",
        "proprietary_small": "User Ratings and Reviews",
        "proprietary_large": "User Rating",
    },
    {
        "merged_name": "User Engagement",
        "merged_description": (
            "How actively and frequently users interact with the app, measured through metrics such "
            "as session frequency, retention, and daily active usage."
        ),
        "open_small": "User Engagement",
        "open_large": "User Engagement",
        "proprietary_small": "Active User Base",
    },
    {
        "merged_name": "Popularity and Market Reach",
        "merged_description": (
            "The app's adoption and audience size, including download or install counts and the "
            "number of active users on the platform."
        ),
        "open_small": "Popularity",
        "open_large": "Popularity and Downloads",
        "proprietary_small": "Monthly Active Users (MAU); User Base & Popularity",
        "proprietary_large": "Number of downloads; User Engagement",
    },
    {
        "merged_name": "Regular Updates and Maintenance",
        "merged_description": (
            "The frequency and quality of app updates, including bug fixes, performance improvements, "
            "new features, and ongoing maintenance."
        ),
        "open_small": "Regular Updates & Maintenance",
        "open_large": "Regular Updates",
        "proprietary_small": "Update Frequency and Innovation",
        "proprietary_large": "Update and Maintenance Frequency",
    },
    {
        "merged_name": "Accessibility Features",
        "merged_description": (
            "Incorporation of accessibility features for users with disabilities, such as screen "
            "reader support, captions, and other inclusive design elements."
        ),
        "open_small": "Accessibility Features",
        "open_large": "Accessibility Features",
    },
    {
        "merged_name": "Feature Breadth and Richness",
        "merged_description": (
            "The range, depth, and quality of features offered by the app, including core "
            "functionality and optional capabilities."
        ),
        "proprietary_large": "Feature Richness",
    },
    {
        "merged_name": "Cross-Platform Synchronization",
        "merged_description": (
            "The ability to keep data and user experience consistent and synchronized across "
            "multiple devices and platforms."
        ),
        "proprietary_large": "Cross-Platform Synchronization",
    },
]

_META_KEYS = frozenset({"merged_name", "merged_description"})


@dataclass
class EquivalenceGroup:
    merged_name: str
    merged_description: str
    members: dict[str, str] = field(default_factory=dict)  # method -> name(s)

    @property
    def methods(self) -> set[str]:
        return set(self.members)

    @property
    def method_count(self) -> int:
        return len(self.members)

    def canonical_name(self) -> str:
        return self.merged_name


def _split_names(cell: str) -> list[str]:
    return [part.strip() for part in cell.split(";") if part.strip()]


def load_method_criteria_names(paths: dict[str, str]) -> dict[str, set[str]]:
    by_method: dict[str, set[str]] = {m: set() for m in METHODS}
    for method, path in paths.items():
        for row in load_rq3_criteria(path):
            by_method[method].add(row["n"])
    return by_method


def validate_semantic_groups(
    groups: list[dict[str, str]],
    criteria_by_method: dict[str, set[str]],
) -> None:
    assigned: dict[str, set[str]] = {m: set() for m in METHODS}
    for group in groups:
        if "merged_name" not in group or "merged_description" not in group:
            raise ValueError("Each semantic group must define merged_name and merged_description")
        for method, names_cell in group.items():
            if method in _META_KEYS:
                continue
            if method not in METHODS:
                raise ValueError(f"Unknown method in semantic group: {method}")
            for name in _split_names(names_cell):
                if name not in criteria_by_method[method]:
                    raise ValueError(
                        f"Unknown criterion '{name}' for {method}; "
                        f"available: {sorted(criteria_by_method[method])}"
                    )
                if name in assigned[method]:
                    raise ValueError(f"Criterion '{name}' assigned to more than one group ({method})")
                assigned[method].add(name)

    for method in METHODS:
        missing = criteria_by_method[method] - assigned[method]
        if missing:
            raise ValueError(f"Unassigned criteria for {method}: {sorted(missing)}")
        extra = assigned[method] - criteria_by_method[method]
        if extra:
            raise ValueError(f"Unknown assigned criteria for {method}: {sorted(extra)}")


def _group_members(group: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in group.items() if k not in _META_KEYS}


def semantic_groups_to_equivalence(groups: list[dict[str, str]]) -> list[EquivalenceGroup]:
    parsed = [
        EquivalenceGroup(
            merged_name=g["merged_name"],
            merged_description=g["merged_description"],
            members=_group_members(g),
        )
        for g in groups
    ]
    parsed.sort(key=lambda g: (-g.method_count, g.merged_name.lower()))
    return parsed


def groups_to_merge_df(groups: list[EquivalenceGroup]) -> pd.DataFrame:
    rows = [{method: group.members.get(method, "") for method in METHODS} for group in groups]
    return pd.DataFrame(rows, columns=list(METHODS))


def groups_to_unified_df(groups: list[EquivalenceGroup]) -> pd.DataFrame:
    canonical = list(CANONICAL_UNIFIED_NAMES)
    by_name = {group.merged_name: group for group in groups}
    missing = [name for name in canonical if name not in by_name]
    if missing:
        raise ValueError(f"Canonical unified criteria missing from semantic groups: {missing}")
    rows = [
        {"name": name, "description": by_name[name].merged_description}
        for name in canonical
    ]
    return pd.DataFrame(rows, columns=["name", "description"])


def compute_statistics(groups: list[EquivalenceGroup], counts: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(metric: str, value: str | int | float, detail: str = "") -> None:
        rows.append({"metric": metric, "value": value, "detail": detail})

    total_groups = len(groups)
    add("total_equivalence_groups", total_groups)
    add("total_distinct_rc_concepts", total_groups)

    for method in METHODS:
        add(f"rc_count_{method}", counts[method])

    for a, b in itertools.combinations(METHODS, 2):
        overlap = sum(1 for g in groups if a in g.methods and b in g.methods)
        add(f"pair_overlap_{a}__{b}", overlap)

    for k in (4, 3, 2, 1):
        matched = [g for g in groups if g.method_count == k]
        add(f"groups_with_{k}_methods", len(matched))
        if matched:
            labels = "; ".join(g.canonical_name() for g in matched)
            add(f"groups_with_{k}_methods_list", labels)

    multi = [g for g in groups if g.method_count >= 2]
    add("groups_with_at_least_2_methods", len(multi))
    add(
        "overall_overlap_rate",
        round(len(multi) / total_groups, 4) if total_groups else 0,
        "Fraction of equivalence groups shared by 2+ methods",
    )

    all_four = [g for g in groups if g.method_count == 4]
    if all_four:
        add("common_all_4_methods", "; ".join(g.canonical_name() for g in all_four))

    return pd.DataFrame(rows)


def merge_rc_lists(
    *,
    rc_paths: dict[str, str],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    criteria_by_method = load_method_criteria_names(rc_paths)
    counts = {m: len(criteria_by_method[m]) for m in METHODS}
    validate_semantic_groups(SEMANTIC_GROUPS, criteria_by_method)
    groups = semantic_groups_to_equivalence(SEMANTIC_GROUPS)

    merge_df = groups_to_merge_df(groups)
    stats_df = compute_statistics(groups, counts)

    merge_path = output / "rc_merge_alignment.csv"
    unified_path = output / "rc_merge_unified.csv"
    stats_path = output / "rc_merge_statistics.csv"
    merge_df.to_csv(merge_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    new_unified_df = groups_to_unified_df(groups)
    if unified_path.is_file():
        existing = pd.read_csv(unified_path)
        if (
            list(existing["name"]) == list(new_unified_df["name"])
            and list(existing["description"]) == list(new_unified_df["description"])
        ):
            # Preserve file formatting when taxonomy content is unchanged.
            pass
        else:
            new_unified_df.to_csv(unified_path, index=False)
    else:
        new_unified_df.to_csv(unified_path, index=False)

    return {"alignment": merge_path, "unified": unified_path, "statistics": stats_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic RC alignment across method suites.")
    parser.add_argument(
        "--output-dir",
        default="data/output/features/rq1/rc/merge",
        help="Output folder for merge CSVs.",
    )
    for method in METHODS:
        parser.add_argument(
            f"--{method.replace('_', '-')}",
            default=DEFAULT_RC_PATHS[method],
            help=f"Path to rc_wo_id for {method}",
        )
    args = parser.parse_args(argv)

    paths = {
        "open_small": args.open_small,
        "open_large": args.open_large,
        "proprietary_small": args.proprietary_small,
        "proprietary_large": args.proprietary_large,
    }
    artifacts = merge_rc_lists(rc_paths=paths, output_dir=args.output_dir)
    for key, path in artifacts.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
