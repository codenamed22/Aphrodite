"""Synthetic user-profile dataset generation.

The paper evaluates on 150 GPT-generated profiles that are not publicly
available. To make Algorithm 1 runnable and testable end-to-end, this module
generates profiles organised into latent interest *clusters*: users in the same
cluster share vocabulary across their attributes, so the ground-truth match set
for a user is the other members of its cluster.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .profiles import UserProfile


@dataclass(frozen=True)
class Theme:
    name: str
    interests: tuple[str, ...]
    hobbies: tuple[str, ...]
    occupations: tuple[str, ...]
    bio_keywords: tuple[str, ...]


THEMES: tuple[Theme, ...] = (
    Theme(
        "technology",
        ("programming", "machine learning", "artificial intelligence", "robotics",
         "data science", "software", "algorithms", "open source"),
        ("coding", "building computers", "hackathons", "gaming", "electronics"),
        ("software engineer", "data scientist", "developer", "researcher"),
        ("technology", "software", "engineering", "innovation", "computers"),
    ),
    Theme(
        "sports",
        ("football", "basketball", "running", "fitness", "cycling", "tennis",
         "swimming", "athletics"),
        ("gym workouts", "marathons", "hiking", "playing soccer", "climbing"),
        ("coach", "athlete", "personal trainer", "physiotherapist"),
        ("sports", "fitness", "training", "competition", "outdoors"),
    ),
    Theme(
        "arts",
        ("painting", "drawing", "sculpture", "photography", "design",
         "illustration", "ceramics", "art history"),
        ("visiting galleries", "sketching", "digital art", "crafting", "sculpting"),
        ("artist", "graphic designer", "illustrator", "curator"),
        ("art", "creativity", "design", "visual", "expression"),
    ),
    Theme(
        "music",
        ("guitar", "piano", "jazz", "classical music", "songwriting",
         "music production", "singing", "drums"),
        ("playing guitar", "composing", "attending concerts", "djing", "band practice"),
        ("musician", "composer", "music producer", "sound engineer"),
        ("music", "melody", "rhythm", "performance", "composition"),
    ),
    Theme(
        "cooking",
        ("cooking", "baking", "cuisine", "gastronomy", "recipes",
         "food science", "pastry", "grilling"),
        ("trying restaurants", "baking bread", "meal prep", "wine tasting", "gardening"),
        ("chef", "baker", "food blogger", "nutritionist"),
        ("food", "cooking", "flavors", "cuisine", "culinary"),
    ),
    Theme(
        "travel",
        ("travel", "hiking", "backpacking", "photography", "geography",
         "cultures", "languages", "adventure"),
        ("exploring cities", "camping", "road trips", "scuba diving", "mountaineering"),
        ("travel writer", "tour guide", "pilot", "photographer"),
        ("travel", "adventure", "exploration", "world", "cultures"),
    ),
)


def _sample(rng: random.Random, pool: Iterable[str], lo: int, hi: int) -> list[str]:
    pool = list(pool)
    count = min(len(pool), rng.randint(lo, hi))
    return rng.sample(pool, count)


def generate_dataset(
    n_users: int = 60,
    themes: tuple[Theme, ...] = THEMES,
    noise: float = 0.15,
    seed: int = 42,
) -> tuple[list[UserProfile], dict[str, set[str]]]:
    """Generate synthetic profiles plus their ground-truth match sets.

    Parameters
    ----------
    n_users:
        Number of profiles to generate.
    themes:
        Interest themes; users are assigned round-robin across them.
    noise:
        Probability of injecting a cross-theme word into an attribute, making
        the matching problem non-trivial.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (profiles, ground_truth)
        ``ground_truth[user_id]`` is the set of other users sharing the theme.
    """
    rng = random.Random(seed)
    profiles: list[UserProfile] = []
    cluster_of: dict[str, int] = {}

    for i in range(n_users):
        theme_idx = i % len(themes)
        theme = themes[theme_idx]
        uid = f"u{i:03d}"

        interests = _sample(rng, theme.interests, 3, 5)
        hobbies = _sample(rng, theme.hobbies, 2, 4)
        occupation = _sample(rng, theme.occupations, 1, 2)

        # Inject occasional cross-theme noise.
        if rng.random() < noise:
            other = themes[rng.randrange(len(themes))]
            interests.append(rng.choice(other.interests))

        kw = _sample(rng, theme.bio_keywords, 2, 3)
        biography = (
            f"I am passionate about {kw[0]} and love spending time on "
            f"{interests[0]}. Professionally I work as a {occupation[0]}, and in "
            f"my free time I enjoy {hobbies[0]}."
        )

        profiles.append(
            UserProfile(
                user_id=uid,
                name=f"User {i}",
                interests=", ".join(interests),
                hobbies=", ".join(hobbies),
                occupation=", ".join(occupation),
                biography=biography,
            )
        )
        cluster_of[uid] = theme_idx

    ground_truth: dict[str, set[str]] = {}
    for uid, c in cluster_of.items():
        ground_truth[uid] = {other for other, oc in cluster_of.items()
                             if oc == c and other != uid}
    return profiles, ground_truth


def save_dataset(
    path: str | Path,
    profiles: list[UserProfile],
    ground_truth: dict[str, set[str]],
) -> None:
    payload = {
        "profiles": [p.as_dict() for p in profiles],
        "ground_truth": {uid: sorted(matches) for uid, matches in ground_truth.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_dataset(path: str | Path) -> tuple[list[UserProfile], dict[str, set[str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = [UserProfile.from_dict(d) for d in payload["profiles"]]
    ground_truth = {uid: set(matches) for uid, matches in payload["ground_truth"].items()}
    return profiles, ground_truth


__all__ = [
    "Theme",
    "THEMES",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
]
