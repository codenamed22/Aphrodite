"""User profile data model for the matchmaking algorithm.

Follows the attribute set defined in the paper (Thaiprayoon & Unger, NLPIR 2023,
Table 1): each profile ``P_i`` has interest, hobby, occupation and biography
attributes ``{A_i1, A_i2, A_i3, A_i4}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# The three "term-based" attributes are embedded with word vectors (Eq. 1),
# while the "context-based" biography attribute is embedded with BERT (Eq. 2).
TERM_ATTRIBUTES: tuple[str, ...] = ("interests", "hobbies", "occupation")
CONTEXT_ATTRIBUTE: str = "biography"
ATTRIBUTES: tuple[str, ...] = TERM_ATTRIBUTES + (CONTEXT_ATTRIBUTE,)


@dataclass
class UserProfile:
    """A single user profile ``P_i``.

    Attributes
    ----------
    user_id:
        Stable unique identifier for the user (node id in the graph).
    name:
        Display name (not used by the algorithm, kept for readability).
    interests, hobbies, occupation:
        Term-based attributes. Each is a free-text string (e.g. a comma or
        space separated controlled vocabulary). These are embedded via the
        term-based model (Word2Vec mean, Eq. 1).
    biography:
        Context-based attribute: a short natural-language paragraph embedded
        via the context model (BERT, Eq. 2).
    """

    user_id: str
    name: str = ""
    interests: str = ""
    hobbies: str = ""
    occupation: str = ""
    biography: str = ""

    def attribute(self, name: str) -> str:
        """Return the raw text of the given attribute."""
        if name not in ATTRIBUTES:
            raise KeyError(f"Unknown attribute {name!r}; expected one of {ATTRIBUTES}")
        return getattr(self, name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "interests": self.interests,
            "hobbies": self.hobbies,
            "occupation": self.occupation,
            "biography": self.biography,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UserProfile":
        return cls(
            user_id=str(data["user_id"]),
            name=str(data.get("name", "")),
            interests=str(data.get("interests", "")),
            hobbies=str(data.get("hobbies", "")),
            occupation=str(data.get("occupation", "")),
            biography=str(data.get("biography", "")),
        )


def validate_profiles(profiles: Sequence[UserProfile]) -> None:
    """Ensure profile ids are unique and non-empty."""
    seen: set[str] = set()
    for p in profiles:
        if not p.user_id:
            raise ValueError("Every profile must have a non-empty user_id")
        if p.user_id in seen:
            raise ValueError(f"Duplicate user_id: {p.user_id!r}")
        seen.add(p.user_id)


__all__ = [
    "UserProfile",
    "validate_profiles",
    "ATTRIBUTES",
    "TERM_ATTRIBUTES",
    "CONTEXT_ATTRIBUTE",
]
