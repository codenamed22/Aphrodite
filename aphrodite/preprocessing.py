"""Text preprocessing pipeline (paper §3.1.1).

Implements the modules described in the paper:

* Tokenization
* Lowercasing
* Removing stop words
* Removing special characters
* Stemming and lemmatization

The pipeline is dependency-light by default (bundled stop-word list, regex
tokenizer, and a compact stemmer) so it runs fully offline. When ``use_nltk`` is
enabled and NLTK is installed, the more faithful Porter stemmer and WordNet
lemmatizer are used instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# A compact but reasonable English stop-word list. Kept inline so the package
# has no data-download dependency for the offline path.
_DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have he her his i in is it its of on
    or that the to was were will with you your yours we they them their this
    these those but not no nor so if then than too very can could would should
    do does did done just about above after again all am any because been
    before being below between both down during each few further here how into
    more most other out over own same some such only once against up down off
    where which who whom why what when while our ours my mine me us also into
    """.split()
)


def _simple_stem(token: str) -> str:
    """A tiny, deterministic suffix-stripping stemmer (offline fallback).

    This is intentionally lightweight: it normalises common inflectional
    suffixes so that e.g. "running"/"runs"/"run" collapse together well enough
    for term matching. For faithful Porter stemming, enable ``use_nltk``.
    """
    for suffix in ("ational", "ization", "iveness", "fulness", "ousness",
                   "ing", "edly", "ies", "ied", "ily", "ely", "ual",
                   "ers", "er", "ed", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            # Undouble a trailing consonant left by verb suffixes so that
            # e.g. "running" -> "runn" -> "run" matches "runs" -> "run".
            if suffix in ("ing", "edly", "ed") and len(stem) >= 2 \
                    and stem[-1] == stem[-2] and stem[-1] not in "aeioulsz":
                stem = stem[:-1]
            return stem
    return token


@dataclass
class Preprocessor:
    """Configurable text preprocessing pipeline.

    Parameters
    ----------
    lowercase:
        Convert all tokens to lower case.
    remove_stopwords:
        Drop tokens that appear in the stop-word list.
    stem:
        Apply stemming/lemmatization to each token.
    stopwords:
        Optional custom stop-word set (defaults to the bundled list).
    use_nltk:
        If True and NLTK is available, use PorterStemmer + WordNetLemmatizer
        and NLTK's stop-word list where possible.
    """

    lowercase: bool = True
    remove_stopwords: bool = True
    stem: bool = True
    stopwords: frozenset[str] = _DEFAULT_STOPWORDS
    use_nltk: bool = False

    _token_re = re.compile(r"[A-Za-z0-9]+")

    def __post_init__(self) -> None:
        self._nltk_stemmer = None
        self._nltk_lemmatizer = None
        if self.use_nltk:
            self._init_nltk()

    def _init_nltk(self) -> None:
        try:  # pragma: no cover - exercised only when nltk installed
            import nltk
            from nltk.corpus import stopwords as nltk_stopwords
            from nltk.stem import PorterStemmer, WordNetLemmatizer

            for pkg, path in (("stopwords", "corpora/stopwords"),
                              ("wordnet", "corpora/wordnet")):
                try:
                    nltk.data.find(path)
                except LookupError:
                    nltk.download(pkg, quiet=True)

            self._nltk_stemmer = PorterStemmer()
            self._nltk_lemmatizer = WordNetLemmatizer()
            self.stopwords = frozenset(nltk_stopwords.words("english"))
        except Exception:
            # Fall back silently to the offline pipeline.
            self.use_nltk = False
            self._nltk_stemmer = None
            self._nltk_lemmatizer = None

    def tokenize(self, text: str) -> list[str]:
        """Split text into alphanumeric tokens, dropping special characters."""
        if not text:
            return []
        return self._token_re.findall(text)

    def _stem_token(self, token: str) -> str:
        if self.use_nltk and self._nltk_lemmatizer is not None:
            # Lemmatize then stem, mirroring the paper's "stemming and
            # lemmatization" step.
            lemma = self._nltk_lemmatizer.lemmatize(token)
            return self._nltk_stemmer.stem(lemma)
        return _simple_stem(token)

    def process(self, text: str) -> list[str]:
        """Run the full preprocessing pipeline and return a list of tokens."""
        tokens = self.tokenize(text)
        out: list[str] = []
        for tok in tokens:
            if self.lowercase:
                tok = tok.lower()
            if self.remove_stopwords and tok in self.stopwords:
                continue
            if self.stem:
                tok = self._stem_token(tok)
            if tok:
                out.append(tok)
        return out

    def process_many(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.process(t) for t in texts]


__all__ = ["Preprocessor"]
