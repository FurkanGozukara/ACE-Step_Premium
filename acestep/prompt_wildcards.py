"""Wildcard expansion for user-facing prompt text fields."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


class _RandomChoice(Protocol):
    def choice(self, seq): ...


@dataclass(frozen=True)
class _Literal:
    text: str


@dataclass(frozen=True)
class _Sequence:
    parts: tuple["_Node", ...]


@dataclass(frozen=True)
class _Choice:
    options: tuple[_Sequence, ...]


_Node = _Literal | _Sequence | _Choice


class WildcardSyntaxError(ValueError):
    """Raised when prompt wildcard brace syntax is malformed."""

    def __init__(self, message: str, *, position: int | None = None) -> None:
        self.position = position
        super().__init__(message)


@dataclass(frozen=True)
class PromptWildcardExpansion:
    """Expanded prompt fields plus a summary of fields that used wildcards."""

    captions: str
    lyrics: str
    flow_edit_source_caption: str
    flow_edit_source_lyrics: str
    expanded_fields: tuple[str, ...] = ()


WILDCARD_HELP_MARKDOWN = (
    "**Wildcards:** use `{option A|option B|option C}`; one option is picked "
    "when you generate. Nested example: `cinematic {piano|guitar {clean|crunchy}} "
    "hook`. Lyrics example: `I feel {alive|ready|free} tonight`. Lyric tags "
    "like `[Verse]` and `[Instrumental]` stay unchanged."
)


def expand_prompt_wildcards(
    text: object,
    *,
    rng: _RandomChoice | None = None,
) -> str:
    """Return ``text`` with prompt wildcards expanded.

    Wildcards use curly braces with top-level pipe-separated options:
    ``{a|b|c}``. Bracketed lyric tags such as ``[Verse]`` are preserved
    as literal text.
    """

    source = "" if text is None else str(text)
    parser = _Parser(source)
    ast = parser.parse()
    return _expand_node(ast, rng or random)


def prompt_uses_wildcards(text: object) -> bool:
    """Return whether ``text`` contains at least one valid wildcard group."""

    source = "" if text is None else str(text)
    parser = _Parser(source)
    ast = parser.parse()
    return _has_choice(ast)


def expand_generation_prompt_fields(
    *,
    captions: object,
    lyrics: object,
    flow_edit_source_caption: object = "",
    flow_edit_source_lyrics: object = "",
    rng: _RandomChoice | None = None,
) -> PromptWildcardExpansion:
    """Expand wildcard syntax in generation prompt/caption/lyrics fields."""

    randomizer = rng or random
    fields = (
        ("Style", "captions", captions),
        ("Lyrics", "lyrics", lyrics),
        ("Flow Edit Source Style", "flow_edit_source_caption", flow_edit_source_caption),
        ("Flow Edit Source Lyrics", "flow_edit_source_lyrics", flow_edit_source_lyrics),
    )
    values: dict[str, str] = {}
    expanded: list[str] = []
    for label, key, value in fields:
        try:
            source = "" if value is None else str(value)
            parser = _Parser(source)
            ast = parser.parse()
            if _has_choice(ast):
                expanded.append(label)
            values[key] = _expand_node(ast, randomizer)
        except WildcardSyntaxError as exc:
            raise WildcardSyntaxError(
                _format_field_error(label, exc),
                position=exc.position,
            ) from exc

    return PromptWildcardExpansion(
        captions=values["captions"],
        lyrics=values["lyrics"],
        flow_edit_source_caption=values["flow_edit_source_caption"],
        flow_edit_source_lyrics=values["flow_edit_source_lyrics"],
        expanded_fields=tuple(expanded),
    )


def _format_field_error(label: str, exc: WildcardSyntaxError) -> str:
    detail = str(exc)
    if exc.position is None:
        return f"Wildcard syntax error in {label}: {detail}"
    return (
        f"Wildcard syntax error in {label}: {detail} "
        f"(character {exc.position + 1})."
    )


def _expand_node(node: _Node, rng: _RandomChoice) -> str:
    if isinstance(node, _Literal):
        return node.text
    if isinstance(node, _Sequence):
        return "".join(_expand_node(part, rng) for part in node.parts)
    option = rng.choice(node.options)
    return _expand_node(option, rng)


def _has_choice(node: _Node) -> bool:
    if isinstance(node, _Choice):
        return True
    if isinstance(node, _Sequence):
        return any(_has_choice(part) for part in node.parts)
    return False


def _merge_parts(parts: list[_Node]) -> tuple[_Node, ...]:
    merged: list[_Node] = []
    for part in parts:
        if isinstance(part, _Literal) and not part.text:
            continue
        if merged and isinstance(merged[-1], _Literal) and isinstance(part, _Literal):
            merged[-1] = _Literal(merged[-1].text + part.text)
        else:
            merged.append(part)
    return tuple(merged)


class _Parser:
    """Small recursive parser for nested wildcard brace groups."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.length = len(text)

    def parse(self) -> _Sequence:
        parts: list[_Node] = []
        index = 0
        while index < self.length:
            char = self.text[index]
            if char == "{":
                node, index = self._parse_group(index)
                parts.append(node)
            elif char == "}":
                raise WildcardSyntaxError("Unexpected closing }.", position=index)
            else:
                literal, index = self._read_literal(index, stop_on_pipe=False)
                parts.append(_Literal(literal))
        return _Sequence(_merge_parts(parts))

    def _parse_group(self, start: int) -> tuple[_Node, int]:
        options: list[list[_Node]] = [[]]
        separators = 0
        index = start + 1
        while index < self.length:
            char = self.text[index]
            if char == "{":
                node, index = self._parse_group(index)
                options[-1].append(node)
            elif char == "}":
                index += 1
                if separators:
                    option_nodes = tuple(
                        _Sequence(_merge_parts(option)) for option in options
                    )
                    return _Choice(option_nodes), index
                literal_parts: list[_Node] = [_Literal("{")]
                literal_parts.extend(options[0])
                literal_parts.append(_Literal("}"))
                return _Sequence(_merge_parts(literal_parts)), index
            elif char == "|":
                separators += 1
                options.append([])
                index += 1
            else:
                literal, index = self._read_literal(index, stop_on_pipe=True)
                options[-1].append(_Literal(literal))

        raise WildcardSyntaxError("Missing closing } for wildcard group.", position=start)

    def _read_literal(self, index: int, *, stop_on_pipe: bool) -> tuple[str, int]:
        start = index
        while index < self.length:
            char = self.text[index]
            if char in "{}":
                break
            if stop_on_pipe and char == "|":
                break
            index += 1
        return self.text[start:index], index
