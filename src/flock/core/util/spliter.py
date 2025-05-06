import re


def _split_fields(spec: str) -> list[str]:
    """Split *spec* on commas that are truly between fields (i.e. that
    are **not** inside brackets, braces, parentheses, string literals,
    or the free-form description).

    Strategy
    --------
    1. Single pass, char-by-char.
    2. Track    • nesting depth for (), [], {}
                • whether we are inside a quoted string
    3. Consider a comma a separator *only* if depth == 0, we are not
       inside quotes **and** the text that follows matches
       `optional-ws + identifier + optional-ws + ':'`
       which can only be the start of the next field.
    """
    fields: list[str] = []
    start = 0
    depth = 0
    quote_char: str | None = None
    i = 0
    ident_re = re.compile(r"[A-Za-z_]\w*")  # cheap Python identifier

    while i < len(spec):
        ch = spec[i]

        # ---------------- string handling ----------------
        if quote_char:
            if ch == quote_char:
                quote_char = None
            i += 1
            continue

        if ch in {"'", '"'}:
            # treat as string delimiter only in places regular Python would
            prev = spec[i - 1] if i else " "
            if depth or prev.isspace() or prev in "=([{,:":  # likely a quote
                quote_char = ch
            i += 1
            continue

        # ---------------- bracket / brace / paren ----------------
        if ch in "([{":
            depth += 1
            i += 1
            continue
        if ch in ")]}":
            depth = max(depth - 1, 0)
            i += 1
            continue

        # ---------------- field separators ----------------
        if ch == "," and depth == 0:
            j = i + 1
            while j < len(spec) and spec[j].isspace():
                j += 1
            if j < len(spec):
                id_match = ident_re.match(spec, j)
                if id_match:
                    k = id_match.end()
                    while k < len(spec) and spec[k].isspace():
                        k += 1
                    if k < len(spec) and spec[k] == ":":
                        # confirmed: the comma ends the current field
                        fields.append(spec[start:i].strip())
                        start = i + 1
                        i += 1
                        continue

        i += 1

    fields.append(spec[start:].strip())
    return [f for f in fields if f]  # prune empties


def parse_schema(spec: str) -> list[tuple[str, str, str]]:
    """Turn the raw *spec* into List[(name, python_type, description)]."""
    result: list[tuple[str, str, str]] = []

    for field in _split_fields(spec):
        if ":" not in field:
            raise ValueError(f"Malformed field (missing ':'): {field!r}")

        name_part, rest = field.split(":", 1)
        name = name_part.strip()

        type_part, _, desc_part = rest.partition("|")
        type_str = type_part.strip()
        description = desc_part.strip()

        result.append((name, type_str, description))

    return result


# ------------------------------ demo ------------------------------
if __name__ == "__main__":
    SAMPLE_1 = (
        " name: str | The character's full name,"
        "race: str | The character's fantasy race,"
        "class: Literal['mage','thief'] | The character's profession,"
        "background: str | A brief backstory for the character"
    )

    SAMPLE_2 = (
        "field_with_internal_quotes: Literal['type_A', "
        '"type_B_with_\'_apostrophe"] | A literal with mixed quotes,'
        " another_field: str"
    )

    print(f"Sample 1: {SAMPLE_1}")
    print(f"Sample 2: {SAMPLE_2}")

    print("\nSplitting Sample 1:")
    split_sample_1 = _split_fields(SAMPLE_1)
    print(split_sample_1)
    for field in split_sample_1:
        print(parse_schema(field))

    print("\nSplitting Sample 2:")
    split_sample_2 = _split_fields(SAMPLE_2)
    print(split_sample_2)
    for field in split_sample_2:
        print(parse_schema(field))
