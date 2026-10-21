import re


def replace_body(source, name, transform):
    match = re.search(r"\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", source)
    if match is None:
        raise ValueError(f"Missing Metal function: {name}")
    start = match.end()
    depth, end = 1, start
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[:start] + transform(source[start : end - 1]) + source[end - 1 :]
