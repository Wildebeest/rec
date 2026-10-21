#!/usr/bin/env python3

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "recursive-logo"
DOTS = [(0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 0, 3), (1, 1, 4), (1, 2, 5), (0, 3, 6), (1, 3, 7)]
STIPPLE = 0b11011101


def polygons(file):
    result = []
    for path in ET.parse(file).getroot():
        tokens = iter(re.findall(r"[A-Za-z]|-?\d+(?:\.\d+)?", path.attrib["d"]))
        points = []
        x = y = 0
        for command in tokens:
            if command == "M":
                points = []
                x, y = float(next(tokens)), float(next(tokens))
                points.append((x, y))
            elif command == "L":
                x, y = float(next(tokens)), float(next(tokens))
                points.append((x, y))
            elif command == "H":
                x = float(next(tokens))
                points.append((x, y))
            elif command == "V":
                y = float(next(tokens))
                points.append((x, y))
            elif command == "Z":
                result.append(points)
            else:
                raise ValueError(f"Unsupported SVG command: {command}")
    return result


def winding(polygon, x, y):
    value = 0
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (bx - ax) * (y - ay) - (x - ax) * (by - ay)
        if ay <= y < by and cross > 0:
            value += 1
        if by <= y < ay and cross < 0:
            value -= 1
    return value


def cells(facets, columns):
    result = []
    for row in range(columns // 2):
        line = []
        for col in range(columns):
            masks = [0, 0, 0, 0]
            for dx, dy, bit in DOTS:
                x = (col * 2 + dx + 0.5) / (columns * 2) * 290
                y = (row * 4 + dy + 0.5) / (columns * 2) * 335
                for layer in range(4):
                    if sum(winding(p, x, y) for p in facets[layer]) != 0:
                        masks[layer] |= 1 << bit
                        break
            # One character has one color: clip it to one face instead of
            # averaging neighboring faces and the outline into a grey mesh.
            owner = max(range(3), key=lambda i: masks[i].bit_count()) if any(masks[:3]) else 3
            line.append({"glyph": chr(0x2800 + (masks[owner] & STIPPLE)), "owner": owner})
        result.append(line)
    return result


def main():
    facets = [polygons(ROOT / "assets" / f"logo-frame-{i}.svg") for i in range(1, 5)]
    shapes = {
        name: cells(facets, width)
        for name, width in (
            ("large", 20),
            ("small", 16),
        )
    }
    (ROOT / "pixels.json").write_text(
        json.dumps(shapes, ensure_ascii=False, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
