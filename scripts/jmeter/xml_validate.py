import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python xml_validate.py <file.xml>")
        return 2

    p = Path(sys.argv[1])
    try:
        ET.parse(p)
        print("XML_OK")
        return 0
    except Exception as e:
        print("XML_ERROR:", e)
        m = re.search(r"line (\d+), column (\d+)", str(e))
        if m:
            line = int(m.group(1))
            col = int(m.group(2))
            print(f"At line={line}, col={col}")
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                lines = p.read_text(errors="replace").splitlines()

            start = max(1, line - 20)
            end = min(len(lines), line + 20)
            for i in range(start, end + 1):
                prefix = ">>" if i == line else "  "
                print(f"{prefix}{i:04d}: {lines[i-1]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
