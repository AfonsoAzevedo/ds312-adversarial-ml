"""Check Python comment conventions without modifying any source file"""

import io
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    #Review all Python source files, including the audit scripts
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths.extend(sorted((ROOT / "scripts").rglob("*.py")))
    violations = []
    long_lines = []

    for path in paths:
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)

        #Tokenize comments so hashes inside strings are not mistaken for comments
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                continue

            comment = token.string
            if comment.startswith("# ") or comment.endswith("."):
                violations.append(f"{relative}:{token.start[0]}: {comment}")
            if "----" in comment or "====" in comment:
                violations.append(f"{relative}:{token.start[0]}: decorative comment")

        #Report lengthy source lines for a separate readability review
        for number, line in enumerate(source.splitlines(), start=1):
            if len(line) > 88:
                long_lines.append(f"{relative}:{number}: {len(line)} characters")

    print(f"Python files checked: {len(paths)}")
    print(f"Comment convention violations: {len(violations)}")
    print(f"Lines longer than 88 characters: {len(long_lines)}")

    for issue in violations:
        print(issue)

    if long_lines:
        print("Long-line examples:")
        for issue in long_lines[:10]:
            print(issue)

    if violations:
        raise SystemExit("Comment convention check FAILED")

    print("Comment convention check PASSED")


if __name__ == "__main__":
    main()
