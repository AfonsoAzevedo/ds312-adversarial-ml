"""Format source with Ruff only if the parsed Python programs remain unchanged"""

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    #Capture the original source so a failed check cannot leave partial edits
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths.extend(sorted((ROOT / "scripts").rglob("*.py")))
    originals = {path: path.read_bytes() for path in paths}

    def restore() -> None:
        for path, contents in originals.items():
            path.write_bytes(contents)

    try:
        #Use Ruff to format the source without changing model algorithms
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", "src", "scripts"],
            check=True,
            cwd=ROOT,
        )

        #Compare parsed programs before and after formatting
        for path, old_contents in originals.items():
            original_tree = ast.dump(
                ast.parse(old_contents.decode("utf-8")),
                include_attributes=False,
            )
            formatted_tree = ast.dump(
                ast.parse(path.read_text(encoding="utf-8")),
                include_attributes=False,
            )
            if original_tree != formatted_tree:
                raise RuntimeError(f"Formatting changed program structure: {path}")

        #Confirm formatting preserved the project's unusual comment conventions
        subprocess.run(
            [sys.executable, "-m", "scripts.check_comment_style"],
            check=True,
            cwd=ROOT,
        )
    except Exception:
        restore()
        print("Source restored after unsuccessful formatting checks")
        raise

    changed = sum(
        path.read_bytes() != contents for path, contents in originals.items()
    )
    print(f"AST-preserving source formatting: PASSED ({changed} files changed)")


if __name__ == "__main__":
    main()
