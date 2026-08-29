#!/bin/bash
# fail if any source file under src/ docs/ scripts/ submission/ is gitignored
bad=$(git ls-files --others --ignored --exclude-standard -- src docs scripts submission 2>/dev/null | grep -vE '__pycache__|\.pyc$' || true)
if [ -n "$bad" ]; then echo "ERROR: source files are gitignored:"; echo "$bad"; exit 1; fi
echo "ok: no source file is ignored"
