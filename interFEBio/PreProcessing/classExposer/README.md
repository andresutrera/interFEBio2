# classExposer tooling cheatsheet

Utilities in this package turn FEBio headers into Python-friendly metadata and
generate strongly-typed dataclass facades. This note keeps the command-line
invocations in one place so you do not have to re-discover flags every time.

## Prerequisites
- Python 3.10 or newer (matches `pyproject.toml`).
- `pip install -e .` from the repository root so `classExposer` is importable.
- `clang` Python bindings (pulled in automatically via the project
  dependencies) and the system `clang`/`libclang` runtime.
- A checkout of the FEBio source tree next to this project or the
  `FEBIO_ROOT` environment variable pointing at it. `classExposer.config`
  uses this to resolve include paths.

## Typical workflow
1. Extract metadata from one or more FEBio headers with the extractor CLI.
2. Feed the resulting JSON into the generator to write Python bindings.

### Metadata extraction
The extractor accepts one or more header paths and can emit a single JSON blob
or a directory containing one file per header. Include paths are forwarded to
clang via repeated `-I/--include` flags.

Example: write a single JSON file while pretty-printing the output.

```bash
python -m classExposer.extractor \
    ../FEBio/FECore/FEMat3dValuator.h \
    -o build/metadata/FEMat3dValuator.json \
    --pretty \
    -I ../FEBio
```

Example: keep extractor output grouped per header and prime manifest updates
without touching the on-disk manifest files.

```bash
python -m classExposer.extractor \
    ../FEBio/FECore/FEMat3dValuator.h \
    ../FEBio/FECore/FEMat3dsValuator.h \
    --dir build/metadata/Core \
    --manifest-dir src/classExposer/manifests \
    --category Core \
    --manifest-dry-run \
    -I ../FEBio
```

Key flags:
- `headers`: one or more FEBio header files to scan.
- `-o/--output`: write a single JSON payload to this path.
- `--dir`: emit one JSON file per header into the directory.
- `--pretty`: indent the JSON output.
- `--manifest-dir`: suggest manifest updates; requires `--category`.
- `--xml-tag` / `--xml-section`: override manifest metadata for quick tweaks.
- `--manifest-dry-run`: preview manifest edits without writing.
- `-I/--include`: forward additional include directories to clang (repeatable).

### Binding generation
The generator consumes the manifest(s) plus the metadata emitted by the
extractor. You can ask for a single consolidated module or a directory
containing one module per manifest category (Core, Materials, ...).

Example: regenerate the Core package while keeping files split by category.

```bash
python -m classExposer.generator \
    --manifest src/classExposer/manifests/Core.json \
    --metadata build/metadata/Core \
    --output-dir src/interFEBio
```

Example: collapse everything into one scratch module for quick inspection.

```bash
python -m classExposer.generator \
    --manifest src/classExposer/manifests \
    --metadata build/metadata \
    --output build/interFEBio_preview.py
```

Generator flags:
- `--manifest`: path to a manifest JSON file or directory to merge.
- `--metadata`: JSON metadata file or directory produced by the extractor.
- `--output`: write a single module containing every manifest entry.
- `--output-dir`: write a package tree keyed by manifest category.

## Handy snippets
- Resolve the FEBio root path that the tooling will use:

  ```python
  from classExposer.config import FEBIO_ROOT
  print(FEBIO_ROOT)
  ```

- Regenerate both Core and Materials packages after refreshing metadata:

  ```bash
  python -m classExposer.generator \
      --manifest src/classExposer/manifests \
      --metadata build/metadata \
      --output-dir src/interFEBio
  ```

Feel free to extend this README with additional recipes (e.g. automation
scripts, frequently-used include paths, or manifest curation tips) as the
workflow evolves.
