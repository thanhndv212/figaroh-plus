# figaroh Documentation

The docs site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material theme](https://squidfunk.github.io/mkdocs-material/), from
Markdown sources in `docs/source/`. The API reference pages use
[mkdocstrings](https://mkdocstrings.github.io/) to pull directly from
docstrings in `src/figaroh/`.

`docs/decisions/` (design-rationale documents) is intentionally **not**
part of the built site — `mkdocs.yml`'s `docs_dir` points at `docs/source/`,
a sibling directory, so `decisions/` is never read or copied.

## Building and viewing locally

1. Activate `figaroh-dev` and install the docs dependencies (from the repo root):

```bash
conda activate figaroh-dev
python -m pip install -e '.[docs]'
```

2. Live-reloading local preview:

```bash
mkdocs serve
```

Open <http://localhost:8000> — the page reloads automatically as you edit
files under `docs/source/`.

3. One-off static build:

```bash
mkdocs build
```

Output is written to `site/` (gitignored). Open `site/index.html` directly,
or serve it:

```bash
cd site
python -m http.server 8000
```

## Adding a page

1. Add the Markdown file under `docs/source/`, in the subfolder matching
   its kind:
   - `concepts/` — how a subsystem works (architecture, backends, config)
   - `tutorials/` — task-driven, narrated walkthroughs of a workflow
   - top-level `docs/source/` — standalone User Guide pages (e.g. Getting
     Started, Reporting & Verification)
   - `examples/` — gallery pages pointing into the
     [figaroh-examples](https://github.com/thanhndv212/figaroh-examples)
     repo (kept short; the full implementation lives in that repo, not here)
   - `api/` — mkdocstrings pages, one per `src/figaroh` subpackage
   - `further_reading/` — roadmap, changelog, FAQ, design decisions
2. Add it to the `nav:` section of `mkdocs.yml` at the repo root, under the
   matching top-level caption — MkDocs only shows pages listed there.

## Adding an API reference page

Add a page under `docs/source/api/` using an mkdocstrings directive:

```markdown
::: figaroh.some.module
    options:
      show_root_heading: false
```

This renders every documented class/function in that module from its
docstrings — no manual member lists to maintain.

## Deployment

`.github/workflows/docs.yml` builds and deploys the site to the `gh-pages`
branch after a successful build on pushes to `main` only. PRs and pushes to
`devel` build without deployment. The build uses the conda environment plus
the `docs` extra; an install/build error fails the job. The separate deploy job
consumes the built artifact with write permission. See [CONTRIBUTING](../CONTRIBUTING.md)
for branch rules and required checks.
