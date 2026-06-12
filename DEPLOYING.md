# Deploying this site

This is an [MkDocs](https://www.mkdocs.org/) site using the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. It is published to GitHub Pages
(the `gh-pages` branch) and served at the URL set as `site_url` in `mkdocs.yml`.

**These steps are identical for every SDD course repository.**

## 1. Install the tools (once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install mkdocs mkdocs-material pymdown-extensions
```

## 2. Find the site sources

The MkDocs project (`mkdocs.yml` + `docs/`) lives in either:

- `src/` — stat-ml, machine-learning, deep-learning, DE; or
- the repository root — reinforcement-learning, SupaeroDataScience.github.io.

`cd` into that directory before running any `make` command below.

## 3. Preview locally

```bash
make serve     # serves at http://localhost:8000 with live reload
```

## 4. Publish

From the repository's **default branch** (`main` or `master`), with your changes committed:

```bash
make deploy    # runs `mkdocs gh-deploy`
```

`mkdocs gh-deploy` builds the site from your current working tree and force-pushes the result to
the `gh-pages` branch on `origin`. It never checks out or modifies your source branch. Wait ~1
minute, then refresh the public URL to see the change.

Other targets: `make build` (build into `site/` without publishing) and `make clean`.

## Slides (DE and deep-learning only)

These two repos embed reveal.js slides that are **pre-built and committed** under
`src/docs/slides/`, so the steps above need no extra tooling. You only touch slides if you edit a
source deck in `src/reveal/*.md`; then rebuild the HTML and commit it:

```bash
npm install -g reveal-md   # once
make slides                # regenerates src/docs/slides/ from src/reveal/
git add docs/slides && git commit -m "docs(slides): rebuild slides"
```
