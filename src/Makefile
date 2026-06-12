# SDD course site — Deep Learning (FSD311)
# Deployment is standardised across all SDD course repos: see DEPLOYING.md.
# Requires: pip install mkdocs mkdocs-material pymdown-extensions

serve:   ## Live-preview the site locally at http://localhost:8000
	mkdocs serve

build:   ## Build the static site into site/
	mkdocs build

deploy:  ## Build and publish the site to the gh-pages branch on origin
	mkdocs gh-deploy

clean:
	rm -rf site

# Copy docs/*.md out to the per-topic folder READMEs (repo-specific helper).
sync:
	bash sync.sh

# --- Slides --------------------------------------------------------------
# The reveal.js slides under src/reveal/*.md are PRE-BUILT into docs/slides/
# and committed, so the normal build/deploy above needs NO node/reveal-md.
# Only the HTML + reveal.js runtime are committed (~6 MB); the slide images
# are NOT duplicated — docs/slides/static is a symlink to reveal/static, which
# mkdocs follows when building the site. Run this only if you edit the decks.
# Requires (this target only): npm install -g reveal-md
slides:  ## Rebuild the committed slides from src/reveal/ (needs reveal-md)
	rm -rf docs/slides
	cd reveal && reveal-md . \
	    --template static/template/reveal.html \
	    --listing-template static/template/listing.html \
	    --css static/css/evo.css \
	    --css static/css/reset.css \
	    --static-dirs static \
	    --static ../docs/slides
	# Replace the duplicated static/ copy with a symlink to the source assets.
	rm -rf docs/slides/static && ln -s ../../reveal/static docs/slides/static

.PHONY: serve build deploy clean sync slides
