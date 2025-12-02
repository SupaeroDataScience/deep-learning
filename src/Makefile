
all:
	mkdocs build

sync:
	bash sync.sh

deploy:
	git checkout gh-pages
	git fetch
	git checkout main
	git fetch
	mkdocs gh-deploy

clean:
	rm -rf site
