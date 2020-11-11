
all:
	mkdocs build
	cd reveal; reveal-md . \
        --template static/template/reveal.html \
        --listing-template static/template/listing.html \
        --css static/css/evo.css \
        --css static/css/reset.css \
        --static-dirs static \
        --static ../../slides

sync:
	bash sync.sh

deploy:
	git checkout master
	mkdocs gh-deploy
	cd reveal; reveal-md . \
        --template static/template/reveal.html \
        --listing-template static/template/listing.html \
        --css static/css/evo.css \
        --css static/css/reset.css \
        --static-dirs static \
        --static ../../slides
	git checkout gh-pages
	git add ../slides
	git commit -m 'reveal update'
	git push origin gh-pages
	git checkout master

clean:
	rm -rf site
	rm -rf ../slides
