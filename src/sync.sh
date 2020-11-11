#!/bin/bash

for i in docs/*md; do
    echo $i
    if [[ "$i" == "docs/index.md" ]]; then
        cp $i ../README.md
    else
        d=$(echo $i | cut -d '/' -f 2 | cut -d '.' -f 1)
        cp $i ../$d/README.md
    fi
done
