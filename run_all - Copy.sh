#!/bin/bash

INDEX=80
INTERPRETER=python3
declare -a NAMES=(  "breast" "flare" "glass" "heart" "iris" 
					"led7" "anneal" "pageBlocks" "pima" "wine" 
					"zoo" "hepati" "horse" "adult" "mushroom" 
					"penDigits" "letRecog" "soybean" "ionosphere" "cylBands")

for NAME in ${NAMES[@]};
do
	echo $NAME
	INTERPRETER sigdirect_test.py $NAME  >> output_bfs_$INDEX;
	echo "done"
done
