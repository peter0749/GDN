#!/bin/bash
cat $2 | xargs -n5 -P$3 ./$1 1> "$4.stdout" 2> "$4.stderr"
