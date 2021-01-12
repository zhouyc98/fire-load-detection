#!/bin/sh

echo 'Init & clear tests/'
mkdir ../tests/ 2> /dev/null
cp -f ./*.py ../tests/
rm -r ../tests/output/ 2> /dev/null
rm -r ../tests/output1/ 2> /dev/null

cd ../tests
# python3.7 train.py -n R50 -b 1 -it 1k
