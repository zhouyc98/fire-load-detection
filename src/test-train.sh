mkdir ../tests/
cd  ../tests/
cp ../src/*.py .
rm -r output0/
rm -r output1/

echo 'Test: python3.7 train.py -b 1'
python3.7 train.py -b 1
