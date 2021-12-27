mkdir -p data
mkdir -p data/corpus
mkdir -p data/dataset
mkdir -p data/corpus/timit
mkdir -p data/dataset/timit

pythonexec=C:/Users/Yang/anaconda3/envs/nlp/python.exe
$pythonexec $PWD/demixing/prepare/prepare_timit.py