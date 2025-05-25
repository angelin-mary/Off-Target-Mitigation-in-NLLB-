apt update
apt install -y python3.8 python3.8-dev python3-pip
###########################
python3.8 -m pip install pip --upgrade --no-cache-dir
python3.8 -m pip install cython --upgrade --no-cache-dir
python3.8 -m pip install multidict datasets torch --no-cache-dir
python3.8 -m pip install pyarrow sacrebleu nltk sacremoses --no-cache-dir
python3.8 -m pip install transformers --no-cache-dir
python3.8 -m pip uninstall tokenizers
python3.8 -m pip install tokenizers --no-cache-dir
#python3.8 -m pip show transformers
python3.8 -m pip show tokenizers
python3.8 -m pip install seaborn pandas numpy tkinter  --no-cache-dir
python3.8 -m pip install matplotlib seaborn  --no-cache-dir
python3.8 -m pip install evaluate  --no-cache-dir
python3.8 -m pip install langid tkinter --no-cache-dir
python3.8 -m pip install torch safetensors --no-cache-dir
python3.8 -m pip install scikit-learn --no-cache-dir
python3.8 -m pip install transformers[torch] --no-cache-dir

python3.8 -m pip install sentencepiece --no-cache-dir
python3.8 -m pip install protobuf --no-cache-dir
python3.8 -m pip install lang2vec deepspeed --no-cache-dir


###########################

python3.8 /path/to/your/script.py --path_of_code_to_be_executed 