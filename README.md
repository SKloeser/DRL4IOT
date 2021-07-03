# DRL4IOT scap-2020-code-release

This is the supplemental material to the paper "DRL for IoT Interoperability" found on: 
https://link.springer.com/chapter/10.1007/978-3-662-62962-8_23

## Setup
Follow these steps to setup the repository:
1. Create a new virtual env using python 3.6
2. Install requirements:\
`pip install -r requirements.txt`
3. Install NLTK data:\
`python -c "import nltk; nltk.download('punkt')"`
4. Download GloVe pre-trained word vectors and extract in weights directory:\
Linux:\
`wget -c http://nlp.stanford.edu/data/wordvecs/glove.6B.zip -P weights` \
`unzip weights/glove.6B.zip -d  weights`


## Model training
Step by Step training example can be found in Training.ipynb notebook.
