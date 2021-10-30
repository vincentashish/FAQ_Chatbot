  
# Chatbot implementation with PyTorch. 
- The implementation is simple using Feed Forward Neural net with 2 hidden layers. 


## Installation

### Create a Virtual environment

```console
mkdir FAQbot
$ cd FAQbot
$ python3 -m venv .
```

### Activate it
Mac / Linux:
```console
. venv/bin/activate
```
Windows:
```console
venv\Scripts\activate
```

### Install PyTorch and dependencies

For Installation of PyTorch see [official website](https://pytorch.org/).

You also need `nltk`:
 ```console
pip install nltk
 ```

If encounterd an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chat.py
```
## Customize
Modify contents of [intents.json](intents.json).
- Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. 
- You have to re-run the training whenever this file is modified.

