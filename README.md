# Introduction - Emo Context
This work was proposed in the course "Techniques et applications du traitement de la langue naturelle" (IFT-7022) given on the Université Laval on Fall/2018.

EmoContext it's a competition proposed to analyse and detecte emotions on conversations. Move details about it can be seen [here](https://www.humanizing-ai.com/emocontext.html) 

## About the project
This project proposes to test the efficiency of both SkLearn and Keras to make the prediction for the competition.

Multiple classifiers are created, but with multiple combination of features to evaluate the better one using the training data.

As soon as the best classifiers is find, the later is used to predict the emotions from the conversations of the competition.

### Sklearn
Currently classifiers been tested: `MultinomialNB`, `LogisticRegression`, `SGDClassifier`, `LinearSVC`, `SVC`.

Currently features been used for vectorizing data: stop words, min and max frequencies, tf-idf, binary, ngram

CUrrently extra-features implemented: Positive/Negative sentiment count, most present emojis count

### Keras
All Keras are using `Sequential` model. Layers vary between 2 to 8. 

Vectorizers been tested: `CountVectorizer`, `Tokenizer` and `Word2Vec`

## Setup 
This project was done using python 3.6.1 and Virtualenv. 

### Virtualenv
To setup the project, do the following on the terminal:

- Create `virtualenv ./env`
- Activate with `source ./env/bin/activate`
- Install requirements `$ pip install -r requirements.txt`

### VSCode

- Install the `Python` extension
- `Shift + cmd + p` and search `python: select interpreter`. Chose the `env` environment to make it run with VSCode

### NLTK data

Open a python prompt and install all NLTK data:

```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')
```

## Executing the code

On VSCode just click right on the main.py and chose `Run python file on terminal`.

Or in the command prompt run `python3 <path_to_the_project/main.py` according to where the code was downloaded

## More useful commands:

- Exporte any changes done on the dependances of the project with `pip freeze > requirements.txt`
- Desactivate the virtualenv with `deactivate`