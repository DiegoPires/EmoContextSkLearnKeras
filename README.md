# Introduction - Emo Context
This work was proposed in the course "Techniques et applications du traitement de la langue naturelle" (IFT-7022) given on the Université Laval on Fall/2018.

EmoContext it's a competition proposed to analyse and detecte emotions on conversations. More details about the competition can be seen [here](https://www.humanizing-ai.com/emocontext.html) 

## About the project
This project proposes to test the efficiency of both SkLearn and Keras to make the prediction for the competition.

Multiple classifiers are created, but using multiple combination of features from each classifier to evaluate the better one using the training data.

Also, for both classifiers there's 4 combinations of pre-treatment of the texts: separating phrases with special tag, replacing some internet slangs with proper english word, remove punctuation, apply lemmatisation, filtering by open words.

|           | Phrase separation | Replace words | Remove punctuation | Apply lemmatization | Filter open words |
|-----------|-------------------|---------------|--------------------|---------------------|-------------------|
| *DTO 1*   | True              | False         | False              | False               | False             | 
| *DTO 2*   | False             | True          | True               | True                | True              | 
| *DTO 3*   | True              | True          | True               | False               | False             | 
| *DTO 4*   | False             | True          | True               | False               | False             | 

As soon as the best classifiers is find, the later is used to predict the emotions from the conversations of the competition.

### Sklearn
Currently classifiers been tested: `MultinomialNB`, `LogisticRegression`, `SGDClassifier`, `LinearSVC`, `SVC`.

Currently features been used for vectorizing data: stop words, min and max frequencies, tf-idf, binary, ngram

Currently extra-features implemented: Positive/Negative sentiment count, most present emojis count

### Keras
All Keras are using `Sequential` model. Layers vary between 2 to 8. 

Vectorizers been tested: `CountVectorizer`, `Tokenizer` and `Word2Vec`

## Results:

Full results can be seen on the `\results` folder of this project.

For SkLearn, here's the 10 best based on the training data

| Classifier |stop_words | min_f | max_f | use_tfid | binary  | ngram  | emoji | sentiment | dto  | Accuracy      |
|------------|-----------|-------|-------|----------|---------|--------|-------|-----------|------|---------------|
| LinearSVC  |	None	 | 1	 | 1	 | TRUE	  | TRUE	| (1, 2) | TRUE	 | TRUE	    | dto3 | *0.850464191* |	
| LinearSVC  |	None	 | 1	 | 1	 | TRUE	  | TRUE	| (1, 2) | TRUE	 | TRUE	    | dto4 | *0.849966844* |	
| LinearSVC  |	None	 | 1	 | 1	 | TRUE	  | TRUE	| (1, 2) | TRUE	 | TRUE	    | dto1 | *0.848972149* |	
| LinearSVC  |	None	 | 1	 | 1	 | TRUE	  | FALSE	| (1, 2) | TRUE	 | TRUE	    | dto3 | *0.845656499* |	
| LinearSVC  |	None	 | 1	 | 1	 | TRUE	  | FALSE	| (1, 2) | TRUE	 | TRUE	    | dto4 | *0.845324934* |	
| LinearSVC  |	None	 | 1	 | 1	 | TRUE	  | FALSE	| (1, 2) | TRUE	 | TRUE	    | dto1 | *0.844330239* |	
| SGD        |	None	 | 1 	 | 1	 | TRUE	  | TRUE	| (1, 2) | TRUE	 | TRUE	    | dto1 | *0.842175066* |	
| SGD 	     |  None	 | 1	 | 1	 | FALSE  | FALSE	| (1, 2) | TRUE	 | TRUE	    | dto3 | *0.840351459* |	
| SGD 	     |  None	 | 1	 | 1	 | FALSE  | FALSE	| (1, 2) | TRUE	 | TRUE	    | dto4 | *0.840185676* |	
| SGD 	     |  None	 | 1	 | 1	 | TRUE	  | TRUE	| (1, 2) | TRUE	 | TRUE	    | dto3 | *0.838527851* |	

For Keras:

| Classifier                        | Accuracy      |
|-----------------------------------|---------------|
| word2vec_denser_dto4	            | *0.848972149* | 
| word2vec_denser_dto1              | *0.848972149* | 
| word2vec_dto1	                    | *0.84375* | 
| word2vec_dto4	                    | *0.84043435* | 
| word2vec_dto2	                    | *0.839273873* | 
| word2vec_denser_dto2	            | *0.838113395* | 
| word2vec_dto3	                    | *0.826342838* | 
| word2vec_denser_dto3	            | *0.815732759* | 
| denser_and_tokenizer_binary_dto2	| *0.805537135* | 
| denser_and_tokenizer_binary_dto1	| *0.794429708* | 

## Setup 
This project was done using python 3.6.1 and Virtualenv. 

### Virtualenv
To setup the project, do the following on the terminal:

- Create `virtualenv ./env`
- Activate with `source ./env/bin/activate`
- Install requirements `pip install -r requirements.txt`

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