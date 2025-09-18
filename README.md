# LLMs as a grammar checker

This project is a demo, evaluation and experimentation framework. It applies generative LLMs, via API, prompted to propose monolingual text grammar correction and grammar check.

Example: 

For German input text `Kleine katze kom her bite.`, *grammar correction* is `Kleine Katze komm her bitte.`. *Grammar check* is highlighting the words that should be corrected, e.g. by wrapping them with `<>`: `Kleine <Katze> <komm> her <bitte.>`

## Installation and prerequisites

- you need Python3 on your computer

- you need a python3 virtual environments with dependencies for the LLM model API that you're going to use. At this moment, there is:

-  **OpenAI API**: Install with `pip install openai==0.28`. You need an OpenAI API key saved in a file named `openai_api_key.txt` in the directory from which you run `python3 grammar_checker.py`.

## Usage

This project is a shas the following modes:

- 1) **Interactive**: select model and language direction, type input, get output, observe grammar corrections and checks. 

```
llm-grammar-checker$ python3 grammar_checker.py --interactive --language de
INFO: Interactive grammar checker and corrector. Type input text and observe automatic check and correction proposal. Note that it might be wrong.
INFO: Model:  GPTGrammarCorrector(model=gpt-3.5-turbo, language=de)
Kleine katze kom her bite.
NUMBER OF DETECTED ERRORS:	 4
HIGHLIGHTED ERRORS:	 Kleine <Katze> <komm> <bitte> <her.>
HIGHLIGHTED CORRECTION:	 Kleine </katze/Katze/> </kom/komm/> </her/bitte/> </bite./her./>
CORRECTED TEXT:	 Kleine Katze komm bitte her.
```

- 2) **Processing**: Assume you have a text to be corrected in a file, one segment per line. By segment we mean a part of text that will be processed by LLM independently, e.g. sentence, few words, or a paragraph. You want the grammar corrections in another file.

```
llm-grammar-checker$ python3 grammar_checker.py < demo-data/demo-cs.errors.txt > demo-data/demo-cs.errors-corrected_gpt3.5.tsv
INFO: Processing mode. Reads lines from stdin, outputs tsv with original and corrected text.
INFO: Model:  GPTGrammarCorrector(model=gpt-3.5-turbo, language=cs)
```

The input file `demo-cs.errors.txt` looks like this -- one segment per line, no tabs or newlines inside of the segments, raw text file:

```
Vážená paní předsedkyně metsolová,
vážené poslankyně a váženi poslanci,
evropa se nachází uprostřed boje.
```

The output `demo-data/demo-cs.errors-corrected_gpt3.5.tsv` is `tsv` = tab separated values. In the first column, there is the original text, in the second column is the LLM-corrected text.

```
Vážená paní předsedkyně metsolová,	Vážená paní předsedkyně Metsolová,
vážené poslankyně a váženi poslanci,	Vážené poslankyně a vážení poslanci,
evropa se nachází uprostřed boje.	Evropa se nachází uprostřed boje.
```

- 3) **Evaluation**: Assume you have original text with errors, gold grammar corrections, and automatic corrections. The evaluation mode 

- first, converts the corrections to checks, by aligning the original text to the corrected one. 

- second, it compares the gold checks to the automatic ones, finds the true and false positives (error detected) and negatives (no error detected), and counts the scores: precision, recall, F1, accuracy, baseline accuracy. First, it prints the scores for each segment (to stderr):

```
$ python3 grammar_checker.py --eval demo-data/demo-cs.errors-correct_gold.tsv < demo-data/demo-cs.errors-corrected_gpt3.5.tsv
INFO: Evaluation mode.
CHECKS:
AUTO	 Vážená paní předsedkyně <Metsolová,>
GOLD	 Vážená paní předsedkyně <Metsolová,>
CORRECTIONS:
ORIG	 Vážená paní předsedkyně metsolová,
AUTO CORRECTED:	 Vážená paní předsedkyně Metsolová,
GOLD CORRECTED:	 Vážená paní předsedkyně Metsolová,
{'confusion_matrix': {'tp': 1, 'fp': 0, 'tn': 3, 'fn': 0}, 'detections': ['TN', 'TN', 'TN', 'TP'], 'baseline_accuracy': 0.75, 'precision': 1.0, 'recall': 1.0, 'accuracy': 1.0, 'f1': 1.0, 'confusion_matrix_perc': {'tp': 25.0, 'fp': 0.0, 'tn': 75.0, 'fn': 0.0}}
{'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
```

Then it aggregates the scores for the whole file:

```
FINAL SCORES:
{'precision': 0.8275862068965517, 'recall': 0.8, 'accuracy': 0.9051724137931034, 'f1': 0.8135593220338982, 'confusion_matrix_perc': {'tp': 20.689655172413794, 'fp': 4.310344827586207, 'tn': 69.82758620689656, 'fn': 5.172413793103448}, 'confusion_matrix': {'tp': 24, 'fp': 5, 'tn': 81, 'fn': 6}, 'baseline_accuracy': 0.7413793103448276}
```


## How to interpret the quality scores:

- read https://en.wikipedia.org/wiki/Precision_and_recall , https://en.wikipedia.org/wiki/Confusion_matrix

- tp, fp, tn, fn are true or false positives or negatives

- `confusion_matrix` is with total counts

- `confusion_matrix_perc` is confusion matrix with percentage, the range is 0-100%

- precision, recall, F1 and accuracy are in 0.0-1.0 range, the higher the better

- baseline accuracy is the accuracy of a model that always detecs non-error. If accuracy is higher than baseline, it's good. If not, there might be some bug in the code.

## Proposed workflow for the tutorial:

The goal of this project is to investigate the ability of state-of-the-art multilingual LLMs to propose grammar corrections. For that, let's integrate the most promising LLMs via API into this code (Task 1), and then create test sets and evaluate the models (Task 2). Task 3, non-LLM baselines, is very optional, low priority. 

### Task 1: LLM Models and APIs

At this moment, there is OpenAI API integrated with gpt-3.5-turbo model. 

- To add another OpenAI API model: simply try another model name. Search in OpenAI API documentation for other options.

- To add another API with other models: in `models.py`, add a new class inherited from `GrammarCorrectorBase`, analogical to `OpenAIGPTGrammarCorrector`.

### Task 2: Languages and data

At this moment, the code is tested on German (`de`) and Czech (`cs`).

- To add a new language: in `models.py` in `GRAMMAR_CORRECTION_PROMPTS`, add a new two-letter language code as key and prompt in that language. Some initial versions are there, they need to be corrected.

**Evaluation data:** 

- in `demo-data/` dir, there is an evaluation set example. It is a short sample from https://czechia.representation.ec.europa.eu/projev-predsedkyne-komise-von-der-leyenove-o-stavu-unie-v-roce-2025-2025-09-10_cs . Dominik introduced some grammar and typo errors there manually. It is just a demo sample.

- Create your own test sample for grammar checker. Start with any clean text, introduce the errors manually. E.g. remove commas, remove words, change articles, etc.

- **Be aware that the project uses 3rd party APIs!** Rather use already published texts. Unless we change the APIs to private ones.

- Or try to search for ready to use evaluation dataset for grammar checkers on the Internet.

### Task 3 (optional, low priority): State of the art non-LLM grammar checkers

You can compare the LLM grammar checkers to any other state-of-the-art, e.g. language specific non-LLM grammar checkers. For example https://lindat.mff.cuni.cz/services/korektor/ for Czech, or another ones that you know or find on the Internet.

If your grammar checker is not proposing corrections, you need to adapt the `GrammarChecker` class. Create a new one that inherits it, and change `check` method. It needs to receive the scoring input and transform it into the same output dictionary.



## Credits:

This project is a very extended version of https://github.com/gamzez/python-proofreader .

Author: Dominik Macháček, machacek@ufal.mff.cuni.cz