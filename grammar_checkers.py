import re
import sys
from continuous_alignment import continuous_alignment

class GrammarCorrectorBase:

    def correct(self, text):
        """Given input text, return the corrected text."""
        raise NotImplemented()

    def info(self):
        '''return: a string with info about the corrector, e.g. model name, language, etc.'''
        raise NotImplemented()


GRAMMAR_CORRECTION_PROMPTS = {
        "de": "Korrigiere den folgenden deutschen Text und gib mir die korrigierte Version. Mach keine Erklärungen, gib nur den korrigierten Text zurück. Wenn der Text bereits korrekt ist, gib den Originaltext zurück.",
        "cs": "Oprav následující text podle pravidel českého pravopisu a vypiš správnou verzi. Nedávej žádná vysvětlení, jen opravený text. Pokud je text už správně, vypiš původní text.",
}


class GPTGrammarCorrector(GrammarCorrectorBase):
    """ Uses OpenAI GPT models for grammar correction.
    """

    def __init__(self, language="cs", model="gpt-3.5-turbo", api_key_file='openai_api_key.txt'):
        import openai
        self.openai = openai
        # Load the OpenAI API key from the file
        with open(api_key_file) as f:
            api_key = f.readline().strip()
        openai.api_key = api_key
        self.set_language(language)
        self.model = model


    def set_language(self, language):
        self.language = language
        self.system_prompt = GRAMMAR_CORRECTION_PROMPTS[language]


    def correct(self, text):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Send the correction request to the API.
        completion = self.openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )

        # Extract the corrected text from the response.
        chat_response = completion.choices[0].message.content
        return chat_response

    def info(self):
        return f"GPTGrammarCorrector(model={self.model}, language={self.language})"

class GrammarChecker:
    """
    1) check: compares original text and corrected text, finds the differences ("C"opy, "S"ubstitution, "D"eletion, "I"nsertion).
    2) evaluate_check: compares the error detections with gold truth,
            and it evaluates precision, recall, F1, etc.
    """

    @staticmethod
    def tokenize(text):
#        # very simple tokenizer. Keeps punctuation as separate tokens.
#        return re.findall(r'\w+(?:-\w+)*\.?|[.,?!]', text)

        # even simpler tokenizer, keeps punctuation attached to the word
        return text.split()

    @staticmethod
    def edit_operation(a,b):
        if a == b:
            return "C"  # copy
        elif a is None:
            return "I"  # insertion
        elif b is None:
            return "D"  # deletion
        else:
            return "S"  # substitution

    def check(self, orig, corrected):
        """orig: raw text with possible grammar errors. A string.
        corrected: corrected text, e.g. proposed by LLM. A string. It can be shorter, longer, equal, different 
        whitespace formatting, etc.

        return: a dict with a structured result and info. See the comments below.
        """
        # Tokenize texts for detailed comparison.
        orig_toks = self.tokenize(orig)
        corrected_toks = self.tokenize(corrected)

        alignment = continuous_alignment(orig_toks, corrected_toks)
        edits = [self.edit_operation(a,b) for a,b in alignment]

        # highlight errors so that they can be displayed
        h_errors = []
        h_subs = []
        # correction operations of the original tokens. E.g. "C+I+I" means that after copying the 
        # original token, two tokens were inserted. This 
        correction_classes = []
        for e,(a,b) in zip(edits,alignment):
            if e == "C":
                h_errors.append(a)
                h_subs.append(a)
                correction_classes.append("C")
            elif e == "S":
                h_errors.append(f"<{b}>")
                h_subs.append(f"</{a}/{b}/>")
                correction_classes.append("S")
            elif e == "D":
                h_errors.append(f"-{a}-")
                h_subs.append(f"</{a}//>")
                correction_classes.append("D")
            elif e == "I":
                h_errors.append(f"+{b}+")
                h_subs.append(f"<//{b}/>")
                if not correction_classes:
                    correction_classes.append("I")
                else:
                    last_e = correction_classes[-1]
                    correction_classes[-1] = last_e+"+I"

        highlighted_errors = " ".join(h_errors)
        highlighted_corrections = " ".join(h_subs)

        return { # orig or corrected text:
                "orig": orig, 
                "corrected": corrected, 

                # tokens:
                "orig_toks": orig_toks, 
                "corrected_toks": corrected_toks,


                # orig tokens with edit operations, such as [("word", "C+I+I")...], which means 
                # two more words were inserted after the copied word "word"
                "orig_toks_edits": correction_classes,

                # info that is nice for debug and display but not for next processing
                "info": {
                    # (a,b) pairs, where a is from orig or None for Insertion, b is from corrected or None for Deletion
                    "orig_corrected_alignment": alignment,

                   # list of C, S, D, I, as long as alignment
                    "edit_operations": edits,

                   # list of triples e,a,b, where e is in C,S,D,I and (a,b) is from alignment
                    "corrections": [(e,a,b) for e,(a,b) in zip(edits,alignment)],

                   # filtered out errors (without any position information, for debug display only)
                    "errors": [(e,a,b) for e,(a,b) in zip(edits,alignment) if e != "C"],

                   # string that inserts check markers into orig. Only for debug display.
                    "highlighted_errors": highlighted_errors,
                   # inserts check+correction proposal into orig. Only for debug display.
                    "highlighted_corrections": highlighted_corrections,
                    }
                }

    def scores(self, conf_matrix):
        tp = conf_matrix["tp"]
        fp = conf_matrix["fp"]
        tn = conf_matrix["tn"]
        fn = conf_matrix["fn"]
        n = tp + fp + tn + fn
        # compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / n if n > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


        return {"precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1,
                "confusion_matrix_perc": {"tp": tp/n*100, "fp": fp/n*100, "tn": tn/n*100, "fn": fn/n*100},
                "confusion_matrix": conf_matrix}

    def evaluate_check(self, result, gold_result):
        k = "orig_toks_edits"
        # It is error detector, a detected error is positive (true/false), non-detecting error is negative (true/false).
        # tp: it was an error in gold and we found it in result
        # tn: both gold and result say it is correct
        # fn: it was not an error in gold but we found it in result
        # fp: it was not an error in gold but we detected it in result
        tp, fp, tn, fn = 0,0,0,0
        n = 0
        detections = []
        for a,b in zip(result[k], gold_result[k]):
            if a == "C":
                if b == "C":
                    tn += 1
                    t = "TN"
                else:
                    fn += 1
                    t = "FN"
            else:
                if b == "C":
                    fp += 1
                    t = "FP"
                else:
                    tp += 1
                    t = "TP"
            n += 1
            detections.append(t)



        conf_matrix = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        r = {
            "confusion_matrix": conf_matrix,

            "detections": detections,

            # the baseline that is always guessing the most frequent class (here: "C")
            "baseline_accuracy": gold_result[k].count("C")/n if n > 0 else 0.0,
        }
        r.update(self.scores(conf_matrix))
        return r

def main_interactive(args):
    corrector = GPTGrammarCorrector(language=args.language)
    checker = GrammarChecker()
    print("INFO: Interactive grammar checker and corrector. Type input text and observe automatic check and correction proposal. " + \
          "Note that it might be wrong.", file=sys.stderr)
    print("INFO: Model ")
    for text_with_errors in iter(input, 'exit'):
        text_with_errors = text_with_errors.strip()
        if not text_with_errors:
            break
        text_corrected = corrector.correct(text_with_errors)
        result = checker.check(text_with_errors, text_corrected)

        print("NUMBER OF DETECTED ERRORS:\t", len(result["info"]["errors"]))
        print("HIGHLIGHTED ERRORS:\t", result["info"]["highlighted_errors"])
        print("HIGHLIGHTED CORRECTION:\t", result["info"]["highlighted_corrections"])
        print("CORRECTED TEXT:\t", text_corrected)
        print()

def main_eval(args):
    print("INFO: Evaluation mode.")
    checker = GrammarChecker()

    conf_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    c = 0
    n = 0
    with open(args.eval) as f:
        for auto_corr, gold_line in zip(sys.stdin, f):
            auto_or, auto_corr = auto_corr.strip().split("\t")
            gold_or, gold_corr = gold_line.strip().split("\t")
            a = checker.check(auto_or, auto_corr)
            g = checker.check(gold_or, gold_corr)
            print("CHECKS:",file=sys.stderr)
            print("AUTO\t",a["info"]["highlighted_errors"],file=sys.stderr)
            print("GOLD\t",g["info"]["highlighted_errors"],file=sys.stderr)
            print("CORRECTIONS:",file=sys.stderr)
            print("ORIG\t",auto_or,file=sys.stderr)
            print("AUTO CORRECTED:\t",a["corrected"],file=sys.stderr)
            print("GOLD CORRECTED:\t",g["corrected"],file=sys.stderr)

            c += g["orig_toks_edits"].count("C")
            n += len(g["orig_toks_edits"])

            ch = checker.evaluate_check(a,g)
            print(ch,file=sys.stderr)

            print(conf_matrix,file=sys.stderr)
            conf_matrix = {k: conf_matrix[k]+ch["confusion_matrix"][k] for k in conf_matrix}

            print(file=sys.stderr)

    print("FINAL SCORES:", file=sys.stderr)
    scores = checker.scores(conf_matrix)
    scores["baseline_accuracy"] = c/n if n > 0 else 0.0
    print(scores)

def main_process(args):
    corrector = GPTGrammarCorrector(language=args.language)
    checker = GrammarChecker()
    print("INFO: Processing mode. Reads lines from stdin, outputs tsv with original and corrected text.", file=sys.stderr)
    print("INFO: Model: ", corrector.info(), file=sys.stderr)
    for text_with_errors in sys.stdin:
        text_with_errors = text_with_errors.strip()
        if not text_with_errors:
            continue
        text_corrected = corrector.correct(text_with_errors)
        result = checker.check(text_with_errors, text_corrected)
        corr = result['corrected'].replace("\n"," ").replace("\t"," ").strip()
        print(f"{result['orig']}\t{corr}", flush=True)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Grammar checker.")
    parser.add_argument("--language", type=str, default="cs", help="Language code, e.g. 'cs' or 'de'.")
    parser.add_argument("--interactive", default=False, action="store_true", 
                        help="Interactive mode: type input and observe checked and corrected output.")
    parser.add_argument("--eval", default=None,  
                        help="Evaluation mode: don't process grammar checker, only evaluate it. " + \
                            "The argument is a tsv filename with gold (two tab-separated columns, orig+gold corrected), " + \
                            "stdin is the system output to be evaluated (two tab-separated columns, orig+auto corrected).")

    args = parser.parse_args()

    if args.interactive and args.eval is not None:
        print("ERROR: --interactive and --eval can not be used at once.", file=sys.stderr)
        sys.exit(1)

    if args.interactive:
        main_interactive(args)
    elif args.eval is not None:
        main_eval(args)
    else:
        main_process(args)
