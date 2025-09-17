import re
import difflib
from continuous_alignment import continuous_alignment
class GrammarCorrector:

    def correct(self, text):
        raise NotImplemented()


GRAMMAR_CORRECTION_PROMPTS = {
        "de": "Korrigiere den folgenden deutschen Text und gib mir die korrigierte Version. Mach keine Erklärungen, gib nur den korrigierten Text zurück. Wenn der Text bereits korrekt ist, gib den Originaltext zurück.",
        "cs": "Oprav následující text podle pravidel českého pravopisu a vypiš správnou verzi. Nedávej žádná vysvětlení, jen opravený text. Pokud je text už správně, vypiš původní text.",
}


class GPTGrammarCorrector(GrammarCorrector):

    def __init__(self, api_key_file='openai_api_key.txt', language="cs"):
        import openai
        self.openai = openai
        # Load the OpenAI API key from the file
        with open('openai_api_key.txt') as f:
            api_key = f.readline().strip()
        openai.api_key = api_key
        self.set_language(language)


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
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Extract the corrected text from the response.
        chat_response = completion.choices[0].message.content
        return chat_response

class GrammarChecker:

    def __init__(self):
        pass

    @staticmethod
    def tokenize(text):
        return re.findall(r'\w+(?:-\w+)*\.?|[.,?!]', text)
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
        # Tokenize texts for detailed comparison.
        orig_toks = self.tokenize(orig)
        corrected_toks = self.tokenize(corrected)
#        corrected_toks = corrected_toks[:50] + ["že","?"] + corrected_toks[50:] 
        print(orig_toks)
        print(corrected_toks)

        alignment = continuous_alignment(orig_toks, corrected_toks)
        edits = [self.edit_operation(a,b) for a,b in alignment]

        # highlight errors so that they can be displayed
        h_errors = []
        h_subs = []
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


        # compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / n if n > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "confusion_matrix_perc": {"tp": tp/n*100, "fp": fp/n*100, "tn": tn/n*100, "fn": fn/n*100},
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
            "detections": detections,

            # the baseline that is always guessing the most frequent class (here: "C")
            "baseline_accuracy": gold_result[k].count("C")/n if n > 0 else 0.0,


        }





g = GPTGrammarCorrector()
a = """    Americký prezident Donald Trump v pátek prohlásil že mu dochází
trpělivost s ruským protějškem Vladimirem Putinem. Trump se od lednového
návratu do černého domu snaží zprostředkovat mír na Ukrajině kam v únoru 2022
na Putinův rozkaz vpadla ruská armáda. Zatímco Ukrajina opakovaně souhlasila z
americkým návrhem na bezpodmínečné příměří moskva na to nepřistoupila.
"""
x = g.correct(a)
ch = GrammarChecker()
x = ch.check(a, x)
#print(x)


gold = """Americký prezident Donald Trump v pátek prohlásil, že mu dochází
trpělivost s ruským protějškem Vladimirem Putinem. Trump se od lednového
návratu do Bílého domu snaží zprostředkovat mír na Ukrajině, kam v únoru 2022
na Putinův rozkaz vpadla ruská armáda. Zatímco Ukrajina opakovaně souhlasila s
americkým návrhem na bezpodmínečné příměří, Moskva na to nepřistoupila.
"""

gold_result = ch.check(a, gold)
print(gold_result)

for k in ["orig",
         "orig_toks",
         "corrected",
         "corrected_toks",
#         "alignment",
        "orig_toks_edits",
         "edit_operations",
         "corrections",
         "errors",
         "highlighted_errors",
         "highlighted_corrections"]:
    if k in gold_result:
        print(k, gold_result[k])

det = ch.evaluate_check(x, gold_result)
print(det)

for t,a,b,c in zip(x["orig_toks"],x["orig_toks_edits"], gold_result["orig_toks_edits"], det["detections"]):
    print(t,a,b,c)