# grammar correctors

# the abstract base class. The correctors must implement this interface -- the correct and info methods.
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

        # TODO: these are suggested by Copilot. They need correction and verification.
        "en": "Correct the following English text and provide me with the corrected version. Do not give any explanations, just return the corrected text. If the text is already correct, return the original text.",
        "fr": "Corrige le texte français suivant et donne-moi la version corrigée. Ne fais pas d'explications, donne-moi seulement le texte corrigé. Si le texte est déjà correct, donne-moi le texte original.",
        "hu": "Javítsd ki a következő magyar szöveget, és add meg a javított változatot. Ne adj magyarázatokat, csak a javított szöveget. Ha a szöveg már helyes, add meg az eredeti szöveget.",
        "bg": "Коригирай следния български текст и ми предостави коригираната версия. Не давай обяснения, просто върни коригирания текст. Ако текстът вече е правилен, върни оригиналния текст.",
        "el": "Διόρθωσε το παρακάτω ελληνικό κείμενο και δώσε μου την διορθωμένη έκδοση. Μην δίνεις εξηγήσεις, απλώς επιστρέψε το διορθωμένο κείμενο. Αν το κείμενο είναι ήδη σωστό, επιστρέψε το αρχικό κείμενο.",
        "lu": "Corrigéiert den folgenden lëtzebuergeschen Text a gitt mir déi korrigéiert Versioun. Maacht keng Erklärungen, gitt just den korrigéierten Text zréck. Wann de Text schonn korrekt",
        # maltese:
        "mt": "Ikkoreġi t-test Malti li ġej u agħtini l-verżjoni kkorreġuta. Tgħaddix spjegazzjonijiet, għid biss it-test kkorreġut. Jekk it-test diġà huwa korrett, agħtini t-test oriġinali.",
}
LANGUAGE_OPTIONS = sorted(list(GRAMMAR_CORRECTION_PROMPTS.keys()))

class OpenAIGPTGrammarCorrector(GrammarCorrectorBase):
    """ Uses OpenAI GPT models for grammar correction.
    """

    def __init__(self, language="cs", model="gpt-3.5-turbo", api_key_file='openai_api_key.txt'):
        # this import is here to avoid requiring openai package if this class is not used.
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



# TODO: the code duplication is not nice: 
class UfalGemmaGrammarCorrector(OpenAIGPTGrammarCorrector):
    """ Uses Ufal Gemma API for grammar correction.
    """

    def __init__(self, language="cs", model="DGT-API.google/gemma-3-27b-it", api_key_file='aiufal_api_key.txt'):
        # this import is here to avoid requiring openai package if this class is not used.
        import openai
        # Load the OpenAI API key from the file
        with open(api_key_file) as f:
            api_key = f.readline().strip()
        self.set_language(language)
        self.model = model
        print(model)

        self.client = openai.Client(
            base_url="https://ai.ufal.mff.cuni.cz/api/v1",
            api_key=api_key,
        )

    def correct(self, text):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Send the correction request to the API.
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        # Extract the corrected text from the response.
        chat_response = completion.choices[0].message.content
        return chat_response
        

    def info(self):
        return f"UfalGemmaGrammarCorrector(language={self.language})"

def corrector_factory(api_model, **kw):
    """
    api_model: a string of the form "api/model", e.g. "openai/gpt-3.5-turbo"
    kw: other keyword arguments passed to the corrector constructor, e.g. language=cs.

    return: a grammar corrector based on the API and model specified by a `api_model` parameter.
    """
    api, model = api_model.split("/")
    if api == "openai":
        return OpenAIGPTGrammarCorrector(model=model, **kw)
    # TODO: add other APIs or models here.
    elif api == "DGT-API.google":
        return UfalGemmaGrammarCorrector(model=api_model, **kw)
    else:
        raise ValueError(f"Unknown API: {api}")