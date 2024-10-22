from eval import GoogleVertexAI
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase

model = GoogleVertexAI("gemini-1.5-pro-002", "GENAI_API_KEY")

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Since you are a chatbot which mimics a medical professional, you should also penalize any incorrect medical advice",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model = model
)
test_case = LLMTestCase(
    input="come si misura la pressione?",
    actual_output="La pressione sanguigna è misurata con un monitor o un sterolettometro, che ti permette di controllare il tuo pulsante arteriale. La pressione sanguigna è composta da due componenti: l’arterioscala e la venoscala, le quali sono messe a confronto rispetto alla presenza del paziente sulla soglia del cuore.",
    expected_output="""
        La pressione si misura con un misuratore di pressione al braccio, possibilmente validato, che puoi acquistare in farmacia o su internet. Durante tutta la misurazione della pressione e fino al completo sgonfiaggio del bracciale, è bene tenere presente le seguenti semplici raccomandazioni:
        Il paziente dovrebbe essere rilassato, seduto comodamente, con le gambe non incrociate, in ambiente tranquillo, con temperatura confortevole da almeno cinque minuti.
        Non si dovrebbe assumere bevande contenenti caffeina nell’ora precedente, né aver fumato da almeno un quarto d’ora (anzi, non si dovrebbe fumare mai!).
        Il paziente non dovrebbe parlare durante la registrazione.
        Il braccio deve essere appoggiato ed il bracciale deve essere all’altezza del cuore (vedi figura). Non importa quale braccio venga usato per la misurazione, ma bisogna ricordare che esistono a volte differenze sensibili nei valori misurati nelle due braccia. In tali casi, si dovrà utilizzare per la misura il braccio con la pressione più elevata. 
        Le dimensioni del bracciale di gomma devono essere adattate alla dimensione del braccio del paziente. Nel caso di bambini o di adulti molto magri, è necessario utilizzare bracciali di dimensioni minori di quelle standard, mentre nel caso di persone molto robuste o di pazienti obesi, il bracciale dovrebbe avere una lunghezza e larghezza superiore, possibilmente di forma tronco-conica. 
"""
)

correctness_metric.measure(test_case)
print(correctness_metric.score)
print(correctness_metric.reason)