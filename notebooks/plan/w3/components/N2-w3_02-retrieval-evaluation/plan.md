# N2 w3_02 Retrieval evaluation (rework)

## 1. Header

- Level: component
- ID: N2
- Meta: ../../plan.md
- Mode: refine. The material exists (notebooks/w3_02_retrieval_evaluation.py, cache under notebooks/data/cache); this plan lists only the rows that change. Unchanged cells keep their current text.
- Language: de and en in one file (t('de', 'en')); final texts below are given in both languages where the row is new.
- Status log:
  - 2026-10-21 created from the material and Hanno's review notes (no earlier plan.md existed in the repository; Jill's "meta plan" referenced in code comments is not in the repo)
  - 2026-10-21 audience and presenter roles skipped on request
  - 2026-10-21 revision 2 confirmed by Hanno; deviations from the rows below: 4.3 (E a, line 72) not applied; X2 translation stage dropped, judge reasons stay English with one hint; F3 and G4 say "small to medium effects in opposite directions, cost decides" instead of "cannot decide"; F4 ends with the held-out set sentence; D2 dictionary card also allows two claims; F7 run band is GitHub issue #33 (aihpi/workshop-rag); the live button stays
  - 2026-10-21 Hanno: examples replaced by three single sentences of at most 150 characters (gold 34, 28, 30 as A, B, C); definitions split into everyday whole, everyday granular, logic (without the purpose link) and working definition with four rules (actions split, reasons and purposes split, a list the Kompendium states as one list stays one claim, repetitions do not count); counts A 1/2/5/5, B 1/2/4/3, C 1/2/9/3; slides rebuilt per hpi-corporate-design (no shadow, sentence case)
  - 2026-10-21 creator (main agent) complete: w3_02_retrieval_evaluation.py, ragkit/w3_eval.py (paired_diff), ragkit/corpus_text.py; marimo check, export de and en, gated cells exercised, pytest 80 passed, ruff clean

## 2. Interface (read-only copy)

| Field | Value |
|---|---|
| ID, type | N2, marimo notebook |
| Minutes | 60 normal (about 35 short), unchanged |
| Goals served | G2 (compute Context Precision and Context Recall by hand), G3 (judge the reliability of a judge score), G4 (new: decide between two configurations with a significance and effect size reading) |
| Entry state | Participant has run w3_01: knows chunk, retrieval, TOP_K, judge, reference answer, gold data set, the KISZ website corpus |
| Exit state | Participant can state in one sentence why a RAGAS number alone does not decide between two configurations, and what else decides (cost, speed, spot checks) |
| Assets | data/cache/w3_02 (kisz demo, retrieval top 10, translations, diagnosis), data/cache/43_komplex__* (four configurations, TOP_K 1/3/5/10), tools/precompute_w3_02.py, ragkit/w3_eval.py |
| File | notebooks/w3_02_retrieval_evaluation.py |
| Version | 2 (version 1 is the merged state of 2026-10-20) |

## 3. Key message after the rework

Quantitative RAG evaluation (RAGAS) gives a rough estimate of how well retrieval works. In the detail it has pitfalls: a claim is not well defined and the split differs from run to run; different judges give different numbers; a small difference between two numbers is usually not significant. When the effect is small, speed and cost decide, not the score.

The current notebook already says most of this (sections E and H) but the final sections F and G then act as if the numbers were precise ("sweet spot", "precision is highest at 3"). The rework makes the pitfalls the spine of the second half.

## 4. Changes, in notebook order

The row number is the current cell order as a participant sees it. "Old" quotes the current English text; "New" is the final text, German first where a new text is needed in both languages.

### 4.1 Cross-cutting (also in the meta plan)

| # | Where | Change |
|---|---|---|
| X1 | every `bi`, `bi_cell`, `_head_cell` call | Notebook language only. In the English version show the English translation as the text, not the German original with the translation under it. In the German version show German. The original goes into one accordion "Original (Deutsch)" directly under the passage, only where the participant needs the wording (extra task part b: the word-match argument "erhöhter Schutzbedarf" vs "(H)" only works in German; keep the original visible there with the translation beside it and say so in one sentence). |
| X2 | judge reasons (sections B, D, E a and b, extra task b) | The RAGAS judge writes English reasons. German version: show the German translation, drop the hint "Die Begründungen sind englisch". Asset need: `translations.json` has 362 en entries but only 31 de entries; `tools/precompute_w3_02.py` gets a translation stage for all cached reasons and claim statements into German (same model as for en). Until the cache has them, the German version falls back to English with the existing hint. |
| X3 | concepts | Terms used before they are introduced in this notebook, with the fix: see meta plan section 6 (the audit list). For w3_02 specifically: Qdrant (quiz option G, FAQ) is never introduced; the quiz option becomes "weil die Suche nicht mehr als 5 liefert" / "because the search returns no more than 5" and the FAQ names Qdrant with one explaining clause ("Qdrant, the vector database in which the chunks are stored"). TOP_K first appears in the task card of section A without a definition; add one grey hint in section F: "TOP_K ist die Zahl der Chunks, die die Suche zurückgibt." / "TOP_K is the number of chunks the search returns." |

### 4.2 Section D: what a claim is (new rows before the claim table)

The current hint "Each claim is a single statement that can be checked on its own; RAGAS uses the same word" is the only definition. New rows D1 to D3 replace it. The Mentimeter round (meta plan, Mentimeter block) runs in the room between D1 and D2.

| # | Cell | Purpose | Final text | Checkpoint |
|---|---|---|---|---|
| D1 | md, after the chain figure | surprise: there is no agreed count | **de** Bevor der Richter zählt, zählen Sie. Wie viele Aussagen stecken in diesem Satz? "Administrationskonten sollten durch Mehr-Faktor-Authentisierung geschützt werden, da sie ein besonders hohes Schadenspotenzial besitzen und bevorzugtes Angriffsziel darstellen." Ihre Antwort geben Sie im Raum über Mentimeter ab. **en** Before the judge counts, you count. How many claims are in this sentence? "Admin accounts should be protected by multi-factor authentication, because they carry a particularly high damage potential and are a preferred target of attacks." Give your answer in the room via Mentimeter. | the Mentimeter histogram spreads from 1 to 4 |
| D2 | md, three definitions as cards | conceptualise: competing definitions | Heading: **de** Was ist eine Aussage? Drei Antworten **en** What is a claim? Three answers. Card 1 **de** Alltag (Wörterbuch): "Eine Aussage ist ein Satz, der etwas feststellt." Danach ist der Beispielsatz eine Aussage. **en** Everyday (dictionary): "A claim is a sentence that states something." By that the example is one claim. Card 2 **de** Logik: Eine Aussage ist etwas, das für sich allein wahr oder falsch sein kann. "A, weil B und C" sind drei Aussagen: A, B, C und außerdem die Behauptung, dass B und C der Grund für A sind. Danach hat der Beispielsatz drei bis vier Aussagen. **en** Logic: a claim is something that can be true or false on its own. "A, because B and C" are three claims: A, B, C, plus the claim that B and C are the reason for A. By that the example has three to four claims. Card 3 **de** NLP-Forschung (FActScore): Eine Aussage ist die kleinste Einheit, die eine Information trägt und die man gegen eine Quelle prüfen kann. Aufzählungen zählen je Element. Danach hat der Beispielsatz drei Aussagen. **en** NLP research (FActScore): a claim is the smallest unit that carries one piece of information and can be checked against a source. Each element of a list counts. By that the example has three claims. | none |
| D3 | md, the working definition as a key sentence, then the RAGAS instruction | one new thought: RAGAS does not define it | **de** Für dieses Notebook gilt: Eine Aussage ist der kleinste Teil der Referenzantwort, der sich für sich allein gegen das Kompendium prüfen lässt. Jedes Element einer Aufzählung, jede Begründung (weil, da) und jeder Zweck (um zu) ist eine eigene Aussage; Wiederholungen und Füllwörter zählen nicht. Mit dieser Definition stimmen Sie jetzt noch einmal ab. **en** For this notebook: a claim is the smallest part of the reference answer that can be checked against the Kompendium on its own. Every element of a list, every reason (because) and every purpose (in order to) is its own claim; repetitions and filler do not count. Vote again with this definition. Then the grey hint: **de** RAGAS gibt dem Richter keine Definition. Die Anweisung lautet nur: "analyze each statement in the answer". Der Richter gpt-oss-120b zerlegt meist je Satz, llama-3-3-70b feiner. **en** RAGAS gives the judge no definition. The instruction only says: "analyze each statement in the answer". The judge gpt-oss-120b mostly splits per sentence, llama-3-3-70b finer. | second Mentimeter round narrower than the first |

The claim table of the demo question and the formula stay. The quiz "4 claims, 3 supported" stays.

### 4.3 Section E a: line 72 (resolution row)

| # | Old | New |
|---|---|---|
| E-a2 | "Rather not. Line 72 belongs to the infrastructure offer, not to the pilot projects. The search did not find the matching line 35." | **de** Kommt auf den Kontext an. Zeile 72 steht auf der Webseite beim Angebot Infrastruktur, nicht bei den Pilotprojekten. Sie sagt aber nicht, dass die Infrastruktur nur außerhalb von Pilotprojekten genutzt werden kann. Wer die Zeile allein liest, wie der Richter, kann die Aussage als belegt lesen; wer die Webseite kennt, eher nicht. Die passende Zeile 35 hat die Suche nicht gefunden. **en** Depends on the context. Line 72 stands on the website under the infrastructure offer, not under the pilot projects. But it does not say that the infrastructure is only available outside pilot projects. Whoever reads the line alone, as the judge does, can read the claim as supported; whoever knows the website, rather not. The search did not find the matching line 35. |
| E-a3 key takeaway | "A judge can be too generous. That is why spot checks against the source ..." | **de** Ob ein Chunk eine Aussage belegt, hängt vom Kontext ab, den der Leser hat. Der Richter hat nur den Chunk. Stichproben gegen die Quelle gehören deshalb zu jeder Evaluation. **en** Whether a chunk supports a claim depends on the context the reader has. The judge has only the chunk. That is why spot checks against the source belong in every evaluation. |

The radio in E a gets the "partly" answer accepted for claim 2 as well (right set: claim 1 "part", claim 2 "part" or "out").

### 4.4 Section E c: prediction per metric

| # | Old | New |
|---|---|---|
| E-c1 | one radio "What do you expect?" (same, slightly different, completely different) | two radios, one per metric, same three options each: **de** "Was erwarten Sie für Context Precision?" and "Was erwarten Sie für Context Recall?" **en** "What do you expect for Context Precision?" and "... for Context Recall?". Feedback per metric, using the existing `_span` text. The diagram appears when both are chosen. |
| E-c2 hint | unchanged | add one sentence: **de** Bei dieser Frage schwankt nur Precision, weil der Richter ein Chunk-Urteil ändert; die Zerlegung bleibt gleich. Bei anderen Fragen schwankt die Zerlegung, und damit Recall. **en** For this question only precision varies, because the judge changes one chunk verdict; the split stays. For other questions the split varies, and with it recall. (Only if the cache still shows runs 1 to 5 as 0.75/0.75/0.75/0.75/1.0 and recall 0.5 throughout; the existing `_why` branch covers this.) |

"Ask the judge again" is already implemented live: `ev.judge_once` sends the two RAGAS prompts through `ragas.llm_factory` over an `AsyncOpenAI` client to `https://api.aisc.hpi.de/` with the key from `notebooks/.env` (OPENAI_API_KEY), model gpt-oss-120b, one call per chunk plus one call for the claims; limit three presses per participant; warning box when the key is missing. No change needed except the button label in the short version (hidden there, fine).

### 4.5 Section F: the TOP_K curve

| # | Old | New |
|---|---|---|
| F1 prediction options | "both rise / recall rises, precision falls at some point / both fall" | name the metrics: **de** "Precision und Recall steigen" / "Recall steigt, Precision fällt ab einem Punkt" / "Precision und Recall fallen" **en** "precision and recall rise" / "recall rises, precision falls from some point on" / "precision and recall fall" |
| F2 figure 1 | single line per metric, grey "sweet spot" band 3 to 5 | drop the band and the label. Add the 95 % interval of the mean as a light band per metric (bootstrap over the 43 questions, see F4). Axis limits fixed (x 0 to 13.5, y 0 to 1) in every state. |
| F3 frame "Sweet Spot" | "Up to TOP_K 3 to 5 every chunk brings clearly more recall; after that recall hardly rises, precision falls, and every chunk costs space in the prompt." | replaced by the significance block F4 and a new frame: **de** Zwischen 3 und 5 kann der Datensatz nicht entscheiden: Die Unterschiede liegen innerhalb der Streuung über die 43 Fragen. Zwischen 1 und 3 und zwischen 5 und 10 kann er es. **en** Between 3 and 5 the data set cannot decide: the differences lie within the spread over the 43 questions. Between 1 and 3 and between 5 and 10 it can. |
| F4 new: "Ist der Unterschied echt?" / "Is the difference real?" | none | New subsection after figure 1 with one table and two sentences of method. Table columns: TOP_K pair, metric, mean difference, 95 % interval (paired bootstrap over the 43 questions, 5000 resamples), questions better / worse / same, effect size (mean difference divided by the standard deviation of the paired differences, Cohen's d_z). Values computed live from the cache (`ev.paired_diff`, new helper, numpy only). Method text: **de** Jede Frage wird bei beiden TOP_K-Werten bewertet, also vergleichen wir je Frage die Differenz. Liegt die Null im 95-%-Intervall des Mittelwerts der Differenzen, kann die Stichprobe von 43 Fragen keinen Unterschied belegen. Die Effektstärke sagt, wie groß der Unterschied gemessen an der Streuung ist: unter 0,2 klein, um 0,5 mittel, ab 0,8 groß. **en** Each question is scored at both TOP_K values, so we compare the difference per question. If zero lies within the 95 % interval of the mean difference, the sample of 43 questions cannot show a difference. The effect size says how large the difference is relative to the spread: below 0.2 small, around 0.5 medium, from 0.8 large. Key takeaway frame: **de** Ein Wert ist nur dann besser als ein anderer, wenn der Unterschied signifikant und groß genug ist. Sonst entscheiden Kosten, Geschwindigkeit und Promptlänge. **en** One value is only better than another when the difference is significant and large enough. Otherwise cost, speed and prompt length decide. |
| F5 cards "Why the curves run this way" | card 1 subtitle "0.047 of the rise of 0.056" | subtitle becomes a count: **de** "{n} der 43 Fragen tragen fast den ganzen Anstieg" **en** "{n} of the 43 questions carry almost the whole rise" (n = len(_flips)). Card 2 ("Precision from 3 on") stays. New card 3 before "Recall": **de** Chunkgröße. Wie viele Chunks nützlich sein können, hängt von der Chunkgröße ab: Große Chunks tragen mehr der Referenzantwort, dann ist Recall schon bei kleinem TOP_K hoch und fällt Precision früher. Die Kurve gilt für diese Chunks (json_structured_sections). **en** Chunk size. How many chunks can be useful depends on chunk size: large chunks carry more of the reference answer, so recall is high at small TOP_K already and precision falls earlier. The curve holds for these chunks (json_structured_sections). |
| F6 accordion "Why can precision rise from TOP_K 1 to 3 at all?" | two sentences | extend with the mechanics: **de** Context Precision einer Frage kann beim Verlängern der Liste nur steigen, wenn bis dahin kein nützlicher Chunk dabei war (Wert 0). Ist schon einer dabei, senkt jeder weitere nützliche Chunk weiter unten den Durchschnitt oder lässt ihn gleich. Von 1 auf 3 haben noch viele Fragen den Wert 0, von 10 auf 13 fast keine mehr; dort kann Precision also kaum noch steigen. Dazu kommt die Streuung des Richters: Er hat für jeden TOP_K-Wert neu geurteilt. **en** Context Precision of one question can only rise when the list grows if no useful chunk was in it so far (value 0). Once one is in, every further useful chunk lower down lowers the average or leaves it. From 1 to 3 many questions still sit at 0, from 10 to 13 almost none; so precision can hardly rise there. On top comes the judge's variance: it judged anew for every TOP_K value. |
| F7 figure 1 run band (asset need) | one judge run per TOP_K | Optional, if the user agrees: `tools/precompute_w3_02.py` gets a stage that judges the standard configuration three times per TOP_K; figure 1 then shows min to max over runs as a thin grey band and the hint says "three judge runs". Cost about 3 × 43 × (1 + 3 + 5 + 10 + 4) calls on the AISC gateway. Without it, the notebook says in one hint that figure 1 is one run and the spread of section E c applies to every point. |

### 4.6 Section G: your TOP_K

| # | Old | New |
|---|---|---|
| G1 key sentence | "TOP_K is a trade-off between precision and recall." | **de** TOP_K ist eine Abwägung: Wo die Zahlen keinen Unterschied belegen, entscheiden Kosten. **en** TOP_K is a trade-off: where the numbers show no difference, cost decides. |
| G2 quiz option c | "because Qdrant returns no more than 5" | "because the search returns no more than 5" (X3) |
| G3 task table | means only | add the columns "Δ to 3 (95 % interval)" for both metrics from F4, so the participant chooses with the uncertainty in view |
| G4 feedback after "Confirm choice" | "Many choose 3 or 5. At 3 precision is highest (...), at 5 recall is higher (...). There is no single right answer; it depends on whether missing information or long prompts cost more in your case." | **de** 1 ist messbar schlechter im Recall, 10 messbar schlechter in der Precision. Zwischen 3 und 5 belegen die 43 Fragen keinen Unterschied. Wer 3 wählt, spart Prompt-Länge; wer 5 wählt, kauft etwas Sicherheit beim Recall, die der Datensatz nicht nachweisen kann. Beides ist vertretbar, 1 und 10 nicht. **en** 1 is measurably worse in recall, 10 measurably worse in precision. Between 3 and 5 the 43 questions show no difference. Choosing 3 saves prompt length; choosing 5 buys some recall safety the data set cannot prove. Both are defensible, 1 and 10 are not. (Remove "There is no right or wrong here".) |

### 4.7 Extra task

| # | Old | New |
|---|---|---|
| Z1 figure (part a) | axes relayout when the configuration changes (annotation length changes tight_layout) | fixed axes: legend inside the axes instead of end-of-line annotations, `set_xlim(0, 11)`, `set_ylim(0, 1)`, no `tight_layout` per redraw (use `constrained_layout=True` once) |
| Z2 part a table | means per configuration | add the paired difference to standard at TOP_K 5 with 95 % interval and effect size (same helper as F4). The cache gives, for Context Recall at 5: minilm clearly worse (large effect), markdown not distinguishable (interval includes 0), llama clearly higher (large effect, but a measurement change). |
| Z3 part a feedback | "Embedding model and chunking change what is found; those are levers on the system. The judge changes only how strictly it scores ..." | keep, and add: **de** Dass markdown_headers schlechter aussieht, belegt der Datensatz nicht: Das Intervall enthält die Null. Hier entscheiden Chunkgröße, Promptlänge und Aufwand, nicht der Score. Und: Jede Konfiguration wurde einmal bewertet; die Streuung aus Abschnitt E c steckt in jedem Wert. **en** That markdown_headers looks worse is not shown by the data set: the interval contains zero. Here chunk size, prompt length and effort decide, not the score. And: each configuration was judged once; the spread of section E c sits in every value. |
| Z4 part a step 1 | "compare its curves with the standard configuration" | "compare its curves and the difference table with the standard configuration" |

### 4.8 Section H summary

| # | Old | New |
|---|---|---|
| H1 card TOP_K | "More chunks raise recall and lower precision; the sweet spot lies at 3 to 5." | **de** Mehr Chunks heben Recall und senken Precision. Zwischen 3 und 5 belegen die Daten keinen Unterschied. **en** More chunks raise recall and lower precision. Between 3 and 5 the data show no difference. |
| H2 new card | none | **de** Signifikanz und Effektstärke. Ein Unterschied zählt erst, wenn er die Streuung über die Fragen übersteigt und groß genug ist. **en** Significance and effect size. A difference counts only when it exceeds the spread over the questions and is large enough. |
| H3 key takeaway | current text | append: **de** Sind Unterschiede klein, entscheiden Kosten, Geschwindigkeit und Promptlänge. **en** When differences are small, cost, speed and prompt length decide. |

## 5. Mentimeter block (runs in the room, not in the notebook)

Format: "Guess the number" (Mentimeter type "Guess the number" or an open number question), two rounds with the same three texts. Round 1 before D2 without a definition, round 2 after D3 with the working definition. Texts from the gold data set (reference answers), chosen so that the count depends on the definition.

| Text | Source | Count by everyday / logic / working definition |
|---|---|---|
| "Administrationskonten sollten durch Mehr-Faktor-Authentisierung geschützt werden, da sie ein besonders hohes Schadenspotenzial besitzen und bevorzugtes Angriffsziel darstellen." | gold 33 | 1 / 3 to 4 / 3 |
| "Erstellen Sie eine Cloud-Nutzungsstrategie (A1), eine Sicherheitsrichtlinie (A2), Service-Definitionen (A3) und legen Sie Verantwortungsbereiche sowie Schnittstellen fest (A4). Diese vier Basis-Anforderungen müssen vorrangig erfüllt werden." | gold 17 | 2 / 5 to 6 / 5 (four documents, "vorrangig"; the second sentence repeats the first, so it does not count) |
| "Bei personellen Veränderungen müssen nicht mehr benötigte Benutzendenkennungen und Berechtigungen unverzüglich entfernt werden, um unberechtigte Zugriffe ehemaliger oder versetzter Mitarbeitender zu vermeiden. Es dürfen Berechtigungen nur hinsichtlich des tatsächlichen Bedarfs zur Aufgabenerfüllung vergeben werden." | gold 27 | 2 / 4 to 5 / 4 (remove accounts, remove rights, immediately, purpose; sentence 2) |

Reserve text if a fourth is wanted: gold 10 (one sentence listing five measures; 1 / 5 / 5).

English versions for the English run are the cached translations (`translations.json`, en); check them once against the counts above.

## 6. Internal bridges (changed ones only)

| From | To | The one new thought |
|---|---|---|
| D chain figure | D1 Mentimeter | before the judge counts claims, you find out that people do not agree on the count |
| D3 working definition | D claim table | the judge works without a definition, so its count is one of several possible counts |
| E a line 72 | E b two judges | whether a claim is supported depends on context, and so does how many claims there are |
| F figure 1 | F4 significance | a difference between two means is only a difference when it exceeds the spread over the questions |
| F4 | G choice | where the data cannot decide, cost decides |
| G | extra task a | the same test applies to configurations, and one of the three changes only looks like a difference |

## 7. Parking lot

| Item | Destination | Reason |
|---|---|---|
| Wilcoxon signed-rank test as a second method | drop | one method is enough for the key message; two methods disagreeing on k1 to k3 precision (bootstrap interval includes 0, Wilcoxon p = 0.008) would need a discussion the slot has no time for |
| Multiple judge runs for all four configurations | FAQ, "Why only one run per configuration?" | cost; the demo question in E c shows the mechanism |
| Non-LLM variants of Context Precision and Recall | already FAQ | unchanged |
| Qdrant and collection setup | FAQ F4 | instructor topic |
