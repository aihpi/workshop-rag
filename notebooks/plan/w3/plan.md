# RAG III notebooks (w3_01 to w3_04), rework plan

## 1. Header

- Level: workshop
- Mode: refine (material exists, no earlier plan.md in the repository; Jill's meta plan that the code comments cite is not checked in)
- Language: de and en in one file per notebook, switched with `t('de', 'en')`
- Status log:
  - 2026-10-21 created from the four notebooks, Hanno's review notes and a concept audit (Explore agent, read all four notebooks)
  - 2026-10-21 audience and presenter roles skipped on request
  - 2026-10-21 revision 2 confirmed by Hanno (plan mode, approved): see "Decisions of revision 2" below
  - 2026-10-21 creator, N1 N3 N4 cross-cutting (fork agent), complete; creator, N2 (main agent), complete; creator, slides S2 (fork agent), complete
- Change log:
  - N2 interface version 1 to 2: goals served gain G4 (decide between two configurations with a significance and effect size reading); exit state as in the component plan. Confirmed 2026-10-21.

## 0. Decisions of revision 2 (Hanno, 2026-10-21)

Terms assumed known from RAG I and II, no hint: chunk, embedding, similarity, retrieval, search, chunking, TOP_K, claim, prompt, system prompt, answer model, median. Section E a of w3_02 (line 72) stays unchanged. All live buttons stay; participants get a key at RAG III and the README gets a short section "Before RAG III: your API key". No translation stage: judge reasons stay English in both versions, one hint per notebook. Repeated judge runs for figure 1 are GitHub issue #33 in aihpi/workshop-rag, not done now. F4 ends with the held-out set sentence. The dictionary card also allows two claims (main and subordinate clause). Slides: separate alt decks slides/rag_iii_slides_alt_german.pdf and rag_iii_slides_alt_english.pdf, each page 1 of rag_ii_slides_english.pdf, eight new claim slides (slides/src/rag_iii_claims_*.pptx, built by slides/src/build_claims.py), pages 10 to 22 of rag_ii_slides_english.pdf. Sections 4 and 5 below are superseded where they list a term from the assumed-known list or the translation stage.

## 2. Learner and key message

Learner as in the RAG III workshop page of kisz-context (practitioners who build or buy a RAG system and must judge whether it works). Key message of the series after the rework: RAGAS numbers estimate how well a RAG system works, roughly. In the detail they have pitfalls (claim definition, judge choice, run-to-run spread, small samples), so a difference between two numbers needs a significance and effect size check before it changes a decision, and when the effect is small, cost and speed decide.

## 3. Components

| ID | File | Minutes | Change in this rework |
|---|---|---|---|
| N1 | w3_01_intro_end_to_end.py | unchanged | cross-cutting only (language, concepts) |
| N2 | w3_02_retrieval_evaluation.py | 60 | main rework, see components/N2-w3_02-retrieval-evaluation/plan.md |
| N3 | w3_03_generation_evaluation.py | unchanged | cross-cutting; section H "When does a difference count?" must agree with N2 F4 (same test, same wording); two quiz references to w3_01 section letters wrong |
| N4 | w3_04_rlhf_rlaif.py | unchanged | cross-cutting only |

## 4. Cross-cutting change 1: one language per notebook

Today every notebook shows corpus text (website lines, Kompendium chunks, gold questions and answers, judge reasons, claims) in German with an English translation in grey below it when the English version is on. The English version therefore reads German first. The German version shows the judge's reasons in English with a hint that RAGAS prompts are English.

Rule after the rework: the participant reads the notebook's language only. The helper `bi` (in all four notebooks) shows the translation as the text in the English version and the original in the German version. The original is available in an accordion "Original (Deutsch)" under the passage, nowhere else. Judge reasons and claim statements get a German translation for the German version. The hint about English RAGAS prompts moves to the FAQ ("Why does the judge think in English?").

One exception, stated in the notebook: w3_02 extra task part b argues from the German wording ("erhöhter Schutzbedarf" vs the abbreviation "(H)"); there the German chunk headings stay visible next to the translation, with one sentence saying why.

Asset need: `tools/precompute_w3_02.py` (and the siblings for w3_01, w3_03, w3_04) get a translation stage into German for every cached English judge text. `data/cache/w3_02/translations.json` has 362 en entries and 31 de entries today, so most reasons lack a German text. Until the stage has run, the German version falls back to English with the current hint.

Implementation note for creator: `bi`, `bi_cell`, `en_of` are copied into each notebook; the rework puts one shared version into `ragkit/theme.py` or a new `ragkit/corpus_text.py` (one function `show(text, t)` plus `cell(text, t, n=None)`) and the four notebooks import it. Not in scope: translating the Kompendium itself; only cached texts shown to participants.

## 5. Cross-cutting change 2: concepts used before they are introduced

Audit of the four notebooks (English strings, 2026-10-21). "Fix" names the smallest change; a one-sentence definition goes where the term first appears, as a grey hint, unless noted.

### Gaps across the series

| Term | First use | Fix |
|---|---|---|
| chunk | w3_01 B (table), w3_02 A | never defined in parts III. One hint in w3_01 A or B: "A chunk is one piece of the corpus, the unit the search returns; in notebook 1 every website line is one chunk." |
| embedding, similarity | w3_01 C (diagram), w3_03 C | one hint in w3_01 C: "An embedding is the vector a model computes for a text; close vectors mean similar texts." |
| retrieval, search, chunking | w3_01 B | hint in w3_01 B: "Chunking cuts the corpus into chunks; retrieval (the search) picks the chunks closest to the question." |
| TOP_K | w3_02 E d caption, before section F; absent from w3_01 | one hint at the first use in w3_02 (section A card or E d): "TOP_K is the number of chunks the search returns." w3_01 E says "the three most similar"; add "(TOP_K 3)". |
| Qdrant, collection | w3_02 G quiz option, FAQ | quiz option reworded to "the search"; FAQ F4 explains Qdrant in a clause. |
| gpt-oss-120b | w3_01 FAQ only, then w3_02 B via the judge name | w3_01 B where the judge is introduced: "here gpt-oss-120b, an open model on the AISC API". |
| llama-3-3-70b, "second judge" | w3_02 E b and E d | one clause at E b: "llama-3-3-70b, a second open model, so a different model family". |
| standard configuration, configuration names (standard, minilm, markdown, llama), miniLM, markdown_headers, embedding model | w3_02 F caption, extra task | one table in w3_02 section A (collapsed accordion "The four configurations"): name, chunking, embedding model, judge, and what differs from standard. The extra task dropdown uses the same names. |
| Kompendium, BSI, module, requirement, (H), B/S/H | w3_01 D names it; w3_02 extra task b uses the jargon; w3_03 D explains B/S/H | move the one-paragraph explanation of the Kompendium (BSI document, modules, requirements with B/S/H) to w3_01 D; w3_03 D keeps a short reminder. |
| claim (statement) | w3_01 E says "statement"; w3_02 D defines "claim" | w3_01 uses "claim" at E too, with "(RAGAS calls it a statement)". w3_02 D gets the definition block (N2 plan 4.2). |
| prompt, system prompt, answer model | w3_02 F ("space in the prompt"), w3_03 B before I | w3_02 F hint: "the prompt is the text the answer model receives: question plus chunks". w3_03 B links forward: "(section I shows it)". |
| cache | error callouts in every notebook | fine for the error case; no fix. |
| median | w3_01 E tiles, w3_03 I | one clause: "median, the middle value of the three runs". |
| hybrid search, reranking | w3_03 G | one clause each, or an FAQ entry "What are hybrid search and reranking?". |
| AC (abbreviation) | w3_03 E | write "Answer Correctness" in full; the abbreviation saves nothing. |
| alignment, pretraining, SFT, PPO, policy, reinforcement learning | w3_04 B, C, D, FAQ | w3_04 C gets the three-step chain with one clause each (pretraining, supervised fine-tuning, preference tuning); PPO stays an FAQ term. |
| reasoning model, temperature | w3_03 FAQ, w3_04 FAQ | fine in FAQ; no fix. |

### Late definitions (term used, defined later in the same notebook)

| Term | Used | Defined | Fix |
|---|---|---|---|
| gold passage (Fundstelle) | w3_01 B | w3_01 D | forward reference "(section D)" at B |
| Faithfulness | w3_01 B table | w3_01 C | one clause in the table cell |
| system prompt | w3_03 B | w3_03 I | forward reference |
| Constitutional AI | w3_04 F feedback | w3_04 G | feedback already says "section G"; fine |

### Wrong cross-references

w3_03 final quiz: "Notebook 1, section E" for RAGAS test data should read section F; "sections C and E" for what the judge needs should read C and F. Cause: w3_01 code comments use other letters than the rendered headings (D is commented "Section D (slide)", F is commented "Section E").

## 6. Interface blocks

Only N2 changes; the block is in the component plan. Entry and exit states of N1, N3 and N4 stay. N3 H ("When does a difference count?") and N2 F4 must use the same method and wording; creator reads N3 H before writing N2 F4 and reports any difference as a change request.

## 7. Mentimeter block for N2 section D

Runs in the room, twice, with the same three texts: once before any definition, once after the working definition. Texts, counts and the definition are in the component plan, section 5. Mentimeter type "Guess the number" (one slide per text, three slides per round) or one open number question with the three texts on the slide.

## 8. Open points (TBD)

| Point | Question to the user |
|---|---|
| Multiple judge runs for figure 1 | Should `precompute_w3_02.py` judge the standard configuration three times per TOP_K so figure 1 shows a run band? Cost about 3 × 43 × 23 calls on the AISC gateway, once. |
| Translation stage | Translate all cached judge reasons and claims into German with gpt-oss-120b (one run of the precompute tool)? Without it the German version keeps English reasons. |
| Hanno's note "the first time when metrics are computed and compared against each other, this should be followed ..." | The sentence breaks off. Assumed: followed by the significance and effect size reading (N2 plan F4). Confirm or complete. |
| Scope of the cross-cutting changes | Apply the language rule and the concept hints to all four notebooks in this round, or N2 only first? |
