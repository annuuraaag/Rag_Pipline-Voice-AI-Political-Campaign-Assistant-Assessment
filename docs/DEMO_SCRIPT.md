# Demo video script (about 6 minutes)

A screen-recording script for showing the assistant end to end. Each scene says what to **do** on
screen and what to **say**; timings are a guide.

The upload in scene 4 uses `sample_data/demo/nellore_coastal_plan.pdf`. This fictional Nellore plan
is **not** in the preloaded corpus, so the video shows the assistant learning new facts live.
To rebuild it, run `python scripts/build_demo_doc.py`.

---

## Before you record (checklist)

- [ ] `.env` has `GROQ_API_KEY=...`. Without it, answers are extracts rather than written sentences.
- [ ] Start from an empty database so the corpus is exactly the sample set:
      `docker compose down -v && docker compose up --build`
- [ ] Open http://localhost:5173 in **Chrome** and allow the microphone.
      Speech in the browser needs Chrome or Edge.
- [ ] Do one warm-up question off camera. The first one loads the models and is slower.
      Then click **New conversation** (the circular-arrow button).
- [ ] Have `sample_data/demo/nellore_coastal_plan.pdf` open in a file explorer window, ready to drag.
- [ ] Close notifications. Use a headset mic, a quiet room and 1080p, with the browser zoom at 100–110 %.

---

## Scene 1 — Intro (0:00–0:30)

**Do:** Show the Assistant page, with the Knowledge base list visible in the sidebar.

**Say:**
> "This is a voice assistant for a political campaign. Voters ask about the manifesto, schemes and
> district plans by voice, and every answer is grounded in the campaign's own documents, with
> citations. Everything you'll see is fictional sample data."

## Scene 2 — Voice question with citations (0:30–1:30)

**Do:** Tap the orb and **speak**: *"I'm from Vijayawada."* Wait for the acknowledgement, then
*"What healthcare schemes are available?"*

**Say (while it answers):**
> "It's transcribing while I talk and starts searching before I finish. Notice it stops listening
> almost as soon as I finish the sentence: it can tell when a question sounds complete."

**Do:** Click a citation number, e.g. **[S1]**, to open the source passage.

**Say:**
> "Each sentence cites its source: file, page and section come from the document metadata, so
> citations can't be made up."

## Scene 3 — Follow-up memory and interruption (1:30–2:30)

**Do:** Speak *"What about education?"*

**Say:**
> "I didn't repeat the district. It remembers I'm in Vijayawada and rewrites the question."

**Do:** Ask *"Who is eligible for the Pratibha
Scholarship?"* While it's speaking the answer, talk over it: *"What is planned for chilli farmers
in Guntur?"*

**Say:**
> "I can interrupt it, just like a person. It stops talking and answers the new question."

## Scene 4 — Upload a new document live (2:30–3:45) ⭐

**Do (before uploading):** Type *"What will fishing families in Nellore get during the fishing
ban?"*

**Say:**
> "Right now it only knows the statewide policy, 10,000 rupees per family."

**Do:** Open **Knowledge base** and drag `nellore_coastal_plan.pdf` onto the drop zone. Point at
the detected metadata (district **Nellore**, category **district profile**) and the passage count.
Click the document to show its passages.

**Say:**
> "I'm uploading a new district plan. It's parsed, split by section and indexed in under a second,
> and the district is detected automatically."

**Do:** Back on **Assistant**, ask the same question again, by voice or text.

**Say:**
> "Now it answers from the new plan: 12,000 rupees under the Samudra Suraksha Scheme, for the 61-day
> ban, with the Nellore PDF as the citation."

**Optional extra questions on the new document:**
- *"Where will the new fishing harbour be built?"* → Krishnapatnam, 400 boats
- *"How many cyclone shelters are planned in Nellore?"* → thirty, 500 people each
- *"Who is the Nellore district coordinator?"* → Ms. Lakshmi Prasanna Reddy

## Scene 5 — Safety: refusing and checking itself (3:45–4:45)

**Do:** Ask *"What is the campaign's policy on lunar mining?"*

**Say:**
> "When the documents don't cover a topic, it refuses instead of guessing. The relevance check
> runs before the language model is even called."

**Do:** Open the **details panel** on a previous answer: show **Sources**, the **Retrieval** trace
(rewritten query, district filter, candidate funnel) and the **Claim check** list.

**Say:**
> "Every sentence of an answer is checked against the passage it cites: numbers, districts and
> names must actually be there. Wrong citations are corrected, and unsupported sentences are flagged."

## Scene 6 — Speed and settings (4:45–5:30)

**Do:** Open the **Latency** tab of an answer, then **Insights**.

**Say:**
> "Here is the time spent in each stage, and live P50, P90 and P95 latency across all
> questions."

**Do:** Open **Settings** and point at **Speech recognition** / **Answer voice** (Browser or
Server), and **Smart end-of-question detection**.

**Say:**
> "Speech can run in the browser or on the server: locally for privacy, or with cloud providers
> like Deepgram for accuracy."

## Scene 7 — Close (5:30–6:00)

**Do:** Show the GitHub README, scrolling to **Evaluation and results**.

**Say:**
> "Retrieval, answers, claim checks, end-of-question detection and voice latency each have an
> evaluation script with stored results in the repo. Thanks for watching."

---

## If something goes wrong while recording

| Problem | Fix |
|---|---|
| The mic mishears a place name | Repeat it more slowly, or type the question. Typing is fine on camera. |
| The first answer is slow | Do a warm-up question before recording (see checklist). |
| The upload says "duplicate" | The PDF is already indexed. Delete it in Knowledge base and upload again. |
| Answers read like copied text | `GROQ_API_KEY` is missing. Add it to `.env` and restart. |
| Interrupting by voice doesn't work | Check **Interrupt by speaking** is on (Settings). With laptop speakers, speak a full question (3+ words) clearly over the voice, or use a headset. |
