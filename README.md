# LearnForge Support Assistant

This is a prototype AI support assistant for LearnForge. It answers user questions using the FAQ/policy/ticket data, keeps track of multi-turn conversations, and hands off to a human
when it is not sure of the answer.

Architecture diagram: [`architecture.md`](architecture.md)


## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your-key-here   # free key: https://aistudio.google.com/apikey

python src/chat.py
```

 You can run
`python src/retriever.py` on its own just to see what it retrieves for a
few sample questions.

To run the test questions:

```bash
python eval/run_eval.py
```

## How it works

1. **Retrieval** — Each FAQ, policy, and ticket in
   `data/` is turned into one chunk of text. Each chunk gets converted into
   a vector (a list of numbers that captures its meaning) using a small
   free model called `all-MiniLM-L6-v2`. When a user asks something, their
   question gets converted into a vector too, and I compare it against
   every chunk's vector to find the closest matches. I first tried a
   simpler method (TF-IDF, which just matches on shared words) but it
   missed things — for example, a search for "money back" didn't find the
   refund FAQ, because "money back" and "refund" don't share any words.
   Vectors fix that, since they capture meaning, not just exact words. I
   didn't use a tool like FAISS for storing the vectors — with only 40
   chunks, comparing a question against all of them takes almost no time,
   so a dedicated vector database wouldn't make anything faster here. It
   would matter once there are thousands of chunks (see
   `ddata_schema.md`).
2. **Generation** — Gemini gets the user's question
   plus the matching chunks, and is told to answer using *only* that
   information. Instead of a plain text reply, it returns a structured
   answer: the answer text, how confident it is, whether the answer is
   actually backed up by the retrieved chunks, which chunks it used, and
   whether a human should take over. This makes it easy for the code to act
   on the "should this escalate?" decision directly, instead of guessing
   based on how the answer sounds.
3. **Multi-turn** — Before searching for
   relevant chunks, the assistant first rewrites the user's newest message
   into a full, standalone question, using the earlier conversation as
   context. So if someone first asks about course refunds, then asks "what
   about subscriptions instead?", that gets turned into a full question
   about subscription refunds before searching.
4. **Deciding when to escalate** — The assistant hands off to a human if
   either of these is true: Gemini itself decides to (it's told to do this
   for fraud, account security issues, refund requests outside the normal
   time window, or anything unclear), or the best-matching chunk wasn't a
   good enough match to trust in the first place. Either one is enough on
   its own to trigger a handoff.

## How it handles failures

- **When it's not confident in an answer** — If the best-matching chunk
  isn't a strong enough match, the assistant hands off to a human instead
  of guessing. Even when the match is good enough, Gemini is asked to say
  how confident it is and whether its answer is actually backed by the
  chunks — and it's told to hand off rather than guess whenever it isn't
  sure.
- **When the data is outdated or contradicts itself** — A few of the
  source documents deliberately include an old, wrong statement that gets
  corrected further down (for example: "an older version of this said 7
  days — that's outdated, the real answer is 14 days"). The assistant is
  told to always go with the current, corrected statement, and to lower
  its confidence (rather than pick one at random) if two chunks genuinely
  disagree and it can't tell which one is right.
- **When the search finds the wrong chunks** — Vector search handles
  different wording much better than plain keyword matching, but it's
  still just comparing the question against a fixed, small snapshot of the
  knowledge base. It can't combine information from several articles at
  once, and it doesn't double-check or re-rank its results beyond the raw
  similarity score. This is the weakest part of the current system, and
  the first thing I'd improve with more time (see Trade-offs below).

## How I'd measure quality (eval plan)

`eval/eval_questions.json` has 10 test questions: a few plain answerable
ones, two that test the outdated/contradicting-data handling, a couple
that should clearly hand off to a human (fraud, account compromise), one
that's completely off-topic, and one multi-turn follow-up.

`eval/run_eval.py` automatically checks two things:

- **Did search find the right chunk?** — for each question, I know which
  source chunk should show up in the results. The script checks whether it
  actually did.
- **Did it hand off when it should have (and not when it shouldn't)?** —
  for each question, I know whether it should escalate. The script checks
  whether the assistant agreed.

What this does *not* check automatically is whether the answer itself is
actually correct and doesn't make anything up. To measure that properly, I
would add:
- A second AI call that reviews each answer against the chunks it was
  given, and scores whether every claim in the answer is actually backed
  up by those chunks — this gives a real hallucination-rate number.
- A person spot-checking a sample of real conversations once this is
  live, since that's what actually reflects whether users are getting good
  help.
- A second set of test questions, phrased informally or with typos, to
  make sure the assistant isn't handing off things a human wouldn't need
  to — right now my test set is weighted toward checking it hands off
  *enough*, not toward checking it doesn't hand off *too much*.

## Trade-offs — what I'd change with more time

- **Search**: already uses real vector search (see "How it works" above).
  The next step at real scale would be moving from directly comparing
  against every chunk to using FAISS or a proper vector database (like
  pgvector or Pinecone) once there are too many chunks to compare against
  one by one, plus combining vector search with plain keyword search for
  things like order numbers that vectors aren't great at matching exactly.
- **Outdated info**: right now, "this is outdated" is something the
  assistant has to figure out from the wording of the text itself, because
  that's how the sample data was written. In a real system, that should
  instead be a clear field in the database — something like "this article
  was replaced by that one" and "last reviewed on this date" — instead of
  something the AI has to guess from prose every time.
- **Handing off to a human**: right now, escalating just means the
  program prints a warning message. In a real system, this needs to
  actually create a support ticket, with the full conversation and the
  chunks it looked at attached, so the human agent isn't starting from
  scratch.
- **Remembering conversations**: right now, the conversation history only
  lives in memory while the program is running, and disappears once you
  close it. A real system needs to save conversations somewhere permanent,
  and probably needs to shorten long conversation histories instead of
  sending the entire thing to the AI every single time.
- **Why Gemini**: I picked Gemini's free tier because it was the easiest
  of the three suggested options to get an API key for quickly, and it
  supports returning answers as structured JSON well. Switching to Groq or
  OpenRouter later would just mean writing one new small file with the
  same two functions (`rewrite_query` and `generate_answer`) — the rest of
  the code doesn't need to change.
