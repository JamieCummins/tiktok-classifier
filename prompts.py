"""
prompts.py — Classification prompts and input formatting.

This is the methodological core of the classifier. Edit here to change:
  - Classification criteria (what counts as mental-health content)
  - How video metadata is presented to the model
  - Few-shot example format

The two prompts implement a deliberate two-stage strategy:
  - SCREEN_SYSTEM_PROMPT: cast a wide net; false positives are acceptable
  - CLASSIFY_SYSTEM_PROMPT: fine-grained; applied only to screen-passing rows
"""

import pandas as pd

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Pass 1: cast a wide net — anything that could plausibly relate to mental
# health should be labelled TRUE. False positives are acceptable here; they
# will be filtered out in the second pass.
SCREEN_SYSTEM_PROMPT = """\
You are a first-pass filter for social media video content.

You will receive up to three metadata fields per video:
- description: text written by the creator
- transcript: auto-generated speech-to-text of the audio
- suggested_words: TikTok-generated search terms derived from the full video content


Label TRUE if the video contains at least one clear signal of mental health
relevance:
- A named mental health condition (depression, anxiety disorder, ADHD, PTSD, OCD, bipolar disorder, eating disorders, self-harm, psychosis, autism, BPD, substance use disorder, etc.) — whether as a hashtag, in a description, or in a transcript
- Reference to psychological or psychiatric treatment (seeing a therapist,
  psychologist, or psychiatrist; psychiatric medication; mental health diagnosis)
- First-person description of psychological symptoms or emotional struggles
  explicitly tied to a condition
- Community or peer support content explicitly framed around mental health

Label FALSE if mental health terms appear only as:
- Clear jokes or colloquial hyperbole ("that's so depressing")
- Song or audio lyrics — identifiable by ♪/♫ symbols in the transcript,
  "(singing)", "(music)", or similar ASR markers — unless the creator is
  clearly speaking about their own experience outside the lyrics
- Generic wellness, self-care, fitness, or relaxation content — including
  colloquial "therapy" (e.g. "ocean therapy", "retail therapy", "colouring therapy")
- Physical health treatments (e.g. chemotherapy) with no psychological framing

When uncertain, label TRUE. Missing a relevant video is worse than passing
an irrelevant one through to the next stage.

Reply with a single token: TRUE or FALSE.

{examples}"""

# Pass 2: applied only to rows that passed the screen. Distinguishes genuine
# mental-health content from spurious matches.
CLASSIFY_SYSTEM_PROMPT = """\
You are a second-pass classifier for social media video content. Every video
you see has already passed a broad screen confirming it contains some mental
health signal. Your task is to determine whether that signal reflects genuine
engagement with mental health.

You will receive up to three metadata fields per video:
- description: text written by the creator
- transcript: auto-generated speech-to-text of the audio
- suggested_words: TikTok-generated search terms derived from the full video content

The content must engage with mental health specifically — not just physical
health, disability, or everyday emotional experience. Recognised mental health
conditions include: depression, anxiety disorders, PTSD, OCD, ADHD, bipolar
disorder, eating disorders, self-harm, psychosis, BPD, phobias, substance use
disorder, autism spectrum, dissociation, and similar. Physical health conditions
(sickle cell, PCOS, hypermobility, chronic pain, wheelchair use, food allergies,
cancer, etc.) do NOT qualify unless the content explicitly discusses a
co-occurring mental health condition by name or clear description.

Read all fields together: a hashtag is not isolated — if the description or
transcript provides MH context, a condition hashtag confirms the topic. Weight
the description heavily; creators often use it to frame what the video is about.

Label TRUE if the content meaningfully engages with mental health through any of:
- First-person description of symptoms, diagnosis, or lived experience of a
  recognised MH condition — including condition-specific language or slang
  (e.g. "crash out" for emotional dysregulation, "being sectioned" for
  psychiatric hospitalisation, "masking" for autism camouflaging)
- Discussing psychological or psychiatric treatment in a personal context:
  therapy sessions, therapeutic modalities (IFS, CBT, DBT, trauma therapy,
  psychotherapy), psychiatric medication, or psychiatric hospitalisation
- Offering or seeking coping strategies, support, or solidarity explicitly
  around a MH condition
- Expressing community membership, identity, or solidarity framed around a
  recognised MH condition — including awareness/educational content,
  participation in MH-coded communities, and meme or humour-format content
  whose central subject is a condition (e.g. "my ADHD be like", BPD community
  content, autism community content)
- Profound grief or trauma (e.g. bereavement, loss of a loved one, severe
  traumatic events) where the psychological impact is explicitly discussed —
  not incidental sadness or routine disappointment
- When the transcript contains only music or is absent: a preponderance of
  converging signals across description, hashtags, and suggested_words that
  make a specific MH condition clearly the central topic. Apply this when the
  condition appears to be the primary subject — not when a single MH hashtag
  sits alongside many unrelated lifestyle or entertainment tags.

Label FALSE if the mental health signal is spurious:
- A single MH-adjacent hashtag (e.g. #anxiety, #stress) embedded among hobby,
  lifestyle, or entertainment hashtags where the video is clearly about something
  else — such hashtags function as personality identifiers, not content
  descriptors. This is different from a condition being the primary subject: if
  the condition is the main or sole hashtag and the description or suggested_words
  are also centred on it, label TRUE.
- Colloquial expressions or jokes with no supporting content (e.g. "I have PTSD
  from this", "that's so OCD", "Teams ringtone PTSD")
- MH terms appear only in song/audio lyrics (identifiable by ♪/♫, "(singing)",
  "(music)" in the transcript) and the creator is not speaking about their own
  experience outside the lyrics
- The mention is incidental to content primarily about something else (e.g.
  therapy mentioned once in passing; ADHD as a throwaway excuse)
- The content discusses a third party's or fictional character's condition
  without the creator expressing personal connection, solidarity, or lived
  experience (e.g. analysing a TV character's diagnosis with no personal framing)
- Physical health conditions, chronic illness, or disability content (sickle
  cell, PCOS, hypermobility, EDS, chronic pain, food allergies, wheelchair use,
  cancer, chemotherapy, cosmetic surgery, etc.) unless a co-occurring MH
  condition is explicitly named or clearly described — "psychological impact"
  or "living with" a physical condition is not sufficient
- Wellness, self-care, or relaxation content using MH-adjacent language with
  no clinical or lived-experience framing (e.g. "ocean therapy", "retail
  therapy", "good for my mental health")
- Routine emotional experiences (unrequited love, heartbreak, relationship
  drama, career frustration, social conflict, general stress) not tied to a
  recognised MH condition — profound grief or trauma may qualify only if the
  psychological impact is explicitly discussed (see TRUE criteria above)
- Content about colloquial "narcissism" or "toxic" behaviour in relationships
  without clinical framing of a personality disorder

Reply with a single token: TRUE or FALSE.

{examples}"""

STAGE_PROMPTS = {
    "screen":   SCREEN_SYSTEM_PROMPT,
    "classify": CLASSIFY_SYSTEM_PROMPT,
}

# Ground-truth column used for evaluation and few-shot examples (varies by stage)
STAGE_TRUTH_COLS = {
    "screen":   "is_superficial_mental_health_handcoded",
    "classify": "is_mental_health_handcoded",
}

# ---------------------------------------------------------------------------
# Input formatting
# ---------------------------------------------------------------------------

# Template wrapping per-video metadata into the user turn
USER_TEMPLATE = "Video metadata:\n{text}"


def row_fields(row: pd.Series, cols: list[str]) -> dict[str, str]:
    """Return the non-empty metadata columns of a row as {name: value}."""
    return {
        c: str(row[c]) for c in cols
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip()
    }


def row_text(row: pd.Series, cols: list[str]) -> str:
    """Concatenate non-empty metadata columns into a labelled prompt input."""
    return "\n".join(f"{c}: {v}" for c, v in row_fields(row, cols).items())


# Jev (discriminative model) answers a true/false question about a structured
# state rather than generating text, so the single-token reply instruction is
# replaced with an explicit question. The criteria themselves are unchanged.
JEV_QUESTION = "Question: should this video be labelled TRUE under the criteria above?"


def to_jev_instructions(system_prompt: str) -> str:
    """Adapt a generative system prompt for use as Jev noul instructions."""
    return system_prompt.replace(
        "Reply with a single token: TRUE or FALSE.", JEV_QUESTION
    )


def format_examples(
    examples_df: pd.DataFrame,
    cols: list[str],
    label_col: str,
    include_rationale: bool = True,
) -> str:
    """Build the few-shot examples block for injection into the system prompt."""
    if examples_df.empty:
        return ""

    parts = ["Examples:\n"]
    for _, ex_row in examples_df.iterrows():
        text = row_text(ex_row, cols)
        if not text.strip():
            print(f"Warning: no metadata for example {ex_row['url']}, skipping")
            continue

        block = f"Video metadata:\n{text}\nLabel: {ex_row[label_col]}"

        if include_rationale:
            rationale = ex_row.get("rationale")
            if pd.notna(rationale) and str(rationale).strip():
                block += f"\nRationale: {rationale}"

        parts.append(block)

    if len(parts) == 1:
        return ""

    return "\n\n".join(parts) + "\n"
