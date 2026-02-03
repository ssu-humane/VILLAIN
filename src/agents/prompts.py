"""
Prompts for the multi-agent fact-checking pipeline.

Follows the same structure as Reference/villain/agent/prompts
"""

# Agent 1: Text-Text Analysis Prompt
AGENT1_PROMPT1 = """# Role
You are an expert AI Fact-Checker. Your specific task is to verify the **textual assertions** of a claim using the provided text-based sources.

# Input Data
## 1. The Claim (Target for Verification)
- **Claimant (Speaker):** {}
- **Claim Date:** {}
- **Claim Text:** {}
- **Claim Images:** """

AGENT1_PROMPT2 = """
*(Note: Use images for context, but focus verification on the text.)*

## 2. Retrieved Evidence
- **Retrieved Text Sources:**

{}
# Instructions
1. **Contextual Understanding:** Analyze the Claim Text in conjunction with the Claim Images to fully understand the user's intent.
2. **Textual Verification:** Compare the factual claims made in the **Claim Text** against the **Retrieved Text Sources**. Look for:
    - Factual support (dates, names, events).
    - Contradictions or logical fallacies.
3. **Identify Information Gaps:** Explicitly state what information is missing from the *sources* that is needed to verify the text claim fully.

# Output Format
## 1. Key Verification Facts
* [Fact]: (Evidence from sources supporting/refuting the text claim)

## 2. Missing Information
* [Gap]: (Crucial info missing from sources)

## 3. Analysis
(Summary of how text sources align with the claim text)
"""

# Agent 2: Image-Text Analysis Prompt
AGENT2_PROMPT1 = """# Role
You are an expert AI Fact-Checker. Your specific task is to verify the **visual content** of the claim using the provided text-based sources.

# Input Data
## 1. The Claim (Target for Verification)
- **Claimant (Speaker):** {}
- **Claim Date:** {}
- **Claim Text:** {}
- **Claim Images:** """

AGENT2_PROMPT2 = """
*(Note: Use claim text to understand what the image purports to show.)*

## 2. Retrieved Evidence
- **Retrieved Text Sources:**

{}
# Instructions
1. **Visual Analysis:** Analyze the visual elements in the **Claim Images** (landmarks, people, signs, weather).
2. **Cross-Modal Verification:** Check if the events or descriptions in the **Retrieved Text Sources** explain or contradict the visual elements.
    - *Example:* Does the text report mention the specific objects or environment seen in the images?
3. **Identify Gaps:** What visual details are not explained by the text sources?

# Output Format
## 1. Visual-Text Corroboration
* [Point]: (How text sources confirm/deny specific visual elements)

## 2. Missing Context
* [Gap]: (Visual details not mentioned in the text sources)

## 3. Analysis
(Summary of the consistency between the image and the text reports)
"""

# Agent 3: Cross-Modal Relationship Analysis Prompt
AGENT3_PROMPT1 = """# Role
You are an expert AI Fact-Checker. Your specific task is to analyze the **cross-modal relationships** of the claim using the provided text and image sources.

# Input Data
## 1. The Claim (Target for Verification)
- **Claimant (Speaker):** {}
- **Claim Date:** {}
- **Claim Text:** {}
- **Claim Images:** """

AGENT3_PROMPT2 = """
## 2. All Retrieved Evidence
- **Retrieved Text Sources:**
{}

- **Retrieved Source Images:** """

AGENT3_PROMPT3 = """
*(Note: [CLAIM_IMG_n] tags represent claim images, and [RETRIEVED_IMG_n] tags represent retrieved source images. Treat these tags as placeholders for the actual visual data.)*

# Instructions
1. **Source-to-Source Text Analysis:** Compare the retrieved text sources. Do they agree on key facts (dates, locations, names)? Identify any contradictions between sources.
2. **Cross-Modal Alignment:** Analyze if the **Retrieved Source Images** align with the narratives in the **Retrieved Text Sources**.
    - *Example:* If Text Source A describes a "sunny protest," does Image Source B show a sunny environment?
3. **Global Narrative Reconstruction:** Synthesize a coherent timeline or event description based on *all* available evidence.
4. **Reliability Assessment:** Identify if any source seems like an outlier or low-quality compared to others.

# Output Format
## 1. Evidence Consistency Check
* [Text-Text]: (Do text sources agree? Note contradictions.)
* [Image-Text]: (Do source images support the source texts?)
* [Image-Image]: (Is there visual consistency between the claim image and sources, and among sources themselves?)

## 2. Global Context Summary
(A unified summary of the event based on the combined evidence, independent of the user's claim)

## 3. Conflict Alert
* [Conflict]: (Critical discrepancies between sources, if any)
"""

# Agent 4: Q&A Generation Prompt (Iterative)
AGENT4_PROMPT1 = """# Role
You are the Lead Fact-Checking Adjudicator. Your task is to synthesize preliminary analyses into **decisive Question-Answer (QA) pairs** that consolidate the key evidence and reasoning required to form a final verdict.

# Input Data
## 1. The Claim (Target for Verification)
- **Claimant (Speaker):** {speaker}
- **Claim Date:** {date}
- **Claim Text:** {original_claim_text}
- **Claim Images:** """

AGENT4_PROMPT2 = """
## 2. Preliminary Analyses
{output_from_prompt_1}
{output_from_prompt_2}
{output_from_prompt_3}

## 3. Few-shot Learning Examples {few_shot_examples}


{previous_qa_section}


# Instructions
## Synthesize Diagnostic QAs (The Reasoning Basis)
Analyze the provided forensic reports to extract the **core information** necessary to predict the verdict. Formulate **{num_qa_to_generate} high-impact QA pairs** that:
- **Isolate Key Evidence:** Focus on dates, locations, inconsistencies, or manipulation traces that act as "smoking guns."
- **Resolve Ambiguity:** Ask and answer questions that clarify whether the evidence is sufficient or conflicting.
- **Serve as Proof:** Each QA must act as a logical premise supporting your final decision.

# Output Format (JSON Only)
```json
{{
    "qa_pairs": [
        {{"question": "<Question 1>", "answer": "<Full statement answer 1>"}},
        {{"question": "<Question 2>", "answer": "<Full statement answer 2>"}}
    ]
}}
```
"""

# Agent 4: Previous QA section template
AGENT4_PREVIOUS_QA = """
## 4. Previously Generated QA Pairs
*(Do NOT repeat these or ask similar questions)*
{previous_qa_list}
"""

# Agent 5: Verdict Generation Prompt
AGENT5_PROMPT1 = """# Role
You are the Lead Fact-Checking Adjudicator. Your task is to select the most relevant QA pairs, assess veracity, and provide a final verdict with justification.

# Input Data
## 1. The Claim (Target for Verification)
- **Claimant (Speaker):** {speaker}
- **Claim Date:** {date}
- **Claim Text:** {original_claim_text}
- **Claim Images:** """

AGENT5_PROMPT2 = """
## 2. Generated Question-Answer Pairs
{qa_pairs_text}
---
# Instructions
1. **Select Best QA Pairs:** From the generated QA pairs above, select the **{num_qa_to_select} most relevant and informative** pairs for verification.
2. **Determine Verdict:** Choose the single best label:
    - **Supported**
    - **Refuted**
    - **Not Enough Evidence**
    - **Conflicting Evidence/Cherrypicking**
3. **Write Justification:** A cohesive summary explaining the verdict based on the selected QA pairs.
4. **JSON Output:** Output **ONLY** a valid JSON object matching the format below.

# Output Format (JSON Only)
```json
{{
    "questions": [
        {{"question": "<Selected question 1>", "answer": "<Answer 1>"}},
        {{"question": "<Selected question 2>", "answer": "<Answer 2>"}}
    ],
    "veracity_verdict": "<String: Supported / Refuted / Not Enough Evidence / Conflicting Evidence/Cherrypicking>",
    "justification": "<String: A cohesive summary explaining the verdict>"
}}
```
"""

