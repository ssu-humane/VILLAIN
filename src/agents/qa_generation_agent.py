"""
Agent 4: QA Generation Agent

Iteratively generates Question-Answer pairs for fact-checking.
Generates in batches (configurable) with context of previously generated pairs.
"""

import os
import json
import re
from typing import List
from dataclasses import dataclass, field

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from .base_agent import BaseAgent, AgentConfig, AgentAnalysis, EvidenceItem, SharedModels
from .prompts import AGENT4_PROMPT1, AGENT4_PROMPT2, AGENT4_PREVIOUS_QA


@dataclass
class QAPair:
    """Question-Answer pair for fact-checking."""
    question: str
    answer: str
    source_evidence: List[str] = field(default_factory=list)


@dataclass 
class QAGenerationResult:
    """Result from QA generation agent."""
    qa_pairs: List[QAPair] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    raw_responses: List[str] = field(default_factory=list)


class QAGenerationAgent(BaseAgent):
    """Agent for iteratively generating Q&A pairs.

    Generates Q&A pairs in batches, with each batch seeing previously
    generated pairs to avoid duplicates and ensure diversity.
    """

    def __init__(
        self,
        config: AgentConfig,
        shared_models: SharedModels = None,
        qa_per_iteration: int = 4,
        max_iterations: int = 5,
        max_qa_pairs: int = 20,
        train_data_path: str = None
    ):
        super().__init__(config, shared_models)
        self.qa_per_iteration = qa_per_iteration
        self.max_iterations = max_iterations
        self.max_qa_pairs = max_qa_pairs
        self.train_data_path = train_data_path or "dataset/AVerImaTeC/train.json"
        self._bm25 = None
        self._train_data = None

    @property
    def name(self) -> str:
        return "qa_generation_agent"

    def _load_train_data(self):
        """Load training data for few-shot examples."""
        if self._train_data is None:
            if os.path.exists(self.train_data_path):
                with open(self.train_data_path, 'r') as f:
                    self._train_data = json.load(f)
            else:
                self._train_data = []
        return self._train_data

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using nltk if available, else simple split."""
        if NLTK_AVAILABLE:
            try:
                return nltk.word_tokenize(text)
            except:
                pass
        return text.lower().split()

    def _init_bm25(self):
        """Initialize BM25 for few-shot retrieval."""
        if not BM25_AVAILABLE:
            return None
        if self._bm25 is None:
            train_data = self._load_train_data()
            if train_data:
                self._valid_indices = []
                tokenized_corpus = []
                for i, item in enumerate(train_data):
                    text = item.get('claim_text', '')
                    if text and text.strip():
                        tokenized_corpus.append(self._tokenize(text))
                        self._valid_indices.append(i)
                if tokenized_corpus:
                    self._bm25 = BM25Okapi(tokenized_corpus)
        return self._bm25

    def _get_few_shot_examples(self, claim_text: str, num_examples: int = 3) -> str:
        """Get few-shot examples using BM25 retrieval."""
        bm25 = self._init_bm25()
        train_data = self._load_train_data()

        if not bm25 or not train_data or not hasattr(self, '_valid_indices'):
            return ""

        tokenized_query = self._tokenize(claim_text)
        scores = bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-num_examples:][::-1]

        few_shot_prompt = ""
        for f_idx, idx in enumerate(top_indices):
            original_idx = self._valid_indices[idx]
            fs_example = train_data[original_idx]

            output_json = {"qa_pairs": []}
            for q in fs_example.get("questions", []):
                ques_text = q.get("question", "")
                answers = q.get("answers", [])
                if not answers:
                    answers = [{"answer": "No answer could be found.", "answer_type": "Unanswerable"}]

                final_answers = []
                for a in answers:
                    answer_type = a.get("answer_type", "Extractive")
                    if answer_type == "Image":
                        imgs = [i.split("#")[-1] for i in a.get('image_answers', [])]
                        final_answers.append(f"[{', '.join(imgs)}]")
                    else:
                        text_ans = a.get("answer_text", a.get("answer", ""))
                        if answer_type == "Boolean":
                            bool_exp = a.get("boolean_explanation", "")
                            if bool_exp:
                                text_ans = f"{text_ans}. {bool_exp}"
                        if text_ans:
                            final_answers.append(text_ans)

                if ques_text:
                    output_json["qa_pairs"].append({
                        "question": ques_text,
                        "answer": " ".join(final_answers) if final_answers else "No answer could be found."
                    })

            json_str = json.dumps(output_json, indent=4)
            few_shot_prompt += f'\n\n### Example {f_idx+1}'
            few_shot_prompt += f'\n**Claim:** "{fs_example.get("claim_text", "")}"'
            few_shot_prompt += f'\n**Output:**\n```json\n{json_str}\n```'

        return few_shot_prompt

    def _format_source_list(self, evidence_items: List[EvidenceItem], prefix: str = "") -> str:
        """Format evidence items as source list for prompt."""
        lines = []
        for i, item in enumerate(evidence_items, 1):
            url = item.url or item.image_path or "Unknown"
            lines.append(f"- Source {prefix}{i}: {url}")
        return "\n".join(lines) if lines else "No sources available."

    def _format_previous_qa(self, qa_pairs: List[QAPair]) -> str:
        """Format previously generated QA pairs for context."""
        if not qa_pairs:
            return ""

        qa_list = []
        for i, qa in enumerate(qa_pairs, 1):
            qa_list.append(f"{i}. Q: {qa.question}\n   A: {qa.answer}")

        return AGENT4_PREVIOUS_QA.format(previous_qa_list="\n".join(qa_list))

    def _parse_qa_response(self, response_text: str) -> List[QAPair]:
        """Parse JSON response to extract QA pairs."""
        if '</think>' in response_text:
            response_text = response_text.split('</think>')[-1].strip()

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return []

            data = json.loads(json_str)
            qa_pairs = []
            # Check both "qa_pairs" (new format) and "questions" (old format)
            qa_list = data.get("qa_pairs", data.get("questions", []))
            for qa in qa_list:
                q = qa.get("question", "")
                a = qa.get("answer", "")
                if q and a:
                    qa_pairs.append(QAPair(question=q, answer=a))
            return qa_pairs

        except (json.JSONDecodeError, Exception) as e:
            print(f"[{self.name}] Parse error: {e}")
            return []

    def generate_qa_pairs(
        self,
        claim_text: str,
        claim_images: List[str],
        text_text_analysis: AgentAnalysis,
        image_text_analysis: AgentAnalysis,
        image_image_analysis: AgentAnalysis,
        speaker: str = "Unknown",
        date: str = "Not Specified"
    ) -> QAGenerationResult:
        """Iteratively generate Q&A pairs."""
        if not self.shared_models:
            return QAGenerationResult()

        result = QAGenerationResult()
        all_qa_pairs: List[QAPair] = []

        # Get few-shot examples once
        few_shot_examples = self._get_few_shot_examples(claim_text)

        # Get valid claim images once
        valid_images = self.get_valid_claim_images(claim_images)

        for iteration in range(self.max_iterations):
            if len(all_qa_pairs) >= self.max_qa_pairs:
                break

            remaining = self.max_qa_pairs - len(all_qa_pairs)
            num_to_generate = min(self.qa_per_iteration, remaining)

            print(f"[{self.name}] Iteration {iteration + 1}/{self.max_iterations}: "
                  f"Generating {num_to_generate} QA pairs (have {len(all_qa_pairs)})")

            # Build prompt with previous QA context
            previous_qa_section = self._format_previous_qa(all_qa_pairs)

            prompt_part1 = AGENT4_PROMPT1.format(
                speaker=speaker,
                date=date,
                original_claim_text=claim_text
            )

            prompt_part2 = AGENT4_PROMPT2.format(
                output_from_prompt_1=text_text_analysis.analysis_text,
                output_from_prompt_2=image_text_analysis.analysis_text,
                output_from_prompt_3=image_image_analysis.analysis_text,
                few_shot_examples=few_shot_examples,
                previous_qa_section=previous_qa_section,
                num_qa_to_generate=num_to_generate
            )

            # Build message content
            content = [{"type": "text", "text": prompt_part1}]
            for img in valid_images:
                img_path = self.get_full_image_path(img)
                if os.path.exists(img_path):
                    content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": prompt_part2})

            messages = [{"role": "user", "content": content}]

            try:
                output_text = self.shared_models.generate_with_vlm(messages)
                result.raw_responses.append(output_text)

                new_qa_pairs = self._parse_qa_response(output_text)
                all_qa_pairs.extend(new_qa_pairs)
                print(f"[{self.name}] Generated {len(new_qa_pairs)} new QA pairs")

            except Exception as e:
                print(f"[{self.name}] Error in iteration {iteration + 1}: {e}")
                continue

        # Populate result
        result.qa_pairs = all_qa_pairs[:self.max_qa_pairs]
        result.questions = [qa.question for qa in result.qa_pairs]
        result.answers = [qa.answer for qa in result.qa_pairs]

        print(f"[{self.name}] Total QA pairs generated: {len(result.qa_pairs)}")
        return result

    def analyze(
        self,
        claim_id: int,
        claim_text: str,
        claim_images: List[str],
        text_text_analysis: AgentAnalysis = None,
        image_text_analysis: AgentAnalysis = None,
        image_image_analysis: AgentAnalysis = None,
        speaker: str = "Unknown",
        date: str = "Not Specified",
        **kwargs
    ) -> AgentAnalysis:
        """Main analysis method - generates Q&A pairs iteratively."""
        if not all([text_text_analysis, image_text_analysis, image_image_analysis]):
            return AgentAnalysis(
                agent_name=self.name,
                analysis_text="Missing required agent analyses."
            )

        qa_result = self.generate_qa_pairs(
            claim_text=claim_text,
            claim_images=claim_images,
            text_text_analysis=text_text_analysis,
            image_text_analysis=image_text_analysis,
            image_image_analysis=image_image_analysis,
            speaker=speaker,
            date=date
        )

        # Combine all evidence
        all_evidence = []
        all_evidence.extend(text_text_analysis.evidence_items)
        all_evidence.extend(image_text_analysis.evidence_items)
        all_evidence.extend(image_image_analysis.evidence_items)

        return AgentAnalysis(
            agent_name=self.name,
            evidence_items=all_evidence,
            questions=qa_result.questions,
            answers=qa_result.answers,
            analysis_text=f"Generated {len(qa_result.qa_pairs)} QA pairs for verification.",
            metadata={
                'qa_pairs': [
                    {'question': qa.question, 'answer': qa.answer}
                    for qa in qa_result.qa_pairs
                ],
                'num_qa_pairs': len(qa_result.qa_pairs),
                'raw_responses': qa_result.raw_responses
            }
        )

