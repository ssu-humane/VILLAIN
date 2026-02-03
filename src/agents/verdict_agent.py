"""
Agent 5: Verdict Agent

Selects the best Q&A pairs and generates the final verdict with justification.
"""

import os
import json
import re
from typing import List, Dict
from dataclasses import dataclass, field

from .base_agent import BaseAgent, AgentConfig, AgentAnalysis, SharedModels
from .qa_generation_agent import QAPair
from .prompts import AGENT5_PROMPT1, AGENT5_PROMPT2


@dataclass
class VerdictResult:
    """Result from the verdict agent."""
    selected_qa_pairs: List[QAPair] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    veracity_verdict: str = ""
    justification: str = ""
    raw_response: str = ""


class VerdictAgent(BaseAgent):
    """Agent for selecting QA pairs and generating final verdict."""

    def __init__(
        self,
        config: AgentConfig,
        shared_models: SharedModels = None,
        num_qa_to_select: int = 10
    ):
        super().__init__(config, shared_models)
        self.num_qa_to_select = num_qa_to_select

    @property
    def name(self) -> str:
        return "verdict_agent"

    def _format_qa_pairs_for_prompt(self, qa_pairs: List[QAPair]) -> str:
        """Format QA pairs for the verdict prompt."""
        lines = []
        for i, qa in enumerate(qa_pairs, 1):
            lines.append(f"{i}. **Q:** {qa.question}")
            lines.append(f"   **A:** {qa.answer}")
        return "\n".join(lines)

    def _parse_verdict_response(self, response_text: str) -> VerdictResult:
        """Parse JSON response from VLM."""
        result = VerdictResult(raw_response=response_text)

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
                    return result

            data = json.loads(json_str)

            for qa in data.get("questions", []):
                q = qa.get("question", "")
                a = qa.get("answer", "")
                if q and a:
                    result.selected_qa_pairs.append(QAPair(question=q, answer=a))
                    result.questions.append(q)
                    result.answers.append(a)

            result.veracity_verdict = data.get("veracity_verdict", "Not Enough Evidence")
            result.justification = data.get("justification", "")

        except (json.JSONDecodeError, Exception) as e:
            print(f"[{self.name}] Parse error: {e}")

        return result

    def _create_fallback_result(self, claim_text: str, qa_pairs: List[QAPair]) -> VerdictResult:
        """Create fallback result when parsing fails."""
        default_question = f"Is the following claim true: {claim_text[:200]}...?" if len(claim_text) > 200 else f"Is the following claim true: {claim_text}?"
        default_answer = "Unable to verify the claim with available evidence."

        # Use provided QA pairs if available
        selected = qa_pairs[:self.num_qa_to_select] if qa_pairs else [
            QAPair(question=default_question, answer=default_answer)
        ]

        return VerdictResult(
            selected_qa_pairs=selected,
            questions=[qa.question for qa in selected],
            answers=[qa.answer for qa in selected],
            veracity_verdict="Not Enough Evidence",
            justification="Unable to generate verdict. Defaulting to 'Not Enough Evidence'."
        )

    def generate_verdict(
        self,
        claim_text: str,
        claim_images: List[str],
        qa_pairs: List[QAPair],
        speaker: str = "Unknown",
        date: str = "Not Specified"
    ) -> VerdictResult:
        """Generate verdict from QA pairs."""
        if not self.shared_models:
            return self._create_fallback_result(claim_text, qa_pairs)

        qa_pairs_text = self._format_qa_pairs_for_prompt(qa_pairs)

        prompt_part1 = AGENT5_PROMPT1.format(
            speaker=speaker,
            date=date,
            original_claim_text=claim_text
        )

        prompt_part2 = AGENT5_PROMPT2.format(
            qa_pairs_text=qa_pairs_text,
            num_qa_to_select=self.num_qa_to_select
        )

        content = [{"type": "text", "text": prompt_part1}]
        valid_images = self.get_valid_claim_images(claim_images)
        for img in valid_images:
            img_path = self.get_full_image_path(img)
            if os.path.exists(img_path):
                content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": prompt_part2})

        messages = [{"role": "user", "content": content}]

        try:
            output_text = self.shared_models.generate_with_vlm(messages)
            result = self._parse_verdict_response(output_text)

            if not result.selected_qa_pairs:
                result = self._create_fallback_result(claim_text, qa_pairs)

            return result

        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_result(claim_text, qa_pairs)

    def analyze(
        self,
        claim_id: int,
        claim_text: str,
        claim_images: List[str],
        qa_pairs: List[QAPair] = None,
        speaker: str = "Unknown",
        date: str = "Not Specified",
        **kwargs
    ) -> AgentAnalysis:
        """Main analysis method - generates verdict from QA pairs."""
        if qa_pairs is None:
            qa_pairs = []

        verdict_result = self.generate_verdict(
            claim_text=claim_text,
            claim_images=claim_images,
            qa_pairs=qa_pairs,
            speaker=speaker,
            date=date
        )

        return AgentAnalysis(
            agent_name=self.name,
            questions=verdict_result.questions,
            answers=verdict_result.answers,
            analysis_text=verdict_result.justification,
            metadata={
                'qa_pairs': [
                    {'question': qa.question, 'answer': qa.answer}
                    for qa in verdict_result.selected_qa_pairs
                ],
                'num_qa_pairs': len(verdict_result.selected_qa_pairs),
                'veracity_verdict': verdict_result.veracity_verdict,
                'justification': verdict_result.justification,
                'raw_response': verdict_result.raw_response
            }
        )

