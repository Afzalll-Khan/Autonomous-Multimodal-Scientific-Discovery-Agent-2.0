"""
RAG Evaluator Orchestrator
Coordinates single-sample and batch evaluations with strict input validation,
isolated metric error handling, prominent individual metric reporting, and
faithfulness gating to prevent high aggregate scores from hiding hallucinations.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.evaluation.metrics import (
        RetrievalRelevanceMetric,
        AnswerFaithfulnessMetric,
        ContextRelevanceMetric,
        SourceAttributionMetric,
    )
except ModuleNotFoundError:
    # Fallback when running evaluator.py directly as a standalone script
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from src.evaluation.metrics import (
        RetrievalRelevanceMetric,
        AnswerFaithfulnessMetric,
        ContextRelevanceMetric,
        SourceAttributionMetric,
    )

class RAGEvaluator:
    """Orchestrates RAG pipeline evaluation suites with strict validation and safety gates."""

    def __init__(self, faithfulness_threshold: float = 0.5):
        """
        Args:
            faithfulness_threshold (float): Minimum faithfulness score required before
                an aggregate score is allowed to reflect high overall quality.
        """
        self.faithfulness_threshold = faithfulness_threshold
        self.metrics = {
            "retrieval_relevance": RetrievalRelevanceMetric(),
            "answer_faithfulness": AnswerFaithfulnessMetric(),
            "context_relevance": ContextRelevanceMetric(),
            "source_attribution": SourceAttributionMetric(),
        }

    def _validate_inputs(
        self,
        query: Any,
        retrieved_contexts: Any,
        answer: Any,
        sources: Optional[Any] = None,
    ) -> None:
        """
        Validates sample inputs strictly rather than silently accepting missing or invalid types.
        Raises ValueError or TypeError on invalid input.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Invalid input: 'query' must be a non-empty string.")

        if not isinstance(retrieved_contexts, list) or not all(
            isinstance(c, str) for c in retrieved_contexts
        ):
            raise TypeError(
                "Invalid input: 'retrieved_contexts' must be a list of strings."
            )

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Invalid input: 'answer' must be a non-empty string.")

        if sources is not None:
            if not isinstance(sources, list) or not all(
                isinstance(s, dict) for s in sources
            ):
                raise TypeError(
                    "Invalid input: 'sources' must be a list of metadata dictionaries if provided."
                )

    def evaluate_sample(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single RAG generation sample across all core evaluation dimensions.

        Returns:
            Dict[str, Any]: Prominent top-level scores, detailed metric diagnostics,
                            validation flags, and a faithfulness-gated aggregate score.
        """
        self._validate_inputs(
            query=query,
            retrieved_contexts=retrieved_contexts,
            answer=answer,
            sources=sources,
        )

        metric_scores: Dict[str, Optional[float]] = {}
        metric_details: Dict[str, Dict[str, Any]] = {}
        valid_scores: List[float] = []
        failed_metrics: List[str] = []

        for name, metric in self.metrics.items():
            try:
                res = metric.evaluate(
                    query=query,
                    retrieved_contexts=retrieved_contexts,
                    answer=answer,
                    sources=sources,
                    ground_truth=ground_truth,
                )
                score = float(res["score"])
                metric_scores[name] = round(score, 4)
                metric_details[name] = res
                valid_scores.append(score)
            except Exception as exc:
                metric_scores[name] = None
                metric_details[name] = {
                    "score": None,
                    "error": f"{type(exc).__name__}: {str(exc)}",
                    "reason": "Metric execution failed; skipped from aggregate calculation.",
                }
                failed_metrics.append(name)

        raw_aggregate = (
            round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0
        )

        # Faithfulness guardrail: prevent a strong aggregate score from masking poor faithfulness
        faithfulness_score = metric_scores.get("answer_faithfulness")
        faithfulness_warning = False

        if faithfulness_score is None:
            # If faithfulness failed to compute, cap aggregate to signal unverified grounding
            gated_aggregate = min(raw_aggregate, self.faithfulness_threshold)
            faithfulness_warning = True
        elif faithfulness_score < self.faithfulness_threshold:
            # Cap aggregate score so it cannot exceed the poor faithfulness score
            gated_aggregate = min(raw_aggregate, faithfulness_score)
            faithfulness_warning = True
        else:
            gated_aggregate = raw_aggregate

        return {
            # Prominent individual metric scores at the top level
            "scores": metric_scores,
            "aggregate_score": round(gated_aggregate, 4),
            "unadjusted_aggregate_score": raw_aggregate,
            "faithfulness_warning": faithfulness_warning,
            "failed_metrics": failed_metrics,
            "details": metric_details,
        }

    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a batch of RAG samples and computes macro averages across the test set.
        """
        if not isinstance(samples, list) or not samples:
            raise ValueError("Invalid input: 'samples' must be a non-empty list of sample dicts.")

        batch_results: List[Dict[str, Any]] = []
        metric_sums: Dict[str, float] = {name: 0.0 for name in self.metrics}
        metric_counts: Dict[str, int] = {name: 0 for name in self.metrics}
        aggregate_sum = 0.0
        flagged_samples = 0

        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(f"Sample at index {idx} must be a dictionary.")

            missing_keys = [
                k for k in ("query", "retrieved_contexts", "answer") if k not in sample
            ]
            if missing_keys:
                raise KeyError(
                    f"Sample at index {idx} is missing required keys: {missing_keys}"
                )

            res = self.evaluate_sample(
                query=sample["query"],
                retrieved_contexts=sample["retrieved_contexts"],
                answer=sample["answer"],
                sources=sample.get("sources"),
                ground_truth=sample.get("ground_truth"),
            )
            batch_results.append(res)
            aggregate_sum += res["aggregate_score"]

            if res["faithfulness_warning"]:
                flagged_samples += 1

            for name in self.metrics:
                score = res["scores"].get(name)
                if score is not None:
                    metric_sums[name] += score
                    metric_counts[name] += 1

        n = len(samples)
        metric_averages = {
            name: (
                round(metric_sums[name] / metric_counts[name], 4)
                if metric_counts[name] > 0
                else None
            )
            for name in self.metrics
        }

        return {
            "total_samples": n,
            "metric_averages": metric_averages,
            "average_aggregate_score": round(aggregate_sum / n, 4),
            "faithfulness_flagged_samples": flagged_samples,
            "detailed_results": batch_results,
        }
