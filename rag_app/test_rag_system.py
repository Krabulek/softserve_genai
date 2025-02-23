import pytest

import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric, ToxicityMetric, GEval
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models.base_model import DeepEvalBaseLLM

import os
from dotenv import load_dotenv


load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-lite-preview-02-05"


class TestGoogleGenerativeAI(DeepEvalBaseLLM):
    """Class to implement Google Generative AI for DeepEval"""
    
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Google Generative AI Model"


custom_model_gemini = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    convert_system_message_to_human=True,
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
    top_k=1,
    top_p=0.9,
    max_output_tokens=8192,
    verbose=True
)

vertexai_gemini = TestGoogleGenerativeAI(model=custom_model_gemini)


dataset = EvaluationDataset()

dataset.add_test_cases_from_csv_file(
    file_path="../data/evaluation_data.csv",
    input_col_name="prompt",
    actual_output_col_name="response",
    expected_output_col_name="ground_truth",
    context_col_name="context",
    context_col_delimiter= ";",
    retrieval_context_col_name="context",
    retrieval_context_col_delimiter= ";"
)

class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"
    
    
@pytest.mark.parametrize(
  "test_case",
  dataset
)
@pytest.mark.asyncio
def test_chat_model(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(
      threshold=0.7,
      model=vertexai_gemini,
      include_reason=True
    )
    bias_metric = FaithfulnessMetric(
       threshold=0.8,
        model=vertexai_gemini,
        include_reason=True
    )
    toxicity_metric = ToxicityMetric(
      threshold=0.5,
      model=vertexai_gemini,
      include_reason=True
    )
    correctness_metric = GEval(
        threshold=0.8,
        name="Correctness",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
            "Heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are not okay"
        ],
        evaluation_params=[
          LLMTestCaseParams.INPUT,
          LLMTestCaseParams.ACTUAL_OUTPUT,
          LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        model=vertexai_gemini
    )
    rouge_metric = RougeMetric(threshold=0.6) #custom metric created above
    assert_test(test_case, [
      bias_metric, 
      toxicity_metric,
      correctness_metric,
      answer_relevancy_metric,
      rouge_metric
    ])

@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")