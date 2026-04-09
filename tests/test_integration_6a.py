"""Autograder tests for Integration 6A — Entity Analysis Pipeline."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from entity_analysis import (load_and_filter_corpus, run_ner_pipeline,
                             aggregate_entity_stats,
                             visualize_entity_distribution, generate_report)


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "climate_articles.csv")

SAMPLE_NER_TEXT = (
    "The United Nations Environment Programme published a report on "
    "Jordan's climate adaptation in March 2024."
)


@pytest.fixture
def corpus():
    data = load_and_filter_corpus(DATA_PATH)
    assert data is not None, "load_and_filter_corpus returned None"
    return data


def test_corpus_loaded(corpus):
    """Corpus loads and contains only English texts."""
    assert len(corpus) > 0, "Corpus is empty"
    required_cols = {"id", "text", "source", "language", "category"}
    assert required_cols.issubset(set(corpus.columns)), (
        f"Missing columns: {required_cols - set(corpus.columns)}"
    )
    assert (corpus["language"] == "en").all(), (
        "Corpus should contain only English texts after filtering"
    )


def test_run_ner_pipeline_returns_dataframe():
    """run_ner_pipeline returns a DataFrame with required columns."""
    texts = [(1, SAMPLE_NER_TEXT)]
    result = run_ner_pipeline(texts)
    assert result is not None, "run_ner_pipeline returned None"
    assert isinstance(result, pd.DataFrame), "Must return a DataFrame"
    required_cols = {"text_id", "entity_text", "entity_label"}
    assert required_cols.issubset(set(result.columns)), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )
    assert len(result) > 0, "No entities extracted from text with known entities"


def test_aggregate_entity_stats():
    """aggregate_entity_stats returns frequency and co-occurrence data."""
    entity_df = pd.DataFrame({
        "text_id": [1, 1, 1, 2, 2],
        "entity_text": ["IPCC", "Jordan", "March 2024", "IPCC", "Dead Sea"],
        "entity_label": ["ORG", "GPE", "DATE", "ORG", "LOC"],
    })
    result = aggregate_entity_stats(entity_df)
    assert result is not None, "aggregate_entity_stats returned None"
    assert isinstance(result, dict), "Must return a dictionary"
    assert "top_entities" in result, "Missing 'top_entities' key"
    assert "label_counts" in result, "Missing 'label_counts' key"
    assert "co_occurrence" in result, "Missing 'co_occurrence' key"

    # top_entities should be a DataFrame
    top = result["top_entities"]
    assert isinstance(top, pd.DataFrame), "top_entities must be a DataFrame"
    assert len(top) > 0, "top_entities is empty"

    # label_counts should have counts
    lc = result["label_counts"]
    assert isinstance(lc, dict), "label_counts must be a dict"
    assert lc.get("ORG", 0) == 2, f"Expected ORG count 2, got {lc.get('ORG')}"

    # co_occurrence should be a DataFrame
    co = result["co_occurrence"]
    assert isinstance(co, pd.DataFrame), "co_occurrence must be a DataFrame"


def test_visualize_entity_distribution(tmp_path):
    """visualize_entity_distribution creates an output image file."""
    stats = {
        "top_entities": pd.DataFrame({
            "entity_text": ["IPCC", "Jordan", "UN"],
            "entity_label": ["ORG", "GPE", "ORG"],
            "count": [10, 8, 5],
        }),
        "label_counts": {"ORG": 15, "GPE": 8},
    }
    output_path = str(tmp_path / "test_chart.png")
    visualize_entity_distribution(stats, output_path=output_path)
    assert os.path.exists(output_path), f"Chart not saved to {output_path}"
    assert os.path.getsize(output_path) > 0, "Chart file is empty"


def test_generate_report():
    """generate_report returns a non-empty string report."""
    stats = {
        "top_entities": pd.DataFrame({
            "entity_text": ["IPCC", "Jordan", "UN", "COP28", "Dead Sea"],
            "entity_label": ["ORG", "GPE", "ORG", "EVENT", "LOC"],
            "count": [10, 8, 5, 4, 3],
        }),
        "label_counts": {"ORG": 15, "GPE": 8, "EVENT": 4, "LOC": 3},
    }
    co_occurrence = pd.DataFrame({
        "entity_a": ["IPCC", "Jordan"],
        "entity_b": ["COP28", "Dead Sea"],
        "co_count": [3, 2],
    })
    result = generate_report(stats, co_occurrence)
    assert result is not None, "generate_report returned None"
    assert isinstance(result, str), "Must return a string"
    assert len(result) > 50, "Report is too short"
