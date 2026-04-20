"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Build a corpus-level entity analysis pipeline that processes climate
articles, extracts entities, computes statistics, and produces
visualizations.

Run: python entity_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy


def load_and_filter_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset and filter to English texts.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame containing only English-language rows, with columns:
        id, text, source, language, category.
    """
    # TODO: Load the CSV, filter rows where language == 'en',
    #       return the filtered DataFrame
    df = pd.read_csv(filepath)
    return df[df["language"] == "en"].reset_index(drop=True)


def run_ner_pipeline(texts):
    """Run Named Entity Recognition on a list of texts using spaCy.

    Args:
        texts: List of (text_id, text_string) tuples.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label.
    """
    # TODO: Load the spaCy model, process each text,
    #       extract entities into a DataFrame
    nlp = spacy.load("en_core_web_sm")
    records = []
    for text_id, text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            records.append({"text_id": text_id, "entity_text": ent.text, "entity_label": ent.label_})
    return pd.DataFrame(records, columns=["text_id", "entity_text", "entity_label"])


def aggregate_entity_stats(entity_df):
    """Compute entity frequency and co-occurrence statistics.

    Args:
        entity_df: DataFrame with columns text_id, entity_text,
                   entity_label.

    Returns:
        Dictionary with keys:
          'top_entities': DataFrame of top 20 entities by frequency
                         (columns: entity_text, entity_label, count)
          'label_counts': dict of entity_label → total count
          'co_occurrence': DataFrame of entity pairs that appear in
                          the same text (columns: entity_a, entity_b,
                          co_count)
    """
    # TODO: Count entity frequencies, find top 20, compute label
    #       totals, and build co-occurrence pairs
    freq = (
        entity_df.groupby(["entity_text", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top_entities = freq.head(20).reset_index(drop=True)

    label_counts = entity_df.groupby("entity_label").size().to_dict()

    # Build co-occurrence: pairs of distinct entities in the same text
    co_pairs = []
    for _, group in entity_df.groupby("text_id"):
        unique_entities = group["entity_text"].unique().tolist()
        for i in range(len(unique_entities)):
            for j in range(i + 1, len(unique_entities)):
                a, b = sorted([unique_entities[i], unique_entities[j]])
                co_pairs.append((a, b))

    if co_pairs:
        co_df = (
            pd.DataFrame(co_pairs, columns=["entity_a", "entity_b"])
            .groupby(["entity_a", "entity_b"])
            .size()
            .reset_index(name="co_count")
            .sort_values("co_count", ascending=False)
            .reset_index(drop=True)
        )
    else:
        co_df = pd.DataFrame(columns=["entity_a", "entity_b", "co_count"])

    return {"top_entities": top_entities, "label_counts": label_counts, "co_occurrence": co_df}


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create a bar chart of the top 20 entities by frequency.

    Args:
        stats: Dictionary from aggregate_entity_stats (must contain
               'top_entities' DataFrame).
        output_path: File path to save the chart.
    """
    # TODO: Create a horizontal bar chart of top entities, labeled
    #       by entity type, save to output_path
    top = stats["top_entities"]
    labels = top["entity_text"] + " (" + top["entity_label"] + ")"
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels[::-1], top["count"][::-1])
    ax.set_xlabel("Frequency")
    ax.set_title("Top 20 Entities by Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def generate_report(stats, co_occurrence):
    """Generate a text summary of entity analysis findings.

    Args:
        stats: Dictionary from aggregate_entity_stats.
        co_occurrence: Co-occurrence DataFrame from stats.

    Returns:
        String containing a structured report with: entity counts
        per type, top 5 most frequent entities, top 3 co-occurring
        pairs, and a brief summary.
    """
    # TODO: Build a formatted report string from the statistics
    lines = ["=== Entity Analysis Report ===", ""]

    lines.append("Entity Counts per Type:")
    for label, count in sorted(stats["label_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"  {label}: {count}")

    lines.append("")
    lines.append("Top 5 Most Frequent Entities:")
    for _, row in stats["top_entities"].head(5).iterrows():
        lines.append(f"  {row['entity_text']} ({row['entity_label']}): {row['count']}")

    lines.append("")
    lines.append("Top 3 Co-occurring Pairs:")
    if co_occurrence is not None and len(co_occurrence) > 0:
        for _, row in co_occurrence.head(3).iterrows():
            lines.append(f"  {row['entity_a']} & {row['entity_b']}: {row['co_count']}")
    else:
        lines.append("  No co-occurrence data available.")

    lines.append("")
    total_entities = sum(stats["label_counts"].values())
    num_types = len(stats["label_counts"])
    lines.append(
        f"Summary: {total_entities} total entities across {num_types} types. "
        f"Most common type: {max(stats['label_counts'], key=stats['label_counts'].get)}."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    # Load and filter corpus
    corpus = load_and_filter_corpus()
    if corpus is not None:
        print(f"Corpus: {len(corpus)} English articles")
        print(f"Categories: {corpus['category'].value_counts().to_dict()}")

        # Run NER
        texts = list(zip(corpus["id"], corpus["text"]))
        entities = run_ner_pipeline(texts)
        if entities is not None:
            print(f"\nExtracted {len(entities)} entities")

            # Aggregate statistics
            stats = aggregate_entity_stats(entities)
            if stats is not None:
                print(f"\nLabel counts: {stats['label_counts']}")
                print(f"\nTop 5 entities:")
                print(stats["top_entities"].head())

                # Visualize
                visualize_entity_distribution(stats)
                print("\nVisualization saved to entity_distribution.png")

                # Generate report
                report = generate_report(stats, stats.get("co_occurrence"))
                if report is not None:
                    print(f"\n{'='*50}")
                    print(report)
