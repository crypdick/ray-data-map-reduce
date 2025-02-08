"""
Examples of using Ray Data to perform a map-reduce job with AggregateFn.

This file demonstrates:
1. Parsing text lines to extract bigrams (pairs of consecutive words).
2. Using built-in aggregator classes (e.g., Sum) for a 'reduce' step.
3. Reducing using a custom aggregator (CollectList)
"""
import ray
import re
from ray.data.aggregate import AggregateFn
from ray.data._internal.aggregate import Sum


def make_list_aggregator(input_key: str, output_key: str = "items"):
    """
    Creates an aggregator that collects grouped values
    into a single (merged) list.

    For example, if rows grouped by {'count': 1} are:
      [{"bigram": "the cat"}, {"bigram": "the man"}],
    the final accumulated value would be:
      ["the cat", "the man"]
    """
    return AggregateFn(
            # Initialize an empty list "accumulator" for each group.
            init=lambda _key: [],
            # Merge two accumulator lists.
            merge=lambda accum1, accum2: accum1 + accum2,
            # The key to store the resulting accumulator under
            name=output_key,
            # Append the relevant field (e.g. row[input_key]) to the accumulator list.
            accumulate_row=lambda acc, row: acc + [row[input_key]],
            # we don't need to do anything with the final accumulated value,
            # so we just return it as-is
            finalize=lambda acc: acc,
        )


def extract_bigrams(item: dict):
    """
    Extract bigrams from text.

    Args:
        item: Dictionary containing text under the 'item' key
        
    Yields:
        dict: Records of the form {"bigram": "word1 word2", "count": 1}
    """
    line = item["item"]
    tokens = re.findall(r"\w+", line.lower())
    for i in range(len(tokens) - 1):
        # gotcha: returning a tuple(w1, w2) will not work, because Ray Data will convert this
        # to an array. Then, the groupby will fail with an error like:
        # `The truth value of an array with more than one element is ambiguous.`
        yield {"bigram": f"{tokens[i]} {tokens[i + 1]}", "count": 1}


def main():
    """Run the bigram analysis pipeline."""
    ray.init()

    # Toy text corpus
    text_lines = [
        "the cat sat on the mat every day",
        "the cat ate a mouse every day",
        "the cat and the man became friends",
        "I like to eat pizza, but so does the cat.",
        "my cat has a meme coin named after him",
        "I eat pizza every day",
    ]

    # Create a dataset of lines.
    ds = ray.data.from_items(text_lines)

    # "Map" step: for each line, produce bigram-count pairs of the form {"bigram": "w1 w2", "count": 1}.
    ds = ds.flat_map(extract_bigrams)

    # "Reduce" step by bigram, summing counts.
    # This deduplicates all identical bigrams so we get {"item": "w1 w2", "count": total_count}.
    # Note: Sum renames the "bigram" key to "item"
    ds = ds.groupby("bigram").aggregate(Sum("count", alias_name="count"))

    # "Reduce" again by "count", but now we use our custom aggregator to collect all bigrams with that count
    # into a single list, e.g. {"count": 1, "bigrams": [("meme", "coin"), ("coin", "named"), ...]}
    # Note: the {"count": count_val, ...} part is left unchanged
    ds = ds.groupby("count").aggregate(make_list_aggregator(input_key="bigram", output_key="bigrams"))

    # Finally, map each list of bigrams to just the length of that list,
    # e.g. giving us {count -> how many bigrams have that count?}
    ds = ds.map(
        lambda row: {
            "count": row["count"],
            "num_bigrams": len(row["bigrams"]),
        }
    )
    results = ds.take_all()  # finally, execute the pipeline

    ## Polished version
    # results = (
    #     ray.data.from_items(text_lines)
    #     .flat_map(extract_bigrams)
    #     .groupby("bigram")
    #     .aggregate(Sum("count", alias_name="count"))
    #     .groupby("count")
    #     .aggregate(make_list_aggregator(input_key="bigram", output_key="bigrams"))
    #     .map(lambda row: {
    #         "count": row["count"],
    #         "num_bigrams": len(row["bigrams"]),
    #     })
    #     .take_all()
    # )

    # Show final result.
    # For example, you might get something like: 
    # [{'count': 1, 'num_bigrams': 31}, {'count': 2, 'num_bigrams': 1}, {'count': 3, 'num_bigrams': 1}, {'count': 4, 'num_bigrams': 1}]
    print("Histogram of how many bigrams share each count:")
    print(results)


if __name__ == "__main__":
    main()