def compute_query_RR_at_10(retrieved_docs, relevant_docs):
    """
    Compute the Mean Reciprocal Rank (MRR) for a single query.

    retrieved_docs: list of retrieved documents
    relevant_docs: list of relevant documents
    """
    assert len(retrieved_docs) == 10
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)
    return 0
