from schema import Answer

LOW_RETRIEVAL_CONFIDENCE = 0.08


class RAGPipeline:
    def __init__(self, retriever, llm_client, top_k=4):
        self.retriever = retriever
        self.llm = llm_client
        self.top_k = top_k

    def answer(self, question: str, history: list) -> Answer:
        standalone_query = self.llm.rewrite_query(history, question)
        retrieved = self.retriever.search(standalone_query, top_k=self.top_k)
        top_score = retrieved[0].score if retrieved else 0.0

        llm_out = self.llm.generate_answer(question, retrieved, history)

        escalate = bool(llm_out.get("escalate"))
        reason = llm_out.get("escalation_reason")

        if top_score < LOW_RETRIEVAL_CONFIDENCE:
            escalate = True
            reason = reason or "Nothing relevant enough found in the knowledge base."

        return Answer(
            answer_text=llm_out.get("answer", ""),
            confidence=llm_out.get("confidence", "low"),
            grounded=bool(llm_out.get("grounded", False)),
            escalate=escalate,
            escalation_reason=reason if escalate else None,
            sources=llm_out.get("sources", []),
            retrieval_top_score=top_score,
            retrieved_ids=[rc.chunk.id for rc in retrieved],
            raw_llm_output=llm_out,
        )
