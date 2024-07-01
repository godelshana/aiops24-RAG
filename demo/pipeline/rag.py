from typing import List
import qdrant_client
from llama_index.core.llms.llm import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
    ServiceContext
)
from llama_index.core.evaluation import (
    RelevancyEvaluator, 
    FaithfulnessEvaluator, 
    CorrectnessEvaluator
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import CompletionResponse

from custom.template import QA_TEMPLATE


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = await self._vector_store.aquery(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = self._vector_store.query(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores


async def generation_with_knowledge_retrieval(
    query_str: str,
    retriever: BaseRetriever,
    llm: LLM,
    embed_model: HuggingFaceEmbedding,
    qa_template: str = QA_TEMPLATE,
    reranker: BaseNodePostprocessor | None = None,
    debug: bool = False,
    progress=None,
) -> CompletionResponse:
    query_bundle = QueryBundle(query_str=query_str)
    node_with_scores = await retriever.aretrieve(query_bundle)
    if debug:
        print(f"retrieved:\n{node_with_scores}\n------")
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
    
    contexts = [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    context_str = "\n\n".join(contexts)
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)
    
    # Create a ServiceContext with the provided LLM
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    # Initialize evaluators
    relevancy_evaluator = RelevancyEvaluator(service_context=service_context)
    faithfulness_evaluator = FaithfulnessEvaluator(service_context=service_context)
    correctness_evaluator = CorrectnessEvaluator(service_context=service_context)

    # Perform evaluations
    relevancy_score = await relevancy_evaluator.aevaluate(
        query=query_str,
        response=ret.text,
        contexts=[node.node.text for node in node_with_scores]
    )
    
    faithfulness_score = await faithfulness_evaluator.aevaluate(
        query=query_str,
        response=ret.text,
        contexts=[node.node.text for node in node_with_scores]
    )
    
    correctness_score = await correctness_evaluator.aevaluate(
        query=query_str,
        response=ret.text,
        contexts=[node.node.text for node in node_with_scores]
    )

    eval_results = {
        "relevancy": relevancy_score,
        "faithfulness": faithfulness_score,
        "correctness": correctness_score
    }

    return ret, eval_results


