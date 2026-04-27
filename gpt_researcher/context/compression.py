"""Context compression utilities for GPT Researcher.

This module provides classes for compressing and retrieving relevant
context from documents using embeddings and similarity filtering.

The compression pipeline:
1. Splits documents into chunks
2. Filters chunks by embedding similarity to the query
3. Returns the most relevant chunks as context

Classes:
    VectorstoreCompressor: Retrieves context from a vector store.
    ContextCompressor: Compresses raw documents using embedding similarity.
    WrittenContentCompressor: Compresses previously written content sections.
"""

import asyncio
import os
from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..memory.embeddings import OPENAI_EMBEDDING_MODEL
from ..prompts import PromptFamily
from ..utils.costs import estimate_embedding_cost, estimate_tokens
from ..vector_store import VectorStoreWrapper
from .retriever import SearchAPIRetriever, SectionRetriever


MAX_EMBEDDING_TOKENS = 7500  # max tokens for embeddings, usually around 8000 but leaving buffer for metadata and estimations


class VectorstoreCompressor:
    """Retrieves and compresses context from a vector store.

    Uses similarity search on an existing vector store to find
    relevant documents for a given query.

    Attributes:
        vector_store: The vector store wrapper to search.
        max_results: Maximum number of results to return.
        filter: Optional filter for vector store queries.
    """

    def __init__(
        self,
        vector_store: VectorStoreWrapper,
        max_results: int = 7,
        filter: Optional[dict] = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the VectorstoreCompressor.

        Args:
            vector_store: The vector store to search.
            max_results: Maximum number of results to return.
            filter: Optional filter dictionary for queries.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.vector_store = vector_store
        self.max_results = max_results
        self.filter = filter
        self.kwargs = kwargs
        self.prompt_family = prompt_family

    async def async_get_context(self, query: str, max_results: int = 5) -> str:
        """Get relevant context from the vector store.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted string of relevant document content.
        """
        results = await self.vector_store.asimilarity_search(query=query, k=max_results, filter=self.filter)
        return self.prompt_family.pretty_print_docs(results)


class IndividualEmbeddingFilter:
    """Filters documents by embedding similarity using batched embeddings.

    Groups documents into token-aware batches to minimize API calls while
    staying within embedding model token limits.
    """

    def __init__(self, embeddings, similarity_threshold: float):
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    def _create_batches(
        self, documents: list[Document], max_tokens: int = MAX_EMBEDDING_TOKENS
    ) -> list[list[Document]]:
        """Group documents into batches that fit within token limit.

        Only splits documents that individually exceed the token limit.
        All other documents are kept intact and batched by token count.
        """
        # First, split only oversized documents into token-safe chunks
        flat_docs = []
        for doc in documents:
            doc_tokens = estimate_tokens(doc.page_content)
            if doc_tokens > max_tokens:
                flat_docs.extend(self._split_oversized(doc, max_tokens))
            else:
                flat_docs.append(doc)

        # Then batch by token count
        batches = []
        current_batch = []
        current_tokens = 0

        for doc in flat_docs:
            doc_tokens = estimate_tokens(doc.page_content)
            if doc_tokens > max_tokens:
                continue
            if current_tokens + doc_tokens > max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(doc)
            current_tokens += doc_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _split_oversized(self, doc: Document, max_tokens: int) -> list[Document]:
        """Split an oversized document into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents([doc])

    async def acompress_documents(
        self, documents: list[Document], query: str, fallback_top_k: int = 10
    ) -> list[Document]:
        """Filter documents by similarity using batched embeddings.

        Embeds documents in token-aware batches to minimize API calls.
        All documents get similarity scores. If none pass the threshold,
        the top ``fallback_top_k`` by score are returned as a fallback.

        Args:
            documents: List of documents to filter.
            query: Query to compare document relevance against.
            fallback_top_k: Number of top-scoring docs to return
                when threshold filtering returns nothing.

        Returns:
            List of documents sorted by similarity (descending).
        """
        batches = self._create_batches(documents)
        query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
        doc_embeddings: list[tuple[Document, list[float]]] = []

        for batch in batches:
            try:
                texts = [doc.page_content for doc in batch]
                embeddings = await asyncio.to_thread(
                    self.embeddings.embed_documents, texts
                )
                for i, doc in enumerate(batch):
                    doc_embeddings.append((doc, embeddings[i]))
            except Exception:
                continue

        scored_docs: list[tuple[Document, float]] = []
        for doc, embedding in doc_embeddings:
            similarity = self._cosine_similarity(query_embedding, embedding)
            scored_docs.append((doc, similarity))

        # Sort all docs by similarity descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Threshold-filtered docs
        filtered = [
            (doc, score)
            for doc, score in scored_docs
            if score >= self.similarity_threshold
        ]

        if filtered:
            for doc, score in filtered:
                doc.metadata["similarity_score"] = score
            return [doc for doc, _ in filtered]
        else:
            # Fallback: return top-k regardless of threshold
            fallback = scored_docs[:fallback_top_k]
            for doc, score in fallback:
                doc.metadata["similarity_score"] = score
            return [doc for doc, _ in fallback]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        return dot / (mag1 * mag2 + 1e-8)


class ContextCompressor:
    """Compresses raw documents to extract relevant context.

    Uses embedding similarity to filter document chunks and return
    only the most relevant content for a given query.

    Attributes:
        documents: List of documents to compress.
        embeddings: Embedding model for similarity calculation.
        max_results: Maximum number of results to return.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(
        self,
        documents,
        embeddings,
        max_results: int = 5,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the ContextCompressor.

        Args:
            documents: List of documents to compress.
            embeddings: Embedding model instance.
            max_results: Maximum number of results to return.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.35))
        self.prompt_family = prompt_family

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
        """Get relevant context from documents asynchronously.

        Optimization: Skip expensive compression pipeline for small document sets.
        When documents are already concise, directly use them without embedding-based filtering.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            cost_callback: Optional callback for tracking embedding costs.

        Returns:
            Formatted string of relevant document content.
        """
        total_chars = sum(len(str(doc.get('raw_content', ''))) for doc in self.documents)
        chunk_threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))

        if total_chars < chunk_threshold and len(self.documents) <= max_results:
            direct_docs = [
                Document(
                    page_content=doc.get('raw_content', ''),
                    metadata=doc
                )
                for doc in self.documents[:max_results]
            ]
            return self.prompt_family.pretty_print_docs(direct_docs, max_results)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_chunks = splitter.split_documents([
            Document(page_content=doc.get('raw_content', ''), metadata=doc)
            for doc in self.documents
        ])

        embedding_filter = IndividualEmbeddingFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold
        )

        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))

        relevant_docs = await embedding_filter.acompress_documents(all_chunks, query, fallback_top_k=max_results)

        relevant_docs.sort(key=lambda d: d.metadata.get("similarity_score", 0), reverse=True)
        return self.prompt_family.pretty_print_docs(relevant_docs[:max_results], max_results)


class WrittenContentCompressor:
    """Compresses previously written content sections.

    Specialized compressor for finding relevant sections from
    previously written report content, preserving section titles
    and structure.

    Attributes:
        documents: List of written content sections.
        embeddings: Embedding model for similarity calculation.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(self, documents, embeddings, similarity_threshold: float, **kwargs):
        """Initialize the WrittenContentCompressor.

        Args:
            documents: List of written content sections.
            embeddings: Embedding model instance.
            similarity_threshold: Minimum similarity score for inclusion.
            **kwargs: Additional keyword arguments.
        """
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> list[str]:
        """Get relevant written content sections asynchronously.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            cost_callback: Optional callback for tracking embedding costs.

        Returns:
            List of formatted section strings.
        """
        sections = [
            Document(
                page_content=doc.get("written_content", ""),
                metadata={"section_title": doc.get("section_title", "")}
            )
            for doc in self.documents
        ]

        embedding_filter = IndividualEmbeddingFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold
        )

        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))

        relevant_docs = await embedding_filter.acompress_documents(sections, query, fallback_top_k=max_results)

        relevant_docs.sort(key=lambda d: d.metadata.get("similarity_score", 0), reverse=True)
        return self._pretty_docs_list(relevant_docs, max_results)

    def _pretty_docs_list(self, docs: list[Document], top_n: int) -> list[str]:
        return [
            f"Title: {d.metadata.get('section_title')}\nContent: {d.page_content}\n"
            for i, d in enumerate(docs)
            if i < top_n
        ]