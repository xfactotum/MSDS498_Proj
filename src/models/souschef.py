from abc import ABC
from typing import List, Dict, Optional, Union, Any

from langchain.chains.base import Chain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.docstore.document import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.vectorstores.base import VectorStore
from langchain.callbacks.manager import CallbackManagerForChainRun

import time


class Chatbot(Chain, ABC):
    llm: BaseLanguageModel
    knowledge_base: VectorStore
    memory: Union[ConversationEntityMemory, None] = None
    input_key = "question"
    output_key = "answer"
    conversation: Union[ConversationChain, None] = None

    @property
    def input_keys(self) -> str:
        return self.input_key

    @property
    def output_keys(self) -> str:
        return self.output_key

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.memory = ConversationEntityMemory(
            llm=self.llm,
            k=100  # Interactions to use
        )
        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=self.memory
        )

    def get_docs(self, query: str, recipe_ids: List[str] = None) -> List[Document]:
        """
        Gets relevant documents from the knowledge base to provide context in answering the question.
        """
        k = 10
        docs = []
        ids = []
        if recipe_ids:
            for rec_id in recipe_ids:
                for doc in self.knowledge_base.similarity_search_with_score(
                                query=query, filter={"recipe_id": rec_id}):
                    if 'recipe_id' in doc[0].metadata.keys() and doc[0].metadata['recipe_id'] not in ids:
                        docs.append(doc)
                        ids.append(doc[0].metadata['recipe_id'])
        else:
            docs = self.knowledge_base.similarity_search_with_score(query=query, k=10)
        sorted_docs = sorted(docs, key=lambda x: x[1], reverse=True)

        return [doc[0] for doc in sorted_docs]

    @staticmethod
    def combine_documents(documents: List[Document]) -> str:
        """ Transforms List[Documents] into string """
        document_details = [(",".join([k for k in doc.metadata.keys() if doc.metadata[k] == 'ing']), doc.page_content)
                            for doc in documents]
        formatted_documents = [f"recipe {detail}\ningredients: {ingredients}"
                               for ingredients, detail in document_details]
        combined_documents = "\n\n".join(formatted_documents)
        return combined_documents

    def _call(self,
              query: str,
              recipe_ids: Optional[Union[List[str], None]] = None,
              run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict:
        inputs = {self.input_keys: query}

        # Retrieve documents from the knowledge-base and format them
        documents = self.get_docs(query=query, recipe_ids=recipe_ids)
        combined_documents = self.combine_documents(documents)

        qa_prompt = f"""Refer to question and use data delimited by triple backticks to answer with an generated recipe.
        Specifically, use the recipe name and description to match the criteria in the question.
        Enhance the given ingredients with measurements.
        Enhance the given instructions by providing a list of numbered steps with detailed and verbose information.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        If the question is completely unrelated to recipes, apologize and state that you can only answer
        questions about recipes.

        ```{combined_documents}```

        Question: {query}
        Detailed Answer:"""
        t_start = time.time()
        answer = self.conversation.run(qa_prompt)
        outputs = {self.output_key: answer}
        tokens_in = self.llm.get_num_tokens(qa_prompt)
        tokens_out = self.llm.get_num_tokens(answer)
        if self.memory is not None:
            self.memory.save_context(inputs, outputs)
        t_elapsed = round(time.time() - t_start,2)
        return answer, tokens_in, tokens_out, t_elapsed

    def answer(self, query: str, recipe_ids: Union[None, str] = None):
        if not recipe_ids:
            prompt = f"""Answer the question only if it is consistent with the provided context related to recipe.
            If the question is completely unrelated, apologize and state that you can only answer
            questions about recipes.
            
            Question: {query}
            """
            t_start = time.time()
            response = self.conversation.run(prompt)
            tokens_in = self.llm.get_num_tokens(prompt)
            tokens_out = self.llm.get_num_tokens(response)
            inputs = {self.input_keys: query}
            outputs = {self.output_key: response}
            if self.memory is not None:
                self.memory.save_context(inputs, outputs)
            t_elapsed = round(time.time() - t_start,2)
            return response, tokens_in, tokens_out, t_elapsed
        return self._call(query, recipe_ids)

    def clear_conversation_history(self):
        self.conversation.memory.clear()
