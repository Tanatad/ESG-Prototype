from typing import List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.persistence.mongodb import MongoDBSaver
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from app.services.neo4j_service import Neo4jService
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
from dotenv import load_dotenv
from app.services.rate_limit import RateLimiter as llm_rate_limiter
import os
from enum import Enum
import time # Import time for debugging

load_dotenv()

RPM_LIMIT = int(os.getenv("REQUESTS_PER_MINUTE", "60"))

class LLMTranslationService:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def translate(self, text: str, target_language: str) -> str:
        if not text or text.strip() == "":
            return ""

        lang_map = {"en": "English", "th": "Thai"}
        target_lang_name = lang_map.get(target_language, target_language)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        f"You are a professional translator. Translate the following text to {target_lang_name}. "
                        "Respond with ONLY the translated text, without any introductory phrases, comments, or explanations."
                    )
                ),
                HumanMessage(content=text),
            ]
        )
        
        chain = prompt | self.llm
        
        try:
            response = await chain.ainvoke({})
            return response.content.strip()
        except Exception as e:
            print(f"--- [ERROR] LLM Translation failed: {e}") # Added error print
            return text

class ContextType(Enum):
    BOTH = "both"
    CYPHER = "cypher"
    VECTOR = "documents"
    EMPTY = "empty"

class State(MessagesState):
    summary: str
    documents : List[Document]
    cypher_answer: str
    context_type: ContextType = ContextType.BOTH
    is_thai_question: bool = False
    question_for_prompt: str = None

class ChatService:
    def __init__(self, 
                 memory: MongoDBSaver,
                 neo4j_service: Neo4jService
                ):
        self.memory = memory
        self.Neo4jService = neo4j_service
        self.graph = self.graph_workflow().compile(checkpointer=self.memory)
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("CHAT_MODEL", "gemini-2.5-flash-preview-05-20"), 
            max_retries=2, # ลด retries ตอนดีบัก
            rate_limiter=llm_rate_limiter(requests_per_minute=RPM_LIMIT) 
        )
        self.translator = LLMTranslationService(llm=self.llm)
        
    @classmethod
    async def create(cls):
        url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME")

        # เรียกใช้งานตรงๆ เพื่อรับ instance ที่การเชื่อมต่อยังคงอยู่
        memory = await MongoDBSaver.from_conn_info(
            url=url, db_name=db_name
        )
        
        neo4j_service = Neo4jService()
        return cls(memory, neo4j_service)
    
    # ... (ส่วน _is_thai และ __asummarize_conversation ไม่ได้แก้ไข) ...
    def _is_thai(self, text: str) -> bool:
        for char in text:
            if "\u0E00" <= char <= "\u0E7F":
                return True
        return False
    
    async def __asummarize_conversation(self, state: State):
        summary = state.get("summary", "")
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        ) if summary else "Create a summary of the conversation above:"

        model = self.llm
        messages = state["messages"][-6:] + [HumanMessage(content=summary_message)]
        response = await model.ainvoke(messages)
        
        return {"summary": response.content}
    
    def __should_continue(self, state: State):
        if len(state["messages"]) > 6:
            return "summarize_conversation"
        return END
    
    async def __retrieve(self, state: State):
        print("\n--- [DEBUG 1] Entered '__retrieve' node ---")
        original_question = state["messages"][-1].content
        print(f"--- [DEBUG 2] Original question: {original_question} ---")
        
        is_thai = self._is_thai(original_question)
        
        if is_thai:
            print("--- [DEBUG 3] Thai question detected. Translating... ---")
            question_for_retrieval = await self.translator.translate(original_question, target_language="en")
            print(f"--- [DEBUG 4] Translated to English: {question_for_retrieval} ---")
        else:
            print("--- [DEBUG 3] English question detected. No translation needed. ---")
            question_for_retrieval = original_question
        
        print("--- [DEBUG 5] Calling Neo4j service to get graph output... ---")
        start_time = time.time()
        retriever = await self.Neo4jService.get_output(question_for_retrieval, k=10)
        end_time = time.time()
        print(f"--- [DEBUG 6] Neo4j call finished. Took {end_time - start_time:.2f} seconds. ---")
        
        result = {
            "cypher_answer": retriever.cypher_answer, 
            "documents": retriever.relate_documents,
            "is_thai_question": is_thai,
            "question_for_prompt": question_for_retrieval 
        }
        print("--- [DEBUG 7] Exiting '__retrieve' node. ---")
        return result
    
    async def __esg_model(self, state: State):
        print("\n--- [DEBUG 8] Entered '__esg_model' node ---")
        model = self.llm
        question_to_use = state.get("question_for_prompt")
        print(f"--- [DEBUG 9] Question for RAG prompt: {question_to_use} ---")

        docs = state.get("documents", [])
        cypher_answer = state.get("cypher_answer", "")

        system_prompt, user_prompt = self.__prompt_rag(question_to_use, docs, cypher_answer)
        
        prompt_template = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder("msgs"),
            user_prompt,
        ])
        
        rag_chain = prompt_template | model 
        
        messages_to_send = state["messages"]
        
        print("--- [DEBUG 10] Calling main RAG chain (LLM)... ---")
        start_time = time.time()
        response = await rag_chain.ainvoke({"msgs": messages_to_send[:-1]})
        end_time = time.time()
        print(f"--- [DEBUG 11] Main RAG chain finished. Took {end_time - start_time:.2f} seconds. ---")

        if state.get("is_thai_question", False):
            print("--- [DEBUG 12] Translating response back to Thai... ---")
            thai_content = await self.translator.translate(response.content, target_language="th")
            response = AIMessage(content=thai_content, id=response.id)
            print("--- [DEBUG 13] Finished translating back to Thai. ---")

        print("--- [DEBUG 14] Exiting '__esg_model' node. ---")
        return {"messages": [response]}
        
    # ... (ส่วน __prompt_rag, delete_by_thread_id, graph_workflow ไม่ได้แก้ไข) ...
    def __prompt_rag(self, question: str, documents: List[Document], structure: str) -> Tuple[SystemMessage, HumanMessage]:
        context = "\n".join([doc.page_content for doc in documents])
        system_prompt_content = """
You are an assistant for question-answering tasks.
You answer the question directly without 'Context: ' or 'Answer: '.
"""
        user_prompt_content = f"""
Question: {question}

Use the provided context and structure to provide a comprehensive and detailed answer.
Synthesize the information from the context to be as helpful as possible.
If the context doesn’t provide enough information, use general knowledge to assist. If neither is sufficient, state that you don’t know the answer.

Structure: {structure}
Context: {context}
"""
        return SystemMessage(content=system_prompt_content), HumanMessage(content=user_prompt_content)
    
    async def delete_by_thread_id(self, thread_id: str):
        await self.memory.delete_by_thread_id(thread_id)
        
    def graph_workflow(self):
        builder = StateGraph(State)
        builder.add_node("retrieve", self.__retrieve)
        builder.add_node("esg_model", self.__esg_model)
        builder.add_node("summarize_conversation", self.__asummarize_conversation)
        
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "esg_model")
        builder.add_conditional_edges(
            "esg_model",
            self.__should_continue,
            {
                "summarize_conversation": "summarize_conversation",
                END: END
            }
        )
        builder.add_edge("summarize_conversation", END)
        return builder