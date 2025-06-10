from typing import List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.persistence.mongodb import AsyncMongoDBSaver
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from app.services.neo4j_service import Neo4jService
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
from dotenv import load_dotenv
from app.services.rate_limit import llm_rate_limiter
import os
from enum import Enum

load_dotenv()

# คลาสสำหรับบริการแปลภาษาโดยใช้ LLM
class LLMTranslationService:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def translate(self, text: str, target_language: str) -> str:
        """
        แปลข้อความโดยใช้ LLM ที่กำหนด
        """
        if not text or text.strip() == "":
            return ""

        lang_map = {"en": "English", "th": "Thai"}
        target_lang_name = lang_map.get(target_language, target_language)
        
        # สร้าง Prompt สำหรับการแปลภาษา
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
            # คาดหวังว่า content จะเป็นข้อความที่แปลแล้ว
            return response.content.strip()
        except Exception as e:
            print(f"LLM Translation failed: {e}")
            # หากการแปลล้มเหลว ให้คืนค่าข้อความเดิมกลับไป
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
    # --- ส่วนที่เพิ่มเข้ามา ---
    # เพิ่ม state เพื่อเก็บคำถามที่จะใช้ใน RAG prompt
    question_for_prompt: str = None
    # --- สิ้นสุดส่วนที่เพิ่มเข้ามา ---

class ChatService:
    def __init__(self, 
                 memory,
                 Neo4jService: Neo4jService
                 ):
        self.memory = memory
        self.Neo4jService = Neo4jService
        self.graph = self.graph_workflow().compile(checkpointer=self.memory)
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("CHAT_MODEL","gemini-2.5-flash-preview-05-20"), #ปรับแก้เป็น Model ที่รองรับ Gemini API
            max_retries=10,
            rate_limiter=llm_rate_limiter
        )
        self.translator = LLMTranslationService(llm=self.llm)
        
    @classmethod
    async def create(cls):
        url = os.getenv("MONGO_URL") or None
        db_name = os.getenv("MONGO_DB_NAME") or None

        async with AsyncMongoDBSaver.from_conn_info(
            url=url, db_name=db_name
        ) as memory:
            neo4j_service = Neo4jService()
            return cls(memory, neo4j_service)
    
    def _is_thai(self, text: str) -> bool:
        for char in text:
            if "\u0E00" <= char <= "\u0E7F":
                return True
        return False
    
    async def __asummarize_conversation(self,state: State):
        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        model = self.llm
        messages = state["messages"][-6:] + [HumanMessage(content=summary_message)]
        response = await model.ainvoke(messages)
        
        return {"summary": response.content}
    
    def __should_continue(self,state: State):
        messages = state["messages"]
        if len(messages) > 6:
            return "summarize_conversation"
        return END
    
    # --- ส่วนที่แก้ไข ---
    async def __retrieve(self,state: State):
        original_question = state.get("messages", "")[-1].content
        
        is_thai = self._is_thai(original_question)
        question_for_retrieval = original_question
        
        # ตรวจสอบว่าเป็นภาษาไทยหรือไม่
        if is_thai:
            print(f"ตรวจพบคำถามภาษาไทย: '{original_question}'")
            # แปลคำถามเป็นภาษาอังกฤษเพื่อใช้ในการ retrieve
            question_for_retrieval = await self.translator.translate(original_question, target_language="en")
            print(f"แปลเป็นภาษาอังกฤษ: '{question_for_retrieval}'")
        
        # กำหนดคำถามที่จะใช้ใน Prompt สุดท้าย ให้เป็นภาษาอังกฤษเสมอหากมีการแปลเกิดขึ้น
        question_for_final_prompt = question_for_retrieval
        
        # ดำเนินการ retrieve ด้วยคำถามภาษาอังกฤษ
        retriever = await self.Neo4jService.get_output(question_for_retrieval,k=10)
        
        # คืนค่าต่างๆ พร้อมกับคำถามสำหรับ Prompt
        return {
            "cypher_answer": retriever.cypher_answer, 
            "documents": retriever.relate_documents,
            "is_thai_question": is_thai,
            "question_for_prompt": question_for_final_prompt 
        }
    # --- สิ้นสุดส่วนที่แก้ไข ---
    
    # --- ส่วนที่แก้ไข ---
    async def __esg_model(self,state: State):
        model = self.llm
        context_type = state.get("context_type", ContextType.BOTH)
        cypher_answer = ""
        docs = []
        
        if context_type == ContextType.CYPHER:
            cypher_answer = state.get("cypher_answer", "")  
        elif context_type == ContextType.VECTOR:
            docs = state.get("documents", [])
        elif context_type == ContextType.BOTH:
            cypher_answer = state.get("cypher_answer", "")
            docs = state.get("documents", [])
            
        # ใช้ 'question_for_prompt' จาก state ซึ่งเป็นภาษาอังกฤษ
        question_to_use = state.get("question_for_prompt")

        system_prompt, user_prompt = self.__prompt_rag(question_to_use,
                                                       docs,
                                                       cypher_answer)
        prompt_template = ChatPromptTemplate([
            system_prompt,
            MessagesPlaceholder("msgs"),
            user_prompt,
        ])
        rag_chain = prompt_template | model 
        messages = state["messages"]
        summary = state.get("summary", "")
        if summary:
            system_message = f"Summary of conversation earlier: {summary}"
            messages = [SystemMessage(content=system_message)] + state["messages"]
        messages = trim_messages(
            state["messages"],
            max_tokens=90000,
            strategy="last",
            token_counter=self.llm,
            allow_partial=False,
        )
        response = await rag_chain.ainvoke({"msgs": messages[:-1]})

        # หากคำถามตั้งต้นเป็นภาษาไทย ให้แปลคำตอบกลับเป็นไทย
        if state.get("is_thai_question", False):
            print(f"ได้รับคำตอบเป็นภาษาอังกฤษ: '{response.content}'")
            thai_content = await self.translator.translate(response.content, target_language="th")
            print(f"แปลกลับเป็นภาษาไทย: '{thai_content}'")
            response = AIMessage(content=thai_content, id=response.id)

        return {"messages": [response]}
    # --- สิ้นสุดส่วนที่แก้ไข ---
        
    def __prompt_rag(self,question:str,documents: List[Document],structure:str) -> Tuple[SystemMessage, HumanMessage]:
        context = "\n".join([doc.page_content for doc in documents])
        system_prompt = f"""
                                    You are an assistant for question-answering tasks.
                                    You answer the question directly without 'Context: ' or 'Answer: '.
                                    """
        
        # --- PROMPT ที่แก้ไขแล้ว ---
        # เปลี่ยนจาก "concise and direct" เป็นการสั่งให้ตอบอย่างละเอียดและครอบคลุม
        user_prompt = f"""
                            Question: {question}

                            Use the provided context and structure to provide a comprehensive and detailed answer.
                            Synthesize the information from the context to be as helpful as possible.
                            If the context doesn’t provide enough information, use general knowledge to assist. If neither is sufficient, state that you don’t know the answer.

                            Structure: {structure}
                            Context: {context}
                                    """
        
        return SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)
    
    async def delete_by_thread_id(self,thread_id: str):
        await self.memory.delete_by_thread_id(thread_id)
        
    def graph_workflow(self):
        builder = StateGraph(State)
        builder.add_node("retrieve", self.__retrieve)
        builder.add_node("esg_model", self.__esg_model)
        builder.add_node("summarize_conversation", self.__asummarize_conversation)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "esg_model")
        builder.add_conditional_edges("esg_model",self.__should_continue,{
            "summarize_conversation": "summarize_conversation",
            END: END
        })
        builder.add_edge("summarize_conversation", END)
        return builder