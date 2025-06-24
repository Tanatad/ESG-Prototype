# streamlit_app.py

import streamlit as st
import asyncio
import traceback
from typing import List, IO, Dict, Union
import os
from dotenv import load_dotenv
import io
from pymongo import MongoClient

# --- Import all necessary classes ---
from app.services.neo4j_service import Neo4jService
from app.services.persistence.mongodb import MongoDBSaver
from app.services.chat_service import ChatService
from app.services.question_generation_service import QuestionGenerationService
from app.models.esg_question_model import ESGQuestion, ESGQuestionSchema
from app.services.rate_limit import RateLimiter
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

# We only need one function: to get a running event loop for the current thread
def get_main_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# UI Helper function remains the same
def display_question(question_doc: Union[ESGQuestion, ESGQuestionSchema], status: str, key_prefix: str):
    color_map = {"new": "#28a745", "updated": "#007bff", "deactivated": "#dc3545", "unchanged": "grey"}
    status_color = color_map.get(status, "grey")
    text_decoration = "text-decoration: line-through;" if status == "deactivated" else ""
    theme_display = question_doc.theme_th or question_doc.theme
    main_q_display = question_doc.main_question_text_th or question_doc.main_question_text_en
    expander_key = f"{key_prefix}_{status}_{str(question_doc.id)}_{question_doc.version}"
    with st.expander(f"{theme_display} (v{question_doc.version}) - {status.upper()}"):
        st.markdown(f'<h5 style="color:{status_color}; {text_decoration}">{theme_display}</h5>', unsafe_allow_html=True)
        st.markdown(f"<small><b>Category:</b> {question_doc.category} | <b>Version:</b> {question_doc.version}</small>", unsafe_allow_html=True)
        st.info(f"**Main Question:** {main_q_display}")
        if question_doc.sub_questions_sets:
            sub_q_display = question_doc.sub_questions_sets[0].sub_question_text_th or question_doc.sub_questions_sets[0].sub_question_text_en
            text_area_key = f"textarea_{expander_key}"
            st.text_area("Sub-Questions", value=sub_q_display, height=120, disabled=True, key=text_area_key)
        if question_doc.related_set_questions:
            st.success(f"**Covers SET ID:** {question_doc.related_set_questions[0].set_id} - {question_doc.related_set_questions[0].title_th}")

# --- Main App Logic ---
st.set_page_config(page_title="ESG Insight Engine", layout="wide")
st.title("🤖 ESG Insight Engine")

# --- NO MORE GLOBAL/CACHED/SESSION_STATE SERVICE INITIALIZATION ---

tab_demo, tab_summary, tab_qa = st.tabs([
    "⚙️ สาธิตการพัฒนา (Evolution Demo)", 
    "✅ สรุปชุดคำถาม (Final Questions)",
    "💬 ระบบถาม-ตอบ (Q&A)"
])

# ==============================================================================
# --- Tab 1: Evolution Demo ---
# ==============================================================================
with tab_demo:
    st.header("อัปเดตและเปรียบเทียบชุดคำถาม")
    st.sidebar.header("⚙️ Control Panel")
    uploaded_files = st.sidebar.file_uploader("1. เลือกไฟล์ PDF", type="pdf", accept_multiple_files=True)
    is_baseline = st.sidebar.checkbox("2. ตั้งค่าเป็น Baseline", value=True)
    if 'display_items' not in st.session_state:
        st.session_state.display_items = []
        
    if st.sidebar.button("3. เริ่มประมวลผล", type="primary", use_container_width=True):
        if uploaded_files:
            async def run_evolution_pipeline():
                status_placeholder = st.sidebar.empty()
                try:
                    # Create services on-demand for this specific task
                    with status_placeholder.container():
                        st.info("Initializing services for this run...")
                    mongo_uri = os.getenv("MONGO_URL")
                    mongo_db_name = os.getenv("MONGO_DB_NAME")
                    embedding_model = CohereEmbeddings(model='embed-v4.0', cohere_api_key=os.getenv("COHERE_API_KEY"))
                    qg_llm = ChatGoogleGenerativeAI(model=os.getenv("QUESTION_GENERATION_MODEL", "gemini-2.5-flash-preview-05-20"),temperature=0.4, max_retries=3,rate_limiter=RateLimiter(requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "60"))))
                    neo4j_service = Neo4jService()
                    mongodb_service = await MongoDBSaver.from_conn_info(url=mongo_uri, db_name=mongo_db_name, embedding_model=embedding_model)
                    qg_service = QuestionGenerationService(llm=qg_llm, neo4j_service=neo4j_service, mongodb_service=mongodb_service, similarity_embedding_model=embedding_model)
                    
                    with status_placeholder.container():
                        st.info("Reading current question state...")
                    before_state_raw = await mongodb_service.get_all_active_questions_raw()
                    before_state = [ESGQuestionSchema(**data) for data in before_state_raw]

                    with status_placeholder.container():
                        st.info("Ingesting documents into Neo4j...")
                    file_streams = [io.BytesIO(f.getvalue()) for f in uploaded_files]
                    file_names = [f.name for f in uploaded_files]
                    processed_doc_ids = await neo4j_service.flow(files=file_streams, file_names=file_names) 
                    
                    if processed_doc_ids:
                        with status_placeholder.container(): st.info("Evolving questions...")
                        await qg_service.evolve_and_store_questions(document_ids=processed_doc_ids, is_baseline_upload=is_baseline)
                    
                    with status_placeholder.container(): st.info("Preparing comparison view...")
                    all_q_raw = await mongodb_service.get_all_questions_raw()
                    after_state_questions = [ESGQuestionSchema(**data) for data in all_q_raw]
                    after_state_questions.sort(key=lambda x: (x.theme, x.version))
                    
                    before_map = {q.theme: q for q in before_state}
                    display_list = []
                    for q_after in after_state_questions:
                        status = "unchanged"; q_before = before_map.get(q_after.theme)
                        if not q_before: status = "new"
                        elif q_after.version > q_before.version: status = "updated"
                        if not q_after.is_active: status = "deactivated"
                        display_list.append((q_after, status))
                    st.session_state.display_items = display_list
                    status_placeholder.success("Process complete!")
                    st.balloons()
                except Exception as e:
                    st.sidebar.error(f"An error occurred: {e}")
                    traceback.print_exc()

            get_main_event_loop().run_until_complete(run_evolution_pipeline())
        else:
            st.sidebar.warning("กรุณาอัปโหลดไฟล์")

    if not st.session_state.display_items: st.info("อัปโหลดไฟล์และกด 'เริ่มประมวลผล' เพื่อดูผลลัพธ์การเปรียบเทียบ")
    else:
        st.subheader("ผลลัพธ์การเปรียบเทียบ (จัดกลุ่มตามการเปลี่ยนแปลง)")
        display_data = {"new": [], "updated": [], "deactivated": [], "unchanged": []}
        for q_doc, status in st.session_state.display_items:
            if status in display_data: display_data[status].append(q_doc)
        for status, questions in display_data.items():
            if questions:
                if status == "new": header_text = f"✅ ใหม่ ({len(questions)})"
                elif status == "updated": header_text = f"🔄 อัปเดต ({len(questions)})"
                elif status == "deactivated": header_text = f"❌ แทนที่ ({len(questions)})"
                else: header_text = f"➖ ไม่เปลี่ยนแปลง ({len(questions)})"
                st.markdown(f"**{header_text}**")
                for q in sorted(questions, key=lambda x: x.theme):
                    display_question(q, status, key_prefix="comparison")

# ==============================================================================
# --- Tab 2: Final Questions Summary ---
# ==============================================================================
with tab_summary:
    st.header("สรุปชุดคำถามเวอร์ชันล่าสุดที่พร้อมใช้งาน")
    if st.button("โหลด/รีเฟรชชุดคำถามล่าสุด", key="load_summary", use_container_width=True):
        with st.spinner("กำลังดึงข้อมูลจาก MongoDB..."):
            try:
                mongo_uri = os.getenv("MONGO_URL"); mongo_db_name = os.getenv("MONGO_DB_NAME"); client = None
                try:
                    client = MongoClient(mongo_uri)
                    db = client[mongo_db_name]
                    collection = db["esg_questions_final"] 
                    results_raw = list(collection.find({"is_active": True}).sort([("category", 1), ("theme", 1)]))
                    st.session_state.summary_questions = [ESGQuestionSchema(**data) for data in results_raw]
                finally:
                    if client: client.close()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}"); traceback.print_exc(); st.session_state.summary_questions = []
    if 'summary_questions' in st.session_state and st.session_state.summary_questions:
        st.success(f"พบชุดคำถามที่พร้อมใช้งานทั้งหมด {len(st.session_state.summary_questions)} รายการ")
        questions_by_cat = {"E": [], "S": [], "G": []}
        for q in st.session_state.summary_questions:
            if q.category in questions_by_cat: questions_by_cat[q.category].append(q)
        for category, questions in questions_by_cat.items():
            if questions:
                st.subheader(f"หมวดหมู่: {category}")
                for question_doc in questions: display_question(question_doc, "unchanged", key_prefix="summary")
    else:
        st.info("กดปุ่ม 'โหลด/รีเฟรช' เพื่อแสดงชุดคำถามที่พร้อมใช้งานล่าสุด")

# ==============================================================================
# --- Tab 3: Q&A Chat ---
# ==============================================================================
with tab_qa:
    st.header("ถาม-ตอบจาก Knowledge Graph (Ad-hoc Query)")
    if "chat_history_langgraph" not in st.session_state: st.session_state.chat_history_langgraph = []
    if "session_id" not in st.session_state: st.session_state.session_id = f"streamlit-thread-{os.urandom(8).hex()}"
    if not st.session_state.chat_history_langgraph:
        st.session_state.chat_history_langgraph.append({"role": "assistant", "content": f"สวัสดีครับ เริ่มการสนทนาได้เลย (Session ID: {st.session_state.session_id})"})

    for message in st.session_state.chat_history_langgraph:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("ถามคำถามของคุณที่นี่..."):
        st.session_state.chat_history_langgraph.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("🧠 กำลังคิดด้วย LangGraph...")
            try:
                async def run_chat_pipeline():
                    # --- FIX: Create ChatService on-demand for this specific request ---
                    chat_service = await ChatService.create()
                    return await chat_service.graph.ainvoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config={"configurable": {"thread_id": st.session_state.session_id}}
                    )

                final_state = get_main_event_loop().run_until_complete(run_chat_pipeline())
                
                ai_response = final_state["messages"][-1]
                response_content = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
                
                response_placeholder.markdown(response_content)
                st.session_state.chat_history_langgraph.append({"role": "assistant", "content": response_content})
                st.rerun()
            except Exception as e:
                error_message = f"ขออภัย, เกิดข้อผิดพลาด: {e}"; response_placeholder.error(error_message)
                st.session_state.chat_history_langgraph.append({"role": "assistant", "content": error_message})
                traceback.print_exc()
                st.rerun()