import streamlit as st
import requests
import traceback
import os
import json
from typing import List, Dict, Union
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
UPLOAD_URL = f"{FASTAPI_BASE_URL}/api/v1/graph/uploadfile"
QUESTIONS_URL = f"{FASTAPI_BASE_URL}/api/v1/question-ai/questions/active"
CHAT_URL = f"{FASTAPI_BASE_URL}/api/v1/chat/invoke"

# --- UI Helper Function ---
# (ฟังก์ชัน display_question เหมือนเดิม ไม่ต้องแก้ไข)
def display_question(question_doc: Dict, status: str, key_prefix: str):
    color_map = {"new": "#28a745", "updated": "#007bff", "deactivated": "#dc3545", "unchanged": "grey"}
    status_color = color_map.get(status, "grey")
    text_decoration = "text-decoration: line-through;" if status == "deactivated" else ""
    
    theme_display = question_doc.get("theme_th") or question_doc.get("theme")
    main_q_display = question_doc.get("main_question_text_th") or question_doc.get("main_question_text_en")
    
    # Use a combination of fields for a more unique key
    doc_id = question_doc.get("_id", str(question_doc.get("theme")))
    expander_key = f"{key_prefix}_{status}_{doc_id}_{question_doc.get('version')}"

    with st.expander(f"{theme_display} (v{question_doc.get('version')}) - {status.upper()}"):
        st.markdown(f'<h5 style="color:{status_color}; {text_decoration}">{theme_display}</h5>', unsafe_allow_html=True)
        st.markdown(f"<small><b>Category:</b> {question_doc.get('category')} | <b>Version:</b> {question_doc.get('version')}</small>", unsafe_allow_html=True)
        st.info(f"**Main Question:** {main_q_display}")
        
        sub_questions_sets = question_doc.get("sub_questions_sets", [])
        if sub_questions_sets:
            sub_q_display = sub_questions_sets[0].get("sub_question_text_th") or sub_questions_sets[0].get("sub_question_text_en")
            st.text_area("Sub-Questions", value=sub_q_display, height=120, disabled=True, key=f"textarea_{expander_key}")
            
        related_set_questions = question_doc.get("related_set_questions", [])
        if related_set_questions:
            st.success(f"**Covers SET ID:** {related_set_questions[0].get('set_id')} - {related_set_questions[0].get('title_th')}")

# --- Main App ---
st.set_page_config(page_title="ESG Insight Engine", layout="wide")
st.title("🤖 ESG Insight Engine")

# --- FIX: Replace st.tabs with st.radio for stateful navigation ---
tabs = ["⚙️ สาธิตการพัฒนา (Evolution Demo)", "✅ สรุปชุดคำถาม (Final Questions)", "💬 ระบบถาม-ตอบ (Q&A)"]
active_tab = st.radio("Navigation", tabs, horizontal=True, label_visibility="collapsed")

# ==============================================================================
# --- Tab 1: Evolution Demo ---
# ==============================================================================
if active_tab == tabs[0]:
    st.header("อัปเดตและเปรียบเทียบชุดคำถาม")
    st.sidebar.header("⚙️ Control Panel")
    uploaded_files = st.sidebar.file_uploader("1. เลือกไฟล์ PDF", type="pdf", accept_multiple_files=True)
    is_baseline = st.sidebar.checkbox("2. ล้างข้อมูลและตั้งค่าเป็น Baseline", value=False)
    
    if 'display_items' not in st.session_state:
        st.session_state.display_items = []
        
    if st.sidebar.button("3. เริ่มประมวลผล", type="primary", use_container_width=True):
        if uploaded_files:
            files_to_upload = [('files', (f.name, f.getvalue(), f.type)) for f in uploaded_files]

            with st.spinner("Processing... This may take several minutes."):
                try:
                    params = {"is_baseline": is_baseline}
                    response = requests.post(UPLOAD_URL, files=files_to_upload, params=params, timeout=600)
                    
                    if response.status_code == 200:
                        st.session_state.display_items = response.json()
                        st.sidebar.success("Process complete!")
                        st.balloons()
                    else:
                        st.sidebar.error(f"Error from API: {response.status_code}")
                        try:
                            st.sidebar.json(response.json())
                        except json.JSONDecodeError:
                            st.sidebar.text(response.text)
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"A network error occurred: {e}")
                except Exception as e:
                    st.sidebar.error(f"An unexpected error occurred: {e}")
                    traceback.print_exc()
        else:
            st.sidebar.warning("กรุณาอัปโหลดไฟล์")

    if not st.session_state.display_items:
        st.info("อัปโหลดไฟล์และกด 'เริ่มประมวลผล' เพื่อดูผลลัพธ์การเปรียบเทียบ")
    else:
        st.subheader("ผลลัพธ์การเปรียบเทียบ (จัดกลุ่มตามการเปลี่ยนแปลง)")
        display_data = {"new": [], "updated": [], "deactivated": [], "unchanged": []}
        for item in st.session_state.display_items:
            status = item.get("status")
            if status in display_data:
                display_data[status].append(item.get("question"))
        
        for status, questions in display_data.items():
            if questions:
                if status == "new": header_text = f"✅ ใหม่ ({len(questions)})"
                elif status == "updated": header_text = f"🔄 อัปเดต ({len(questions)})"
                elif status == "deactivated": header_text = f"❌ แทนที่ ({len(questions)})"
                else: header_text = f"➖ ไม่เปลี่ยนแปลง ({len(questions)})"
                
                st.markdown(f"**{header_text}**")
                for q in sorted(questions, key=lambda x: x.get("theme", "")):
                    display_question(q, status, key_prefix="comparison")

# ==============================================================================
# --- Tab 2: Final Questions Summary ---
# ==============================================================================
elif active_tab == tabs[1]:
    st.header("สรุปชุดคำถามเวอร์ชันล่าสุดที่พร้อมใช้งาน")
    if 'summary_questions' not in st.session_state:
        st.session_state.summary_questions = None

    if st.button("โหลด/รีเฟรชชุดคำถามล่าสุด", key="load_summary", use_container_width=True):
        with st.spinner("กำลังดึงข้อมูลจาก API..."):
            try:
                response = requests.get(QUESTIONS_URL)
                if response.status_code == 200:
                    st.session_state.summary_questions = response.json()
                else:
                    st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {response.text}")
                    st.session_state.summary_questions = []
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: {e}")
                st.session_state.summary_questions = []

    if st.session_state.summary_questions is not None:
        if st.session_state.summary_questions:
            st.success(f"พบชุดคำถามที่พร้อมใช้งานทั้งหมด {len(st.session_state.summary_questions)} รายการ")
            questions_by_cat = {"E": [], "S": [], "G": []}
            for q in st.session_state.summary_questions:
                cat = q.get("category")
                if cat in questions_by_cat:
                    questions_by_cat[cat].append(q)
            
            for category, questions in questions_by_cat.items():
                if questions:
                    st.subheader(f"หมวดหมู่: {category}")
                    for question_doc in sorted(questions, key=lambda x: x.get("theme", "")):
                        display_question(question_doc, "unchanged", key_prefix="summary")
        else:
            st.warning("ไม่พบชุดคำถามที่พร้อมใช้งานในระบบ")
    else:
        st.info("กดปุ่ม 'โหลด/รีเฟรช' เพื่อแสดงชุดคำถามที่พร้อมใช้งานล่าสุด")

# ==============================================================================
# --- Tab 3: Q&A Chat ---
# ==============================================================================
elif active_tab == tabs[2]:
    st.header("ถาม-ตอบจาก Knowledge Graph (Ad-hoc Query)")
    
    # --- Session State Initialization ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"st-session-{os.urandom(8).hex()}"

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("ถามคำถามของคุณที่นี่..."):
        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response and display
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    payload = {
                        "question": prompt,
                        "thread_id": st.session_state.session_id,
                    }
                    response = requests.post(CHAT_URL, json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        chat_response = response.json()
                        messages = chat_response.get("messages", [])
                        if messages:
                            ai_message = messages[-1]
                            response_content = ai_message.get("content", "Sorry, I couldn't get a proper response.")
                        else:
                            response_content = "The chat service returned no messages."
                    else:
                        response_content = f"Error from Chat API ({response.status_code}): {response.text}"
                
                except requests.exceptions.RequestException as e:
                    response_content = f"A network error occurred: {e}"
                except Exception as e:
                    response_content = f"An unexpected error occurred: {e}"
                    traceback.print_exc()
                
                st.markdown(response_content)
                # Add AI response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response_content})