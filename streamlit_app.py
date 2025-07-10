import streamlit as st
import requests
import traceback
import os
import json
from typing import List, Dict, Union
from dotenv import load_dotenv
import markdown2

# --- Configuration ---
load_dotenv()
FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
# URL สำหรับ Use Case 1: พัฒนาชุดคำถาม
UPLOAD_URL = f"{FASTAPI_BASE_URL}/api/v1/graph/uploadfile"
# URL สำหรับ Use Case 2: สร้างรายงาน
REPORT_GEN_URL = f"{FASTAPI_BASE_URL}/api/v1/report/generate" 
QUESTIONS_URL = f"{FASTAPI_BASE_URL}/api/v1/question-ai/questions/active"
CHAT_URL = f"{FASTAPI_BASE_URL}/api/v1/chat/invoke"

# --- UI Helper Functions ---

def render_report_as_page(markdown_text: str) -> str:
    """Converts markdown to HTML and wraps it in a page-like CSS style."""
    
    # Convert markdown to HTML
    html_body = markdown2.markdown(
        markdown_text, 
        extras=["tables", "fenced-code-blocks", "spoiler", "header-ids"]
    )
    
    # --- FIX: Improved CSS for a more professional look ---
    styled_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
        
        .report-container {{
            font-family: 'Sarabun', sans-serif;
            background-color: #FFFFFF;
            color: #333333;
            padding: 50px 60px;
            border-radius: 3px;
            box-shadow: 0 6px 20px 0 rgba(0,0,0,0.19);
            max-width: 850px;
            margin: auto;
            line-height: 1.6;
        }}
        .report-container h1 {{
            color: #1a1a1a;
            text-align: center;
            border-bottom: 2px solid #005A9C;
            padding-bottom: 15px;
            margin-bottom: 40px;
        }}
        .report-container h2 {{
            color: #005A9C;
            border-bottom: 1px solid #DDDDDD;
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px;
        }}
        .report-container h3 {{
            color: #333333;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .report-container p {{
            text-align: justify;
        }}
        .report-container ul {{
            padding-left: 20px;
        }}
        .report_container hr {{
            border: 1px solid #EEEEEE;
            margin: 40px 0;
        }}
    </style>
    <div class="report-container">
        {html_body}
    </div>
    """
    return styled_html

def display_question(question_doc: Dict, status: str, key_prefix: str):
    """Displays a single question item in an expander."""
    color_map = {"new": "#28a745", "updated": "#007bff", "deactivated": "#dc3545", "unchanged": "grey"}
    status_color = color_map.get(status, "grey")
    text_decoration = "text-decoration: line-through;" if status == "deactivated" else ""
    
    theme_display = question_doc.get("theme_th") or question_doc.get("theme")
    main_q_display = question_doc.get("main_question_text_th") or question_doc.get("main_question_text_en")
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

tabs = [
    "📝 สร้างรายงาน (Report)",
    "⚙️ สาธิตการพัฒนา (Dev Demo)", 
    "✅ สรุปชุดคำถาม (Final Questions)",
    "💬 ระบบถาม-ตอบ (Q&A)"
]
active_tab = st.radio("Navigation", tabs, horizontal=True, label_visibility="collapsed")

# ==============================================================================
# --- Tab 1: Report Generation (Use Case 2) ---
# ==============================================================================
if active_tab == tabs[0]:
    st.header("สร้างรายงานความยั่งยืนอัตโนมัติ")
    st.info("อัปโหลดไฟล์ PDF ของบริษัทคุณ (เช่น 56-1 One Report, รายงานประจำปี) เพื่อให้ระบบวิเคราะห์และสร้างร่างรายงานความยั่งยืนเบื้องต้น")

    company_name = st.text_input("ชื่อบริษัท (Company Name)", placeholder="เช่น บริษัท เอสซีจี แพคเกจจิ้ง จํากัด (มหาชน)")
    report_uploaded_files = st.file_uploader("เลือกไฟล์ PDF ของคุณ", type="pdf", accept_multiple_files=True, key="report_uploader")

    if st.button("🚀 เริ่มสร้างรายงาน", type="primary", use_container_width=True):
        if report_uploaded_files and company_name:
            files_to_upload = [('files', (f.name, f.getvalue(), f.type)) for f in report_uploaded_files]
            form_data = {"company_name": company_name}

            with st.spinner("กำลังวิเคราะห์เอกสารและสร้างรายงานฉบับสมบูรณ์... (ขั้นตอนนี้อาจใช้เวลาหลายนาที กรุณาอย่าปิดหน้าต่างนี้)"):
                try:
                    # --- FIX: Increase the timeout to 1 hour (3600 seconds) ---
                    response = requests.post(
                        REPORT_GEN_URL, 
                        files=files_to_upload, 
                        data=form_data,
                        timeout=3600 
                    ) 
                    # -----------------------------------------------------------
                    
                    if response.status_code == 200:
                        st.session_state.report_output = response.json()
                        st.success("สร้างรายงานสำเร็จ!")
                    else:
                        st.sidebar.error(f"Error from API: {response.status_code}")
                        st.sidebar.json(response.json())
                except requests.exceptions.ReadTimeout:
                    st.error("การประมวลผลใช้เวลานานเกินไป (Timeout) กรุณาลองอีกครั้งด้วยจำนวนไฟล์ที่น้อยลง หรือตรวจสอบสถานะของเซิร์ฟเวอร์")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
                    traceback.print_exc()
        else:
            st.warning("กรุณากรอกชื่อบริษัทและอัปโหลดไฟล์ PDF")

    if 'report_output' in st.session_state:
        st.subheader("ผลลัพธ์การวิเคราะห์")
        report_data = st.session_state.report_output
        
        markdown_report = report_data.get("markdown_report", "# ไม่สามารถสร้างรายงานได้")
        raw_data = report_data.get("raw_data", [])
        
        sufficient_count = sum(1 for item in raw_data if item.get("status") == "sufficient")
        
        st.metric("หัวข้อที่ข้อมูลเพียงพอ (Sufficient)", f"{sufficient_count} / {len(raw_data)}")
        
        st.subheader("ร่างรายงานความยั่งยืน (Sustain Report Draft)")
        
        report_html = render_report_as_page(markdown_report)
        st.components.v1.html(report_html, height=800, scrolling=True)
        
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📥 ดาวน์โหลด (Markdown)",
                data=markdown_report,
                file_name="sustainability_report_draft.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        # --- FIX: Add the "Download PDF" button and its logic ---
        with col2:
            if st.button("📥 ดาวน์โหลด (PDF)", type="primary", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_response = requests.post(
                            f"{FASTAPI_BASE_URL}/api/v1/report/create-pdf",
                            json={"markdown_content": markdown_report},
                            timeout=120
                        )
                        if pdf_response.status_code == 200:
                            # Create a download button for the received PDF bytes
                            st.download_button(
                                label="✅ คลิกที่นี่เพื่อดาวน์โหลด PDF!",
                                data=pdf_response.content,
                                file_name="sustainability_report.pdf",
                                mime="application/pdf",
                                key="pdf_download_final"
                            )
                        else:
                            st.error("Failed to generate PDF.")
                            st.json(pdf_response.json())
                    except Exception as e:
                        st.error(f"An error occurred during PDF generation: {e}")
        # -----------------------------------------------------------
        
        with st.expander("ดูข้อมูลดิบและ Context ที่ใช้ (สำหรับ Debug)"):
            st.json(raw_data)

# ==============================================================================
# --- Tab 2: Question Evolution Demo (Use Case 1) ---
# ==============================================================================
elif active_tab == tabs[1]:
    st.header("อัปเดตและเปรียบเทียบชุดคำถาม")
    st.sidebar.header("⚙️ Control Panel")
    uploaded_files = st.sidebar.file_uploader("1. เลือกไฟล์ PDF (สำหรับพัฒนาชุดคำถาม)", type="pdf", accept_multiple_files=True, key="dev_uploader")
    is_baseline = st.sidebar.checkbox("2. ล้างข้อมูลและตั้งค่าเป็น Baseline", value=False)
    
    if 'display_items' not in st.session_state:
        st.session_state.display_items = []
        
    if st.sidebar.button("3. เริ่มประมวลผล (Dev)", type="primary", use_container_width=True):
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
# --- Tab 3: Final Questions Summary ---
# ==============================================================================
elif active_tab == tabs[2]:
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
# --- Tab 4: Q&A Chat ---
# ==============================================================================
elif active_tab == tabs[3]:
    st.header("ถาม-ตอบจาก Knowledge Graph (Ad-hoc Query)")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"st-session-{os.urandom(8).hex()}"

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("ถามคำถามของคุณที่นี่..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

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
                st.session_state.chat_history.append({"role": "assistant", "content": response_content})