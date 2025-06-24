from typing import List, Dict, Any

benchmark_questions: List[Dict[str, Any]] = [
    {
        "id": "SET_E_1", "dimension": "E", "theme_set": "Reporting on SIA/EIA",
        "question_text_th": "โรงงานมีการการเปิดเผยข้อมูลการประเมินผลกระทบด้านสังคมและสิ่งแวดล้อม (SIA/EIA) หรือไม่?",
        "question_text_en": "Does the factory disclose its social and environmental impact assessment (SIA/EIA)?"
    },
    {
        "id": "SET_E_2", "dimension": "E", "theme_set": "Reporting on SIA/EIA",
        "question_text_th": "โรงงานมีการการเปิดเผยข้อมูลเกี่ยวกับกระบวนการติดตามผลกระทบด้านสังคมและสิ่งแวดล้อม (SIA/EIA) หรือไม่?",
        "question_text_en": "Does the factory disclose its process for monitoring social and environmental impacts (SIA/EIA)?"
    },
    {
        "id": "SET_E_3", "dimension": "E", "theme_set": "Environmentally Friendly Products",
        "question_text_th": "โรงงานมีนโยบายและแนวปฏิบัติเกี่ยวกับการป้องกันการปนเปื้อนหรือรั่วไหลจากกระบวนการผลิตหรือไม่?",
        "question_text_en": "Does the factory have a policy and guidelines for preventing contamination or leakage from its production processes?"
    },
    {
        "id": "SET_E_4", "dimension": "E", "theme_set": "Environmentally Friendly Products",
        "question_text_th": "โรงงานมีการการประเมินผลกระทบและวัฏจักรชีวิตของผลิตภัณฑ์ (life cycle impact assessment) หรือไม่?",
        "question_text_en": "Has the factory conducted a life cycle impact assessment of its products?"
    },
    {
        "id": "SET_E_5", "dimension": "E", "theme_set": "Environmentally Friendly Products",
        "question_text_th": "ร้อยละของยอดขายผลิตภัณฑ์ที่เป็นมิตรต่อสิ่งแวดล้อม (eco products) ต่อยอดขายผลิตภัณฑ์ทั้งหมด เท่าไหร่?",
        "question_text_en": "Does the factory report the percentage of sales from environmentally friendly products (eco products) compared to total product sales?"
    },
    {
        "id": "SET_E_6", "dimension": "E", "theme_set": "Biodiversity and Cessation of Deforestation",
        "question_text_th": "โรงงานมีนโยบายและแนวปฏิบัติเกี่ยวกับการอนุรักษ์ความหลากหลายทางชีวภาพและยุติการตัดไม้ทำาลายป่า โดยครอบคลุมกระบวนการดำาเนินธุรกิจและห่วงโซ่อุปทานของโรงงานหรือไม่?",
        "question_text_en": "Does the factory have a policy and guidelines regarding the conservation of biodiversity and cessation of deforestation, covering its business operations and supply chain?"
    },
    {
        "id": "SET_E_7", "dimension": "E", "theme_set": "Biodiversity and Cessation of Deforestation",
        "question_text_th": "โรงงานมีการการประเมินความเสี่ยงและผลกระทบต่อความหลากหลายทางชีวภาพจากการดำเนินธุรกิจ หรือไม่?",
        "question_text_en": "Has the factory assessed the risks and impacts on biodiversity resulting from its business operations?"
    },
    {
        "id": "SET_E_8", "dimension": "E", "theme_set": "Biodiversity and Cessation of Deforestation",
        "question_text_th": "จำนวนพื้นที่การดำเนินธุรกิจของโรงงานที่มีการอนุรักษ์ความหลากหลายทางชีวภาพ เป็นจำนวนเท่าใด(ตารางเมตร)?",
        "question_text_en": "Does the factory report the area of its business operations with biodiversity conservation efforts (in square meters)?"
    },
    {
        "id": "SET_E_9", "dimension": "E", "theme_set": "Biodiversity and Cessation of Deforestation",
        "question_text_th": "จำนวนพื้นที่ที่ได้รับการรักษาภายใต้การดูแลของโรงงาน เป็นจำนวนเท่าใด(ตารางเมตร)?",
        "question_text_en": "Does the factory report the area of land conserved or protected under its care (in square meters)?"
    },
    {
        "id": "SET_E_10", "dimension": "E", "theme_set": "Biodiversity and Cessation of Deforestation",
        "question_text_th": "โรงงานมีแผนงานหรือโครงการอนุรักษ์ความหลากหลายทางชีวภาพในการดำเนินธุรกิจ หรือไม่?",
        "question_text_en": "Does the factory have biodiversity conservation plans or projects in its business operations?"
    },
    {
        "id": "SET_E_11", "dimension": "E", "theme_set": "Biodiversity and Cessation of Deforestation",
        "question_text_th": "โรงงานมีแผนงานหรือโครงการอนุรักษ์พื้นที่ป่าในการดำเนินธุรกิจ หรือไม่?",
        "question_text_en": "Does the factory have forest conservation plans or projects in its business operations?"
    },
    {
        "id": "SET_E_12", "dimension": "E", "theme_set": "Environmentally Friendly Materials",
        "question_text_th": "ปริมาณน้ำหนักรวมของวัสดุทั้งหมด เป็นเท่าใด (กิโลกรัม)?",
        "question_text_en": "Does the factory report the total weight of all materials used (in kilograms), optionally classified by type (e.g., non-renewable, renewable)?"
    },
    {
        "id": "SET_E_13", "dimension": "E", "theme_set": "Environmentally Friendly Materials",
        "question_text_th": "ร้อยละของวัสดุรีไซเคิลที่นำกลับมาใช้ในการพัฒนาผลิตภัณฑ์ เท่าไหร่?",
        "question_text_en": "Does the factory report the percentage of recycled input materials used in its product development?"
    },
    {
        "id": "SET_E_14", "dimension": "E", "theme_set": "Environmentally Friendly Materials",
        "question_text_th": "ร้อยละของวัสดุจากของเหลือหมดอายุหรือเสื่อมคุณภาพ (reclaimed) และถูกนำกลับมาใช้ในการพัฒนาผลิตภัณฑ์ เท่าไหร่?",
        "question_text_en": "Does the factory report the percentage of reclaimed materials (from expired or deteriorated sources) reused in its product development?"
    },
    {
        "id": "SET_E_15", "dimension": "E", "theme_set": "Air Pollution",
        "question_text_th": "ปริมาณมลพิษทางอากาศจากการดำเนินธุรกิจ - Nitrogen oxide (NOₓ), Sulfur dioxide (SOₓ), Persistent organic pollutants (POP), Volatile organic compounds (VOC), Hazardous air pollutants (HAP), Particulate matter (PM), อื่น ๆ เป็นเท่าใด?",
        "question_text_en": "Does the factory report the volume of its air pollutants from business operations (e.g., NOₓ, SOₓ, POPs, VOCs, HAPs, PM, others)?"
    },
    {
        "id": "SET_E_16", "dimension": "E", "theme_set": "Climate Change Risks",
        "question_text_th": "โรงงานมีการการประเมินความเสี่ยงจากการเปลี่ยนแปลงสภาพภูมิอากาศ โดยอธิบายผลกระทบที่อาจส่งผลต่อการดำเนินธุรกิจ หรือไม่?",
        "question_text_en": "Has the factory conducted a climate change risk assessment, including an explanation of potential impacts on its business operations?"
    },
    {
        "id": "SET_E_17", "dimension": "E", "theme_set": "Climate Change Risks",
        "question_text_th": "โรงงานมีเป้าหมาย แผนงาน และมาตรการบรรเทาความเสี่ยงจากการเปลี่ยนแปลงสภาพภูมิอากาศ หรือไม่?",
        "question_text_en": "Does the factory have established goals, plans, and measures to mitigate climate change risks?"
    },
    {
        "id": "SET_S_1", "dimension": "S", "theme_set": "Local Employment",
        "question_text_th": "โรงงานมีนโยบายและแนวปฏิบัติเกี่ยวกับการจ้างแรงงานท้องถิ่น หรือไม่?",
        "question_text_en": "Does the factory have a policy and guidelines regarding local employment?"
    },
    {
        "id": "SET_S_2", "dimension": "S", "theme_set": "Local Employment",
        "question_text_th": "ร้อยละของพนักงานที่มาจากชุมชนท้องถิ่น เท่าไหร่?",
        "question_text_en": "Does the factory report the percentage of its employees from local communities?"
    },
    {
        "id": "SET_S_3", "dimension": "S", "theme_set": "Respecting Diversity and Equality",
        "question_text_th": "โรงงานมีนโยบายและแนวปฏิบัติเกี่ยวกับการเคารพความแตกต่างและความเสมอภาคภายในองค์กรและห่วงโซ่อุปทาน โดยไม่แบ่งแยกเพศ อายุ เชื้อชาติ ความพิการ ศาสนา หรืออื่น ๆ หรือไม่?",
        "question_text_en": "Does the factory have a policy and guidelines regarding respect for diversity and equality within the organization and its supply chain (e.g., non-discrimination based on gender, age, race, disability, religion, or other factors)?"
    },
    {
        "id": "SET_S_4", "dimension": "S", "theme_set": "Respecting Diversity and Equality",
        "question_text_th": "สถิติจำนวนพนักงานตามเพศและสัญชาติ เป็นจำนวนเท่าใด?",
        "question_text_en": "Does the factory disclose employee statistics categorized by gender and nationality?"
    },
    {
        "id": "SET_S_5", "dimension": "S", "theme_set": "Respecting Diversity and Equality",
        "question_text_th": "จำนวนเหตุการณ์หรือข้อร้องเรียนเกี่ยวกับการละเมิดสิทธิ ความเสมอภาค และการปฏิบัติต่อแรงงานอย่างไม่เป็นธรรม พร้อมมาตรการแก้ไขและเยียวยา เป็นจำนวนเท่าใด?",
        "question_text_en": "Does the factory report the number of incidents or complaints related to violations of rights, equality, and unfair labor treatment, along with corresponding remediation and mitigation measures?"
    },
    {
        "id": "SET_S_6", "dimension": "S", "theme_set": "Promotion of Female Workforce",
        "question_text_th": "โรงงานมีนโยบายและแนวปฏิบัติเกี่ยวกับการส่งเสริมแรงงานสตรีในสถานประกอบการอย่างเท่าเทียมกัน หรือไม่?",
        "question_text_en": "Does the factory have a policy and guidelines related to promoting equal opportunities for the female workforce in the workplace?"
    },
    {
        "id": "SET_S_7", "dimension": "S", "theme_set": "Promotion of Female Workforce",
        "question_text_th": "ข้อมูลพนักงานหญิงแบ่งตามตำแหน่งงานมีจำนวนเท่าใด?",
        "question_text_en": "Does the factory disclose the number of female employees categorized by employment level (e.g., senior management, management, staff level)?"
    },
    {
        "id": "SET_S_8", "dimension": "S", "theme_set": "Monitoring and Assessing Impacts on Communities",
        "question_text_th": "โรงงานมีการการติดตามและประเมินผลกระทบต่อชุมชนจากการดำเนินธุรกิจของโรงงาน หรือไม่?",
        "question_text_en": "Does the factory monitor and assess the impacts of its business operations on communities?"
    },
    {
        "id": "SET_S_9", "dimension": "S", "theme_set": "Monitoring and Assessing Impacts on Communities",
        "question_text_th": "จำนวนกรณีพิพาทหรือเหตุการณ์ร้องเรียนเกี่ยวกับการละเมิดสิทธิชุมชน พร้อมมาตรการแก้ไขและเยียวยา เป็นจำนวนเท่าใด?",
        "question_text_en": "Does the factory report the number of disputes or complaints regarding community rights violations, along with corresponding remediation and mitigation measures?"
    },
    {
        "id": "SET_G_1", "dimension": "G", "theme_set": "Cybersecurity and Personal Data Protection",
        "question_text_th": "โรงงานมีนโยบายและแนวปฏิบัติเกี่ยวกับด้านความปลอดภัยทางไซเบอร์และการป้องกันข้อมูลส่วนบุคคล หรือไม่?",
        "question_text_en": "Does the factory have a policy and guidelines on cybersecurity and personal data protection?"
    },
    {
        "id": "SET_G_2", "dimension": "G", "theme_set": "Cybersecurity and Personal Data Protection",
        "question_text_th": "ร้อยละของจำนวนโครงสร้างพื้นฐานด้านเทคโนโลยีที่ได้รับการรับรองมาตรฐานด้านความปลอดภัยทางไซเบอร์ เช่น ISO 27001 หรือมาตรฐานอื่น ๆ เป็นต้น เท่าไหร่?",
        "question_text_en": "Does the factory report the percentage of its technology infrastructures that have been certified with cybersecurity standards (e.g., ISO 27001 or other relevant standards)?"
    },
    {
        "id": "SET_G_3", "dimension": "G", "theme_set": "Cybersecurity and Personal Data Protection",
        "question_text_th": "โรงงานมีมาตรการและแนวปฏิบัติเกี่ยวกับการใช้ข้อมูลส่วนบุคคล หรือไม่?",
        "question_text_en": "Does the factory have established measures and guidelines related to personal data usage?"
    },
    {
        "id": "SET_G_4", "dimension": "G", "theme_set": "Cybersecurity and Personal Data Protection",
        "question_text_th": "ร้อยละของพนักงานที่ได้รับการอบรมด้านความปลอดภัยทางไซเบอร์และการใช้ข้อมูลส่วนบุคคล เท่าไหร่?",
        "question_text_en": "Does the factory report the percentage of its employees who have received training in cybersecurity and personal data usage?"
    },
    {
        "id": "SET_G_5", "dimension": "G", "theme_set": "Cybersecurity and Personal Data Protection",
        "question_text_th": "จำนวนเหตุการณ์หรือกรณีที่โรงงานถูกโจมตีทางไซเบอร์ พร้อมมาตรการแก้ไข เป็นจำนวนเท่าใด?",
        "question_text_en": "Does the factory report the number of incidents or cases of cyberattacks against the factory, along with corresponding mitigation measures?"
    },
    {
        "id": "SET_G_6", "dimension": "G", "theme_set": "Cybersecurity and Personal Data Protection",
        "question_text_th": "จำนวนเหตุการณ์หรือกรณีข้อมูลส่วนบุคคลรั่วไหล พร้อมมาตรการแก้ไข เป็นจำนวนเท่าใด?",
        "question_text_en": "Does the factory report the number of incidents or cases of personal data breaches, along with corresponding mitigation measures?"
    },
    {
        "id": "SET_G_7", "dimension": "G", "theme_set": "Product Quality and Recall",
        "question_text_th": "โรงงานมีนโยบายและแนวปฏิบัติเกี่ยวกับการจัดการด้านคุณภาพของผลิตภัณฑ์ ตามมาตรฐานสากล เช่น ISO 9001:2015 หรือมาตรฐานอื่น ๆ เป็นต้น หรือไม่?",
        "question_text_en": "Does the factory have a policy and guidelines for product quality management according to international standards (e.g., ISO 9001:2015 or other standards)?"
    },
    {
        "id": "SET_G_8", "dimension": "G", "theme_set": "Product Quality and Recall",
        "question_text_th": "โรงงานมีแผนการเรียกคืนผลิตภัณฑ์ หรือไม่?",
        "question_text_en": "Does the factory have a product recall plan?"
    },
    {
        "id": "SET_G_9", "dimension": "G", "theme_set": "Product Quality and Recall",
        "question_text_th": "จำนวนกรณีหรือเหตุการณ์เรียกคืนผลิตภัณฑ์ พร้อมมาตรการแก้ไขและเยียวยา เป็นจำนวนเท่าใด?",
        "question_text_en": "Does the factory report the number of cases or incidents of product recall, along with corresponding remediation and mitigation measures?"
    }
]