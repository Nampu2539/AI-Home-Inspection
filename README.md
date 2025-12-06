🏠 AI Home Inspection System

ระบบตรวจสอบบ้านด้วย AI – วัดพื้นที่, ประมาณราคาวัสดุ, และตรวจหารอยเสียหาย

โปรเจกต์นี้เป็นเว็บแอปพลิเคชันที่พัฒนาโดย Streamlit และใช้โมเดลจาก Segment Anything (SAM)
เพื่อช่วยผู้ใช้งานทำการวิเคราะห์ภาพของห้องหรืออาคาร เช่น
✔ ตีเส้นแบ่งวัตถุ (Segmentation)
✔ ประมาณการวัสดุก่อสร้างที่ต้องใช้
✔ ประเมินราคา
✔ ตรวจหารอยแตก / คราบ / เชื้อรา
✔ สร้างรายงานสรุปผล

🚀 Features
1. 🎯 Object Segmentation

ผู้ใช้สามารถ คลิกบนภาพ เพื่อเลือกตำแหน่งที่ต้องการให้ SAM ทำการ Segment

ระบบรองรับการเลือกหลายจุด

แสดงผลทั้ง Binary Mask, Overlay, และ Segmentation Metrics

สามารถดาวน์โหลด Mask ออกเป็น PNG ได้

2. 📏 Material Estimation

ระบบประมาณราคาวัสดุก่อสร้าง เช่น

สี

กระเบื้อง

ไม้ปูพื้น

วอลเปเปอร์

รองรับการคำนวณ:

พื้นที่ใช้งาน (m²)

ปริมาณวัสดุที่ต้องใช้

การเผื่อของเสีย 10–15%

ราคาที่ผู้ใช้กำหนดเอง

3. 🔍 Damage Detection

ตรวจสอบรอยเสียหายพื้นฐานที่อาจพบในบ้าน เช่น

รอยแตก (Cracks)

คราบ (Stains)

เชื้อรา (Mold)

สีลอก (Peeling)

สนิม (Rust)

การตรวจใช้เทคนิค Computer Vision แบบ Heuristic เช่น

Edge Detection

Thresholding

Contour Analysis

4. 📊 Full Inspection Report

ระบบจะบันทึกข้อมูลจากแต่ละ Tab

แสดงข้อมูลสรุปพื้นที่ / วัสดุ / ค่าใช้จ่าย / ความเสียหาย

รองรับการ export ออกมาเก็บไว้ภายนอก (optional)

📦 Technology Stack
Component	Description
Streamlit	ใช้สร้างเว็บ UI
Segment Anything (SAM)	ใช้สร้างระบบ segmentation
OpenCV	ใช้ตรวจวิเคราะห์ความเสียหาย
NumPy	ประมวลผลภาพ
Pillow (PIL)	จัดการไฟล์รูปภาพ
Python 3.9+	ภาษาในการพัฒนาระบบ
📁 Project Structure (โครงสร้างไฟล์)
project/
│── app.py                 # Main Streamlit app
│── models/
│     └── sam_vit_h_4b8939.pth   # SAM checkpoint
│── README.md
│── requirements.txt

🛠 Installation
1. Clone project
git clone <your-repo-url>
cd project

2. Install dependencies
pip install -r requirements.txt

3. Download SAM model

ดาวน์โหลดไฟล์ sam_vit_h_4b8939.pth และวางในโฟลเดอร์ models/

ดาวน์โหลดได้จาก:
https://github.com/facebookresearch/segment-anything

4. Run Streamlit
streamlit run app.py

⚙️ Key Configurations
Sidebar Settings

ประเภทห้อง

ราคาวัสดุ

ค่าคาลิเบรต Pixel → Meter

ปุ่มรีเซ็ตข้อมูล

ปุ่มโหลดโมเดล SAM

SAM Model Loading
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
sam.to(device)

🧠 How It Works (Flow การทำงาน)

1.ผู้ใช้อัปโหลดภาพ

2.ระบบแสดงตัวเลือกให้ผู้ใช้คลิกจุดสำหรับ segmentation

3.ผู้ใช้กด “Run Segmentation” → SAM สร้าง masks

4.ผู้ใช้เลือกวัสดุ → ระบบคำนวณราคา

5.ผู้ใช้กดตรวจความเสียหาย → ระบบประมวลผลภาพ

6.ผู้ใช้ไปที่ “Full Report” เพื่อดูข้อมูลสรุปทั้งหมด

🧩 Future Improvements

เพิ่มโมเดล Deep Learning สำหรับ Damage Detection จริง ๆ


