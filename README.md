🏠 AI Home Inspection System

Automated Room Segmentation | Material Estimation | Damage Detection | Evaluation System

📌 1. ความเป็นมาและวัตถุประสงค์ของโปรเจกต์
ปัจจุบันงานตรวจสอบบ้าน (Home Inspection) ยังทำโดยมนุษย์เป็นหลัก เช่น
ตรวจสภาพผนัง พื้น เพดาน
ประเมินพื้นที่สำหรับทาสี ปูกระเบื้อง
ดูรอยแตกร้าว/ความเสียหาย
คำนวณค่าใช้จ่ายซ่อมแซม
สิ่งเหล่านี้ใช้เวลาและขึ้นกับความชำนาญของผู้ตรวจสอบ
โปรเจกต์นี้จึงถูกพัฒนาเพื่อ:
ใช้ AI ช่วยวิเคราะห์ภาพถ่ายห้อง
แยกวัตถุในภาพด้วย Segment Anything Model (SAM)
ประมาณพื้นที่จริงด้วย Pixel-to-Meter
ประเมินวัสดุและค่าใช้จ่าย
ตรวจสอบความเสียหายของผนังหรือพื้น
ทำ Pixel-level Evaluation เทียบกับ Ground Truth

📌 2. ฟีเจอร์หลักของระบบ
✔ 1. Image Segmentation
คลิกบนภาพเพื่อเลือกจุดที่ต้องการ segment
ใช้ SAM เพื่อแยกวัตถุ เช่น ผนัง ประตู หน้าต่าง พื้น
แสดง Mask, Overlay, Pixel Count, Coverage

✔ 2. Material Estimation
ใช้พื้นที่ของ Mask เพื่อคำนวณ:
ปริมาณสี
จำนวนกระเบื้อง
พื้นไม้
ค่าวัสดุรวมตามราคาใน Sidebar
✔ 3. Damage Detection
ตรวจรอยร้าว/คราบ/เสียหายบนผนังด้วย Algorithm (OpenCV/ML)
✔ 4. Pixel-level Evaluation
อัปโหลด Ground Truth mask
ระบบปรับขนาดอัตโนมัติ
คำนวณ Confusion Matrix + Accuracy + Precision + Recall + F1
แสดง Error Map (TP/FP/FN) ชัดเจน
Export รายงานเป็น JSON

📌 3. โครงสร้างการทำงานของระบบ (Pipeline)
1. ผู้ใช้ Upload รูปภาพห้อง  
2. ผู้ใช้คลิกบริเวณที่ต้องการ Segment  
3. ระบบใช้ SAM สร้าง Mask  
4. นำ Mask → คำนวณพื้นที่ (pixel → m²)  
5. นำพื้นที่ → คำนวณวัสดุ & ราคา  
6. (ออปชัน) อัปโหลด GT Mask → ประเมินผลลัพธ์  
7. แสดงผล + อนุญาตให้ดาวน์โหลดรายงาน JSON  

📌 4. Libraries และ Packages ที่ใช้
| Library                 | ใช้ทำอะไร                                |
| ----------------------- | ---------------------------------------- |
| **streamlit**           | UI/Frontend สำหรับ web app               |
| **numpy**               | ประมวลผลอาร์เรย์และ mask                 |
| **Pillow (PIL)**        | โหลด/แปลง/จัดการภาพ                      |
| **opencv-python (cv2)** | Image processing และงาน damage detection |
| **matplotlib**          | Visualization ของ mask และ error map     |
| **pandas**              | สร้าง confusion matrix                   |
| **json**                | Export รายงานผล                          |
| **BytesIO**             | บันทึกไฟล์ภาพให้ดาวน์โหลด                |

📌 5. โมเดลที่ใช้: SAM (Segment Anything Model)
เวอร์ชัน: vit_h
Checkpoint: sam_vit_h_4b8939.pth
จุดเด่น:
Segment อะไรก็ได้จาก point prompt
แม่นยำสูง
รองรับหลาย mask ต่อจุดเดียว
📌 6. การวัดประสิทธิภาพระบบ Segmentation
ระบบรองรับการอัปโหลด Ground Truth mask เพื่อคำนวณเมตริกต่าง ๆ

✔ Confusion Matrix (Pixel-level)
Terms	ความหมาย
TP	Pixel ที่เป็นวัตถุ และทำนายถูก
TN	Pixel ที่เป็นฉากหลัง และทำนายถูก
FP	Pixel ที่ทำนายเป็นวัตถุ แต่จริง ๆ คือฉากหลัง
FN	Pixel ที่เป็นวัตถุ แต่ทำนายพลาด
✔ Metrics ที่คำนวณได้

Accuracy = (TP + TN) / Total
Precision = TP / (TP + FP)
Recall (Sensitivity) = TP / (TP + FN)
F1-score = 2PR / (P + R)

✔ Error Map Visualization
สีเขียว → TP
สีแดง → FP
สีเหลือง → FN
สีเทา → TN

📌 7. โครงสร้างไฟล์ที่แนะนำ
project/
│ app.py
│ README.md
│ requirements.txt
│
└── models/
    └── sam_vit_h_4b8939.pth

📌 8. วิธีใช้งาน (Setup)
1) ติดตั้งไลบรารี
pip install streamlit numpy pillow opencv-python matplotlib pandas segment-anything streamlit-image-coordinates
2) ดาวน์โหลดโมเดล SAM vit_h
ดาวน์โหลดจาก Meta GitHub
แล้ววางไว้ที่:
models/sam_vit_h_4b8939.pth
3) รันแอป
streamlit run app.py
