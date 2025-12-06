# 🏠 AI Home Inspection System

ระบบวิเคราะห์ภาพบ้านอัตโนมัติด้วย AI – ช่วยแยกวัตถุ ประเมินวัสดุ ตรวจจับความเสียหาย และสร้างรายงานแบบครบวงจร

---

## 📌 ความเป็นมาของโปรเจกต์ (Background)

ปัจจุบันการตรวจสอบบ้าน (Home Inspection) ยังทำด้วยวิธีแบบ manual ซึ่งใช้เวลามาก และต้องอาศัยผู้เชี่ยวชาญ ประกอบกับราคาวัสดุและค่าแรงสูงขึ้น การประเมินราคารวบยอดจึงมักคลาดเคลื่อน
โปรเจกต์นี้ถูกสร้างขึ้นเพื่อให้ **ผู้ใช้สามารถอัปโหลดภาพห้อง แล้วให้ AI วิเคราะห์พื้นที่ วัสดุ และปัญหาที่อาจเกิดขึ้นได้แบบอัตโนมัติ** โดยใช้โมเดล Vision ระดับ State-of-the-art

---

## 🎯 วัตถุประสงค์ของโปรเจกต์ (Project Objectives)

* 🔍 สร้างระบบ **Segmentation** ของวัตถุในบ้านแบบคลิกเลือกพื้นที่
* ⚒️ ประเมิน **พื้นที่** และ **ค่าใช้จ่ายวัสดุ** เช่น สี กระเบื้อง ไม้
* 🧩 ตรวจจับรอยเสียหายพื้นฐาน เช่น
  * รอยแตกร้าว
  * คราบน้ำ
  * รอยลอก
* 📊 สร้างรายงาน Home Inspection แบบเต็มรูปแบบ
* 📐 รองรับการวัดประสิทธิภาพด้วย Ground Truth mask (Pixel-Level Evaluation)

---

## 🧠 โมเดลและเทคโนโลยีที่ใช้ (Models & Libraries)

### 📌 Core AI Model

| Component                                         | Description                                                                 |
| ------------------------------------------------- | --------------------------------------------------------------------------- |
| **Segment Anything Model (SAM – ViT-H)**          | โมเดล segmentation จาก Meta AI ใช้สำหรับแบ่งพื้นที่ในภาพตามจุดที่ผู้ใช้คลิก |
| **Custom Damage Detection (optional/extendable)** | สามารถเพิ่มโมเดล CNN/Transformer เพื่อตรวจจับความเสียหายในอนาคต             |

### 📌 Python Libraries

* `streamlit` — UI
* `segment_anything` — SAM model
* `numpy`, `pandas` — คำนวณข้อมูล
* `PIL` — จัดการรูปภาพ
* `opencv-python` — preprocessing
* `matplotlib` — visualization
* `streamlit_image_coordinates` — สำหรับคลิกบนภาพ
* `json`, `datetime`, `BytesIO` — export รายงาน

---

## 📂 โครงสร้างโปรเจกต์

```
project/
│── app.py
│── models/
│     └── sam_vit_h_4b8939.pth
│── requirements.txt
│── README.md
```

---

## 🚀 วิธีใช้งาน (How to Use)

### 1️⃣ ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ ดาวน์โหลด SAM checkpoint

(SAM ViT-H – 2.6GB) แล้ววางใน `/models/sam_vit_h_4b8939.pth`

### 3️⃣ รันแอป

```bash
streamlit run app.py
```

### 4️⃣ วิธีใช้งานบนแอป

#### ✔ Tab 1 – Segmentation

1. อัปโหลดภาพห้อง
2. คลิกตำแหน่งบนภาพเพื่อเลือกจุดที่ต้องการ segment
3. กด **Run Segmentation**
4. เลือกดู mask, overlay และผลลัพธ์ metrics

#### ✔ Pixel-level Evaluation (GT Mask)

* อัปโหลด Ground Truth mask (white=object, black=background)
* ระบบคำนวณ:

  * TP / FP / FN / TN
  * Accuracy
  * Precision
  * Recall
  * F1-score
* แสดง Error Map (สีเขียว/แดง/เหลือง)

#### ✔ Tab 2 – Material Estimation

* เลือกวัสดุ เช่น สี กระเบื้อง ไม้
* ระบบคำนวณ:

  * พื้นที่ m²
  * ราคาวัสดุ
  * ปริมาณวัสดุที่ต้องใช้

#### ✔ Tab 3 – Damage Detection

(สามารถเพิ่ม Custom Model ได้)

#### ✔ Tab 4 – Full Report

* ดูข้อมูลสรุปทั้งหมด
* ดาวน์โหลด JSON / Mask / Metrics

---

## 📐 เกณฑ์วัดประสิทธิภาพ (Evaluation Metrics)

### 1️⃣ Pixel-Level Confusion Matrix

| ค่า    | ความหมาย                               |
| ------ | -------------------------------------- |
| **TP** | พื้นที่ที่โมเดลทำนายถูกว่าเป็นวัตถุ    |
| **TN** | พื้นที่ที่โมเดลทำนายถูกว่าเป็นพื้นหลัง |
| **FP** | ทำนายว่าเป็นวัตถุ แต่จริงเป็นพื้นหลัง  |
| **FN** | ทำนายว่าเป็นพื้นหลัง แต่จริงเป็นวัตถุ  |

### 2️⃣ Metrics ที่ใช้ประเมิน

* **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
* **Precision** = TP / (TP + FP)
* **Recall** = TP / (TP + FN)
* **F1-score** = 2 × Precision × Recall / (Precision + Recall)

### 3️⃣ Error Map Visualization

ระบบแสดงผลสีเพื่อเข้าใจความผิดพลาดง่ายขึ้น

* 🟩 **Green = True Positive (ถูก)**
* 🟥 **Red = False Positive (เกินจริง)**
* 🟨 **Yellow = False Negative (ขาดบางส่วน)**

---

## 🧩 จุดเด่นของระบบนี้

* ใช้ SAM segmentation ระดับ SOTA
* มี UI แบบ interactive
* ประเมินวัสดุได้ทันทีจาก area
* รองรับการตรวจสอบความแม่นยำด้วย Ground Truth
* สามารถ export รายงานได้

---

## 📌 สิ่งที่สามารถพัฒนาเพิ่มเติม

* Damage detection ด้วย CNN/ResNet/ViT
* การประเมินค่าแรงตามพื้นที่
* ระบบวิเคราะห์หลายรูปพร้อมกัน
* ส่งออกเป็น PDF Report
* เชื่อมต่อฐานข้อมูลเพื่อเก็บประวัติการตรวจบ้าน

---

## 📞 Contact

0626560205 
id line : 123ohiop
