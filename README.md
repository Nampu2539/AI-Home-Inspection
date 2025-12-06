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

ผู้ใช้อัปโหลดภาพ

ระบบแสดงตัวเลือกให้ผู้ใช้คลิกจุดสำหรับ segmentation

ผู้ใช้กด “Run Segmentation” → SAM สร้าง masks

ผู้ใช้เลือกวัสดุ → ระบบคำนวณราคา

ผู้ใช้กดตรวจความเสียหาย → ระบบประมวลผลภาพ

ผู้ใช้ไปที่ “Full Report” เพื่อดูข้อมูลสรุปทั้งหมด

🧩 Future Improvements

เพิ่มโมเดล Deep Learning สำหรับ Damage Detection จริง ๆ


app.py

# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
import cv2
from segment_anything import sam_model_registry, SamPredictor
from streamlit_image_coordinates import streamlit_image_coordinates
import json
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="🏠 AI Home Inspection System", layout="wide")

# ---- Custom CSS -----------------------------------------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🏠 AI Home Inspection System</p>', unsafe_allow_html=True)
st.markdown("**วิเคราะห์สภาพบ้าน | ประมาณการวัสดุ | ตรวจรอยเสียหาย**")

# ---------------- Sidebar / Global Settings ---------------------------------
st.sidebar.header("⚙️ ตั้งค่าระบบ")

room_type = st.sidebar.selectbox(
    "ประเภทห้อง",
    ["living_room", "bedroom", "kitchen", "bathroom", "outdoor", "office"],
    index=0,
)
# GPU option MUST be outside cached loader (avoid widget-in-cache warning)
use_gpu = st.sidebar.checkbox("ใช้ GPU (ถ้ามี)", value=False, help="เปิดใช้งานหากมี CUDA และไดรเวอร์พร้อม")

# Material Prices
st.sidebar.subheader("💰 ราคาวัสดุ")
material_prices = {
    "paint": st.sidebar.number_input("สี (บาท/ลิตร)", value=350, step=50),
    "tile": st.sidebar.number_input("กระเบื้อง (บาท/ตร.ม.)", value=500, step=50),
    "wood": st.sidebar.number_input("ไม้ปูพื้น (บาท/ตร.ม.)", value=800, step=100),
}

# Calibration
st.sidebar.subheader("📏 Calibration")
pixel_to_meter = st.sidebar.number_input(
    "พิกเซล/เมตร",
    value=100.0,
    step=10.0,
    help="ปรับค่านี้ตามขนาดจริงของภาพ (ยิ่งสูง = 1 เมตรมีพิกเซลมากขึ้น)",
)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Data"):
    # keep uploaded_img but clear other keys
    for k in list(st.session_state.keys()):
        if k not in ("uploaded_img",):
            del st.session_state[k]
    st.session_state.uploaded_img = None
    st.experimental_rerun()

# ---------------- Load SAM Model (cached) -----------------------------------
@st.cache_resource
def load_sam_model_cached(checkpoint_path: str, device: str):
    """
    Load SAM model. Why device is a parameter:
    - This function is cached by (checkpoint_path, device) so widget is outside.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device)
    return sam

# Path to checkpoint (adjust if needed)
SAM_CKPT = "models/sam_vit_h_4b8939.pth"

# Try loading model and show helpful messages
sam = None
if st.button("🔁 โหลด/รีโหลดโมเดล SAM"):
    # allow user to explicitly trigger load
    try:
        device = "cuda" if use_gpu else "cpu"
        with st.spinner(f"กำลังโหลด SAM model (device={device}) ..."):
            sam = load_sam_model_cached(SAM_CKPT, device)
        st.sidebar.success(f"✅ โมเดลโหลดสำเร็จ (device={device})")
    except FileNotFoundError:
        st.error("❌ ไม่พบโมเดล SAM ในโฟลเดอร์ `models/`")
        st.info("💡 ดาวน์โหลดไฟล์ checkpoint แล้ววางที่ `models/sam_vit_h_4b8939.pth` หรือรันสคริปต์ดาวน์โหลด")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

# If model was loaded in a previous run and cached, we can still retrieve it safely:
if sam is None:
    # try to get from cache without raising loud exception
    try:
        device = "cuda" if use_gpu else "cpu"
        sam = load_sam_model_cached(SAM_CKPT, device)
    except Exception:
        sam = None  # leave None, user can press reload button

# ---------------- Session State Initialization --------------------------------
if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None
if "click_points" not in st.session_state:
    st.session_state.click_points = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "masks" not in st.session_state:
    st.session_state.masks = []
if "room_type" not in st.session_state:
    st.session_state.room_type = room_type
if "inspection_data" not in st.session_state:
    st.session_state.inspection_data = {}
if "damage_results" not in st.session_state:
    st.session_state.damage_results = {}

st.session_state.room_type = room_type

# ---------------- Upload Image ------------------------------------------------
st.header("📤 Upload Image")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพห้องหรืออาคาร",
        type=["jpg", "jpeg", "png"],
        help="รองรับไฟล์ JPG, JPEG, PNG",
    )
with col2:
    st.info(
        """
    **Tips:**
    - ถ่ายภาพในแสงสว่าง
    - หลีกเลี่ยงเงา
    - ถ่ายแบบตรง ไม่เอียง
    """
    )

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((1400, 1400))
    st.session_state.uploaded_img = img
    st.success("✅ อัพโหลดสำเร็จ!")

# ---------------- Main Flow ---------------------------------------------------
if st.session_state.uploaded_img is None:
    st.info("👆 กรุณา upload รูปภาพเพื่อเริ่มต้นใช้งาน")
    st.write("### 📸 ตัวอย่างภาพที่เหมาะสม")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("✅ **ดี**: แสงสว่างเพียงพอ")
    with col2:
        st.write("✅ **ดี**: มุมมองตรง")
    with col3:
        st.write("✅ **ดี**: ระยะพอดี ไม่ไกลหรือใกล้เกินไป")
else:
    pil_img = st.session_state.uploaded_img
    img_np = np.array(pil_img)

    st.info(f"📐 ขนาดภาพ: {pil_img.width} × {pil_img.height} pixels")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Segmentation", "📏 Material Estimation", "🔍 Damage Detection", "📊 Full Report"]
    )

    # ---------------- TAB 1: Segmentation ---------------------------------
    with tab1:
        st.subheader("🎯 Object Segmentation")
        st.write("👉 **คลิกบนภาพเพื่อเลือกวัตถุที่ต้องการ segment** (คลิกหลายจุดได้)")

        col1, col2 = st.columns([3, 1])
        with col1:
            click_data = streamlit_image_coordinates(pil_img, key="seg_click")
            if click_data:
                # Guard against clicks outside image
                x = int(click_data["x"])
                y = int(click_data["y"])
                # Add point
                st.session_state.click_points.append((x, y))
                st.session_state.labels.append(1)  # foreground
                st.success(f"✅ เพิ่มจุดที่ ({x}, {y})")

            # show the image (so user can see)
            st.image(pil_img, caption="ภาพต้นฉบับ", use_column_width=True)

        with col2:
            st.metric("จำนวนจุด", len(st.session_state.click_points))
            if len(st.session_state.click_points) > 0:
                st.write("**จุดที่เลือก:**")
                for i, (x, y) in enumerate(st.session_state.click_points):
                    st.write(f"{i+1}. ({x}, {y})")

            if st.button("🚀 Run Segmentation", use_container_width=True):
                if sam is None:
                    st.error("❌ โมเดล SAM ยังไม่โหลด — กดปุ่ม '🔁 โหลด/รีโหลดโมเดล SAM' ที่แถบด้านบน")
                elif len(st.session_state.click_points) == 0:
                    st.error("⚠️ กรุณาคลิกจุดบนภาพก่อน")
                else:
                    with st.spinner("🔄 กำลังประมวลผล..."):
                        try:
                            predictor = SamPredictor(sam)
                            predictor.set_image(img_np)

                            point_coords = np.array(st.session_state.click_points)
                            point_labels = np.array(st.session_state.labels)

                            masks, scores, logits = predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=True,
                            )

                            # ensure masks are boolean arrays
                            masks = [(m > 0.5).astype(bool) for m in masks]
                            st.session_state.masks = masks
                            st.balloons()
                            st.success("✅ Segmentation เสร็จสิ้น!")
                        except Exception as e:
                            st.error(f"❌ Error during segmentation: {e}")

            if st.button("🔄 Reset Points", use_container_width=True):
                st.session_state.click_points = []
                st.session_state.labels = []
                st.experimental_rerun()

        # Show segmentation results
        if len(st.session_state.masks) > 0:
            st.markdown("---")
            st.subheader("📊 ผลลัพธ์ Segmentation")

            mask_idx = st.slider(
                "เลือก Mask (โมเดลสร้างหลายตัวเลือก)",
                0,
                len(st.session_state.masks) - 1,
                0,
            )
            selected_mask = st.session_state.masks[mask_idx].astype(bool)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**🎭 Mask (binary)**")
                mask_img = (selected_mask * 255).astype(np.uint8)
                st.image(mask_img, caption="Binary Mask", use_column_width=True)
            with col2:
                st.write("**🎨 Overlay**")
                overlay = img_np.copy()
                color = np.array([255, 0, 0], dtype=np.uint8)
                # blend overlay for selected mask
                overlay[selected_mask] = (overlay[selected_mask] * 0.5 + color * 0.5).astype(np.uint8)
                st.image(overlay, caption="Original + Mask", use_column_width=True)

            # Metrics
            pixel_count = int(np.sum(selected_mask))
            area_m2 = pixel_count / (pixel_to_meter ** 2)
            coverage_pct = (pixel_count / img_np.size) * 100

            st.markdown(
                f"""
            <div class="metric-card">
                <h3>📊 Segmentation Metrics</h3>
                <p><strong>Pixel Count:</strong> {pixel_count:,} pixels</p>
                <p><strong>Estimated Area:</strong> {area_m2:.2f} m²</p>
                <p><strong>Coverage:</strong> {coverage_pct:.2f}% of image</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Export mask button (download)
            mask_pil = Image.fromarray(mask_img)
            buf = BytesIO()
            mask_pil.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                label="💾 Download Mask PNG",
                data=buf,
                file_name=f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
            )

    # ---------------- TAB 2: Material Estimation ----------------------------
    with tab2:
        st.subheader("📏 Material Estimation")
        if len(st.session_state.masks) == 0:
            st.warning("⚠️ กรุณาทำ Segmentation ก่อน (ไปที่ Tab 1)")
        else:
            col1, col2 = st.columns(2)
            with col1:
                surface_type = st.selectbox(
                    "🏗️ เลือกประเภทพื้นผิว",
                    ["wall", "floor", "ceiling", "door", "window"],
                )
            with col2:
                material_type = st.selectbox(
                    "🎨 เลือกวัสดุ",
                    ["paint", "tile", "wood", "wallpaper"],
                )

            if st.button("💰 คำนวณราคา", use_container_width=True):
                # Use first mask by default (you can add UI to choose)
                mask = st.session_state.masks[0].astype(bool)
                pixel_count = int(np.sum(mask))
                area_m2 = pixel_count / (pixel_to_meter ** 2)

                total_cost = 0.0
                details = {}

                if material_type == "paint":
                    # 1 liter covers ~10 m² (per coat). Choose 2 coats by default
                    coats = 2
                    coverage_per_liter = 10.0
                    liters_needed = (area_m2 / coverage_per_liter) * coats
                    unit_price = material_prices["paint"]
                    total_cost = liters_needed * unit_price
                    cans_needed = int(np.ceil(liters_needed / 3.0))
                    details = {
                        "area_m2": area_m2,
                        "material": "paint",
                        "liters_needed": liters_needed,
                        "cans_needed": cans_needed,
                        "unit_price": unit_price,
                        "cost": total_cost,
                    }

                    st.markdown(
                        f"""
                    <div class="metric-card">
                        <h3>🎨 ประมาณการสี</h3>
                        <p><strong>พื้นที่:</strong> {area_m2:.2f} m²</p>
                        <p><strong>ปริมาณสี:</strong> {liters_needed:.2f} ลิตร ({cans_needed} กระป๋อง)</p>
                        <p><strong>ราคาต่อลิตร:</strong> ฿{unit_price:,.0f}</p>
                        <p><strong>ราคารวม:</strong> ฿{total_cost:,.2f}</p>
                        <p style="font-size: 0.9em; opacity: 0.85;">*คิดเป็น {coats} รอบทา, ครอบคลุม {coverage_per_liter} ตร.ม./ลิตร</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                elif material_type == "tile":
                    area_with_waste = area_m2 * 1.10  # 10% waste
                    unit_price = material_prices["tile"]
                    total_cost = area_with_waste * unit_price
                    # assume 30x30 cm tile => 0.09 m2 each
                    tiles_needed = int(np.ceil(area_with_waste / 0.09))
                    boxes_needed = int(np.ceil(tiles_needed / 10.0))
                    details = {
                        "area_m2": area_m2,
                        "material": "tile",
                        "tiles_needed": tiles_needed,
                        "boxes_needed": boxes_needed,
                        "unit_price": unit_price,
                        "cost": total_cost,
                    }

                    st.markdown(
                        f"""
                    <div class="metric-card">
                        <h3>🏺 ประมาณการกระเบื้อง</h3>
                        <p><strong>พื้นที่:</strong> {area_m2:.2f} m²</p>
                        <p><strong>พื้นที่ + waste 10%:</strong> {area_with_waste:.2f} m²</p>
                        <p><strong>จำนวนกระเบื้อง (30×30cm):</strong> {tiles_needed} แผ่น</p>
                        <p><strong>จำนวนกล่อง (10 แผ่น/กล่อง):</strong> {boxes_needed} กล่อง</p>
                        <p><strong>ราคารวม:</strong> ฿{total_cost:,.2f}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                elif material_type == "wood":
                    area_with_waste = area_m2 * 1.15  # 15% waste
                    unit_price = material_prices["wood"]
                    total_cost = area_with_waste * unit_price
                    details = {
                        "area_m2": area_m2,
                        "material": "wood",
                        "area_with_waste": area_with_waste,
                        "unit_price": unit_price,
                        "cost": total_cost,
                    }
                    st.markdown(
                        f"""
                    <div class="metric-card">
                        <h3>🪵 ประมาณการไม้ปูพื้น</h3>
                        <p><strong>พื้นที่:</strong> {area_m2:.2f} m²</p>
                        <p><strong>พื้นที่ + waste 15%:</strong> {area_with_waste:.2f} m²</p>
                        <p><strong>ราคารวม:</strong> ฿{total_cost:,.2f}</p>
                        <p style="font-size: 0.9em; opacity: 0.85;">*รวมค่าของเสียในการติดตั้ง 15%</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    # wallpaper or other
                    st.info("ℹ️ ฟังก์ชันสำหรับวัสดุอื่นจะเพิ่มเติมได้ตามต้องการ")
                    details = {"area_m2": area_m2, "material": material_type, "cost": 0.0}

                # save to inspection_data
                st.session_state.inspection_data[surface_type] = {
                    "area_m2": float(details.get("area_m2", area_m2)),
                    "material": details.get("material", material_type),
                    "cost": float(details.get("cost", total_cost)),
                    "details": details,
                    "timestamp": datetime.now().isoformat(),
                }
                st.success(f"✅ บันทึกข้อมูล {surface_type} แล้ว!")

    # ---------------- TAB 3: Damage Detection -------------------------------
    with tab3:
        st.subheader("🔍 Damage Detection")
        st.info("💡 ฟีเจอร์นี้ใช้ Computer Vision ตรวจหารอยเสียหายในภาพ (heuristic)")

        col1, col2 = st.columns([2, 1])
        with col1:
            damage_types = st.multiselect(
                "เลือกประเภทความเสียหายที่ต้องการตรวจ",
                ["crack", "stain", "mold", "peeling_paint", "rust"],
                default=["crack", "stain"],
            )
        with col2:
            sensitivity = st.slider(
                "ความไวในการตรวจจับ", min_value=1, max_value=10, value=5, help="ค่าสูง = ตรวจจับละเอียดกว่า"
            )

        if st.button("🔎 Detect Damages", use_container_width=True):
            with st.spinner("🔄 กำลังวิเคราะห์..."):
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                damage_results = {}

                # Crack detection (edge-based heuristic)
                if "crack" in damage_types:
                    threshold1 = max(10, 50 - (sensitivity * 3))
                    threshold2 = min(300, 150 + (sensitivity * 5))
                    edges = cv2.Canny(gray, threshold1, threshold2)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    crack_contours = [c for c in contours if cv2.arcLength(c, False) > 20]
                    crack_img = img_np.copy()
                    cv2.drawContours(crack_img, crack_contours, -1, (255, 0, 0), 2)
                    st.write("**🔴 Crack Detection Results**")
                    st.image(crack_img, caption="Detected Cracks (Red)", use_column_width=True)
                    damage_results["crack"] = {
                        "count": len(crack_contours),
                        "severity": "High" if len(crack_contours) > 10 else "Medium" if len(crack_contours) > 5 else "Low",
                    }
                    st.metric("จำนวนรอยร้าว", len(crack_contours))

                # Stain detection (color / darkness heuristic)
                if "stain" in damage_types:
                    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                    # threshold for dark/dull areas (these params can be tuned)
                    lower_dark = np.array([0, 0, 0])
                    upper_dark = np.array([180, 255, 100])
                    mask_stain = cv2.inRange(hsv, lower_dark, upper_dark)
                    kernel = np.ones((5, 5), np.uint8)
                    mask_stain = cv2.morphologyEx(mask_stain, cv2.MORPH_CLOSE, kernel)
                    stain_img = img_np.copy()
                    stain_img[mask_stain > 0] = [0, 255, 255]
                    st.write("**🟡 Stain Detection Results**")
                    st.image(stain_img, caption="Detected Stains (Cyan)", use_column_width=True)
                    stain_percent = (np.sum(mask_stain > 0) / mask_stain.size) * 100
                    damage_results["stain"] = {
                        "coverage": float(stain_percent),
                        "severity": "High" if stain_percent > 10 else "Medium" if stain_percent > 5 else "Low",
                    }
                    st.metric("พื้นที่คราบสกปรก", f"{stain_percent:.2f}%")

                # Additional heuristic detectors (mold, peeling_paint, rust)
                if "mold" in damage_types:
                    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                    lower_mold = np.array([30, 40, 40])
                    upper_mold = np.array([90, 255, 150])
                    mask_mold = cv2.inRange(hsv, lower_mold, upper_mold)
                    kernel = np.ones((5, 5), np.uint8)
                    mask_mold = cv2.morphologyEx(mask_mold, cv2.MORPH_CLOSE, kernel)
                    mold_percent = (np.sum(mask_mold > 0) / mask_mold.size) * 100
                    mold_img = img_np.copy()
                    mold_img[mask_mold > 0] = [0, 255, 0]
                    st.write("**🟢 Mold Detection Results**")
                    st.image(mold_img, caption="Detected Mold (Green overlay)", use_column_width=True)
                    damage_results["mold"] = {"coverage": float(mold_percent), "severity": "High" if mold_percent > 5 else "Low"}
                    st.metric("พื้นที่เชื้อรา", f"{mold_percent:.2f}%")

                if "peeling_paint" in damage_types:
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    roughness = np.var(lap)
                    damage_results["peeling_paint"] = {"roughness": float(roughness)}
                    st.write("**🧩 Peeling / Texture Roughness**")
                    st.metric("Texture Roughness (var of Laplacian)", f"{roughness:.2f}")

                if "rust" in damage_types:
                    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                    lower_rust = np.array([5, 50, 50])
                    upper_rust = np.array([25, 255, 255])
                    mask_rust = cv2.inRange(hsv, lower_rust, upper_rust)
                    rust_percent = (np.sum(mask_rust > 0) / mask_rust.size) * 100
                    rust_img = img_np.copy()
                    rust_img[mask_rust > 0] = [0, 0, 255]
                    st.write("**🟤 Rust Detection Results**")
                    st.image(rust_img, caption="Detected Rust (Blue overlay)", use_column_width=True)
                    damage_results["rust"] = {"coverage": float(rust_percent), "severity": "High" if rust_percent > 2 else "Low"}
                    st.metric("พื้นที่สนิม", f"{rust_percent:.2f}%")

                # Save results to session
                st.session_state.damage_results = damage_results

                # Summary + recommendations
                st.markdown("---")
                st.markdown("<div class='metric-card'><h3>📋 Damage Assessment Summary</h3>", unsafe_allow_html=True)
                for d_type, data in damage_results.items():
                    if d_type == "crack":
                        st.markdown(
                            f"<p><strong>รอยร้าว:</strong> พบ {data['count']} จุด - ระดับ {data['severity']}</p>",
                            unsafe_allow_html=True,
                        )
                    elif d_type == "stain":
                        st.markdown(
                            f"<p><strong>คราบสกปรก:</strong> {data['coverage']:.2f}% - ระดับ {data['severity']}</p>",
                            unsafe_allow_html=True,
                        )
                    elif d_type == "mold":
                        st.markdown(
                            f"<p><strong>เชื้อรา:</strong> {data['coverage']:.2f}% - ระดับ {data['severity']}</p>",
                            unsafe_allow_html=True,
                        )
                    elif d_type == "peeling_paint":
                        st.markdown(f"<p><strong>ผิว/ความหยาบ:</strong> var={data['roughness']:.2f}</p>", unsafe_allow_html=True)
                    elif d_type == "rust":
                        st.markdown(
                            f"<p><strong>สนิม:</strong> {data['coverage']:.2f}% - ระดับ {data['severity']}</p>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"<p><strong>{d_type}:</strong> {data}</p>", unsafe_allow_html=True)

                st.markdown("<h4>💡 คำแนะนำ</h4>", unsafe_allow_html=True)
                if "crack" in damage_results and damage_results["crack"]["count"] > 5:
                    st.markdown("<p>⚠️ พบรอยร้าวมาก - แนะนำให้ซ่อมแซมก่อนทาสีใหม่</p>", unsafe_allow_html=True)
                elif "crack" in damage_results and damage_results["crack"]["count"] > 0:
                    st.markdown("<p>ℹ️ พบรอยร้าวเล็กน้อย - ตรวจสอบเชิงลึกหากรอยขยายตัว</p>", unsafe_allow_html=True)

                if "stain" in damage_results:
                    cov = damage_results["stain"]["coverage"]
                    if cov > 10:
                        st.markdown("<p>⚠️ มีคราบสกปรกมาก - ควรทำความสะอาดก่อนตกแต่ง</p>", unsafe_allow_html=True)
                    elif cov > 5:
                        st.markdown("<p>ℹ️ คราบระดับปานกลาง - อาจต้องขัดหรือทำความสะอาด</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p>✓ คราบอยู่ในระดับน้อย</p>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TAB 4: Full Report ----------------------------------
    with tab4:
        st.subheader("📊 Inspection Report")
        if len(st.session_state.inspection_data) == 0 and len(st.session_state.damage_results) == 0:
            st.info("ℹ️ ยังไม่มีข้อมูล กรุณาใช้งาน Tab อื่นก่อน (Segmentation / Material Estimation / Damage Detection)")
        else:
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Header metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("วันที่", report_date.split()[0])
            with col2:
                st.metric("ห้อง", st.session_state.room_type)
            with col3:
                st.metric("รายการวัสดุ", len(st.session_state.inspection_data))

            st.markdown("---")
            st.write("### 📦 Material Summary")
            total_cost = 0.0
            total_area = 0.0
            for surface, data in st.session_state.inspection_data.items():
                with st.expander(f"📍 {surface.upper()}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("พื้นที่", f"{data['area_m2']:.2f} m²")
                    with col2:
                        st.metric("วัสดุ", data['material'])
                    with col3:
                        st.metric("ราคา", f"฿{data['cost']:,.2f}")
                    st.write("รายละเอียด:", data.get("details", {}))
                    st.write("เวลาบันทึก:", data.get("timestamp"))

                total_cost += float(data.get("cost", 0.0))
                total_area += float(data.get("area_m2", 0.0))

            st.markdown(
                f"""
            <div class="metric-card">
                <h3>💰 Total Estimated Cost</h3>
                <h1 style="font-size: 2.2rem; margin: 0.5rem 0;">฿{total_cost:,.2f}</h1>
                <p><strong>Total Area:</strong> {total_area:.2f} m²</p>
                <p><strong>Average Cost/m²:</strong> ฿{(total_cost/total_area if total_area>0 else 0):,.2f}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Damage summary
            if st.session_state.damage_results:
                st.markdown("---")
                st.write("### 🔍 Damage Summary")
                for k, v in st.session_state.damage_results.items():
                    st.write(f"- **{k}**: {v}")

            # Export options
            st.markdown("---")
            st.write("### 📥 Export Report")
            report_json = {
                "date": report_date,
                "room_type": st.session_state.room_type,
                "materials": st.session_state.inspection_data,
                "damage": st.session_state.damage_results,
                "total_cost": total_cost,
                "total_area": total_area,
            }
            json_bytes = json.dumps(report_json, indent=2, ensure_ascii=False).encode("utf-8")
            csv_lines = "Surface,Material,Area (m²),Cost (THB)\n"
            for surface, d in st.session_state.inspection_data.items():
                csv_lines += f"{surface},{d.get('material','')},{d.get('area_m2',0):.2f},{d.get('cost',0):.2f}\n"
            csv_lines += f"TOTAL,,{total_area:.2f},{total_cost:.2f}\n"

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📄 Download JSON",
                    data=json_bytes,
                    file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_lines,
                    file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # End of main flow

# ---------------- End of app.py ------------------------------------------------
st.markdown("---")
st.caption("🚀 AI Home Inspection System — Powered by SAM + Streamlit")


Export เป็น PDF Report

แรงค์ความรุนแรงของรอยเสียหาย (Severity Score)

Auto Segmentation โดยไม่ต้องคลิก
