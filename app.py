import cv2
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image
import io

st.set_page_config(page_title="Capital Letter Detector", layout="centered")
st.title("🧠 Capital Letter Detector from Handwritten Notes")

st.write("Upload a handwritten image. The app will highlight detected **capital letters** using OCR and OpenCV.")

uploaded_file = st.file_uploader("Choose a handwritten image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # OCR for capital letters only
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if data['text'][i].isupper():
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(img_np, caption="Capital Letters Highlighted", use_column_width=True)

    # Show extracted capital letters
    extracted_text = pytesseract.image_to_string(thresh, config=config)
    st.subheader("📋 Extracted Capital Letters")
    st.code(extracted_text)
