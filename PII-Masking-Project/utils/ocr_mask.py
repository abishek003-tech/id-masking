import os
from typing import Dict, Tuple, List
import re

import cv2
import pytesseract
from PIL import Image

# ---- Windows: set tesseract path automatically if installed in default location
_DEFAULT_WIN_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
for _p in _DEFAULT_WIN_PATHS:
    if os.path.exists(_p):
        pytesseract.pytesseract.tesseract_cmd = _p
        break

# ---- Regexes for detection
SSN_REGEX = re.compile(r"\b(\d{3}[- ]?\d{2}[- ]?\d{4})\b")

# A very simple "name-ish" detector:
# - words near labels like "Name", "Applicant", "Holder"
# - or two consecutive capitalized words (e.g., "John Doe")
NAME_LABELS = {"name", "applicant", "holder", "bearer"}

def _read_words_with_boxes(image_bgr) -> List[dict]:
    """
    Uses pytesseract to get words with bounding boxes.
    Returns list of dicts: {'text': str, 'left': int, 'top': int, 'width': int, 'height': int}
    """
    # pytesseract image_to_data needs RGB or PIL
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        if not txt:
            continue
        words.append({
            "text": txt,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "conf": float(data["conf"][i]) if data["conf"][i].replace('.', '', 1).isdigit() else -1.0
        })
    return words

def _expand_bbox(left, top, width, height, pad=4) -> Tuple[int, int, int, int]:
    return max(0, left - pad), max(0, top - pad), width + 2*pad, height + 2*pad

def _merge_overlaps(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Merge overlapping rectangles (simple union via iterative pass).
    Boxes are (x, y, w, h)
    """
    if not boxes:
        return []

    # Convert to x1,y1,x2,y2
    rects = []
    for (x, y, w, h) in boxes:
        rects.append([x, y, x+w, y+h])

    merged = True
    while merged:
        merged = False
        new_rects = []
        used = [False]*len(rects)

        for i in range(len(rects)):
            if used[i]: 
                continue
            x1, y1, x2, y2 = rects[i]
            for j in range(i+1, len(rects)):
                if used[j]:
                    continue
                a1, b1, a2, b2 = rects[j]
                # Check overlap
                if not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1):
                    # Merge
                    x1 = min(x1, a1); y1 = min(y1, b1)
                    x2 = max(x2, a2); y2 = max(y2, b2)
                    used[j] = True
                    merged = True
            used[i] = True
            new_rects.append([x1, y1, x2, y2])

        rects = new_rects

    # back to x,y,w,h
    out = []
    for (x1, y1, x2, y2) in rects:
        out.append((x1, y1, x2 - x1, y2 - y1))
    return out

def _find_ssn_boxes(words: List[dict]) -> List[Tuple[int, int, int, int]]:
    hits = []
    for w in words:
        if SSN_REGEX.search(w["text"]):
            x, y, wdt, hgt = _expand_bbox(w["left"], w["top"], w["width"], w["height"], pad=2)
            hits.append((x, y, wdt, hgt))
    return _merge_overlaps(hits)

def _find_name_boxes(words: List[dict]) -> List[Tuple[int, int, int, int]]:
    hits = []

    # 1) After a label like "Name:"
    for i, w in enumerate(words):
        t = w["text"]
        if t.strip(":").lower() in NAME_LABELS:
            # take the next 1-3 words as "name"
            for k in range(1, 4):
                if i + k < len(words):
                    nxt = words[i+k]
                    x = min(w["left"], nxt["left"])
                    y = min(w["top"], nxt["top"])
                    x2 = max(w["left"] + w["width"], nxt["left"] + nxt["width"])
                    y2 = max(w["top"] + w["height"], nxt["top"] + nxt["height"])
                    xx, yy, ww, hh = _expand_bbox(x, y, x2 - x, y2 - y, pad=2)
                    hits.append((xx, yy, ww, hh))

    # 2) Two consecutive Capitalized words (e.g., John Doe)
    def is_cap_word(s: str) -> bool:
        return len(s) > 1 and s[0].isupper() and s[1:].islower()

    i = 0
    while i < len(words) - 1:
        w1, w2 = words[i], words[i+1]
        if is_cap_word(w1["text"]) and is_cap_word(w2["text"]):
            x = min(w1["left"], w2["left"])
            y = min(w1["top"], w2["top"])
            x2 = max(w1["left"] + w1["width"], w2["left"] + w2["width"])
            y2 = max(w1["top"] + w1["height"], w2["top"] + w2["height"])
            xx, yy, ww, hh = _expand_bbox(x, y, x2 - x, y2 - y, pad=2)
            hits.append((xx, yy, ww, hh))
            i += 2
        else:
            i += 1

    return _merge_overlaps(hits)

def mask_sensitive_info(input_path: str, output_path: str) -> Tuple[str, Dict[str, int]]:
    """
    Reads the image, finds SSNs and probable names, draws black rectangles, saves output.
    Returns: (output_path, {'ssn': n1, 'names': n2, 'total': n1+n2})
    """
    # Read with OpenCV
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    # OCR words
    words = _read_words_with_boxes(image)

    # Find sensitive boxes
    ssn_boxes = _find_ssn_boxes(words)
    name_boxes = _find_name_boxes(words)

    # Draw masks
    for (x, y, w, h) in ssn_boxes + name_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)

    # Save output
    cv2.imwrite(output_path, image)

    counts = {
        "ssn": len(ssn_boxes),
        "names": len(name_boxes),
        "total": len(ssn_boxes) + len(name_boxes),
    }
    return output_path, counts