import os
import time
import cv2
import torch
import sqlite3
import subprocess
import easyocr
import sys
import tty
import termios
import json
import pyaudio
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from vosk import Model, KaldiRecognizer
from datetime import datetime
from torch.nn.functional import cosine_similarity
import re
import select
import face_recognition

try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError:
    InputDevice = None
    ecodes = None
    list_devices = None

# =====================================================
# Text-to-Speech
# =====================================================
def speak_arabic(text):
    subprocess.run([
        "espeak-ng",
        "-v", "mb-ar1",
        "-s", "140",
        text
    ])

def speak_english(text):
    subprocess.run([
        "espeak-ng",
        "-v", "en-us",
        "-s", "140",
        text
    ])

# =====================================================
# صوت البداية
# =====================================================
speak_arabic("يرجى الانتظار، جاري تحميل النظام")

# =====================================================
# تحميل MobileNet
# =====================================================
print("🔄 تحميل MobileNet...")
embedding_model = models.mobilenet_v2(pretrained=True)
embedding_model.classifier = torch.nn.Identity()
embedding_model.eval()
print("✅ تم تحميل MobileNet\n")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================================================
# تحميل YOLO
# =====================================================
print("🔄 جاري تحميل YOLO...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
print("✅ تم تحميل YOLO\n")

# =====================================================
# EasyOCR
# =====================================================
reader = easyocr.Reader(['ar', 'en'])

# =====================================================
# تحميل Vosk
# =====================================================
vosk_model = Model("/home/basira/yolov5/vosk-model-ar-0.22-linto-1.1.0")

# =====================================================
# قاعدة البيانات
# =====================================================
conn = sqlite3.connect("basira.db", check_same_thread=False)
cursor = conn.cursor()

# =====================================================
# Face Recognition Database
# =====================================================
cursor.execute("""
CREATE TABLE IF NOT EXISTS face_embeddings (
    name TEXT PRIMARY KEY,
    encoding BLOB
)
""")
conn.commit()

# =====================================================
# صوت بعد التحميل
# =====================================================
speak_arabic("مرحبًا بك في مشروع بصيرة")
speak_arabic("الزر الأول اكتشاف العناصر، الزر الثاني قراءة النصوص، الزر الثالث تسجيل عنصر شخصي، الزر الرابع معرفة الوقت، الزر الخامس التعرف على الأشخاص، الزر السادس وصف ملامح الوجه")

# =====================================================
# Detect Arabic
# =====================================================
def is_arabic(text):
    return re.search(r'[\u0600-\u06FF]', text)

# =====================================================
# Smart Mixed Speech
# =====================================================
def speak_mixed_text(text):
    words = text.split()

    arabic_part = []
    english_part = []

    for word in words:
        if is_arabic(word):
            arabic_part.append(word)
        else:
            english_part.append(word)

    if arabic_part:
        speak_arabic(" ".join(arabic_part))

    if english_part:
        speak_english(" ".join(english_part))

# =====================================================
# التقاط صورة
# =====================================================
def capture_image(filename="frame.jpg", w=640, h=480):
    subprocess.run([
        "rpicam-still",
        "-o", filename,
        "--width", str(w),
        "--height", str(h),
        "-n"
    ])
    return filename

# =====================================================
# Beep
# =====================================================
def beep():
    subprocess.run(["play", "-nq", "-t", "alsa", "synth", "0.1", "sine", "1000"])

# =====================================================
# Embedding
# =====================================================
def get_embedding(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        vec = embedding_model(img)
    return vec.squeeze().numpy()

# =====================================================
# Personal Matching
# =====================================================
def match_personal(frame):
    query_vec = get_embedding(frame)

    cursor.execute("SELECT name, vector FROM personal_embeddings")
    items = cursor.fetchall()

    best_name = None
    best_score = 0

    for name, blob in items:
        stored_vec = np.frombuffer(blob, dtype=np.float32)
        score = cosine_similarity(
            torch.tensor(query_vec),
            torch.tensor(stored_vec),
            dim=0
        ).item()

        if score > best_score:
            best_score = score
            best_name = name

    if best_score > 0.80:
        return best_name

    return None

# =====================================================
# Translation
# =====================================================
def get_arabic_translation(label):
    cursor.execute("SELECT arabic_label FROM yolo_translations WHERE english_label=?", (label,))
    result = cursor.fetchone()
    return result[0] if result else label

# =====================================================
# Time
# =====================================================
def speak_time():
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    period = "صباحًا" if hour < 12 else "مساءً"

    hour12 = hour if 1 <= hour <= 12 else abs(hour - 12)
    if hour12 == 0:
        hour12 = 12

    if minute == 0:
        sentence = f"الساعة {hour12} تمامًا {period}"
    else:
        sentence = f"الساعة {hour12} و {minute} دقيقة {period}"

    print(sentence)
    speak_arabic(sentence)

# =====================================================
# Object detection using YOLO
# =====================================================
def run_detection():
    img_path = capture_image()
    frame = cv2.imread(img_path)

    if frame is None:
        print("\n Error: Failed to capture image\n")  # ← إضافة
        speak_arabic("حدث خطأ في التصوير")
        return

    item = match_personal(frame)
    if item:
        print(f"\n Detected (Personal): {item}\n")  # ← إضافة
        speak_arabic(f"أمامك {item}")
        return

    results = model(frame)
    detections = results.pred[0]
    labels = results.names

    if len(detections) == 0:
        print("\n No object detected\n")  # ← إضافة
        speak_arabic("لم يتم التعرف")
        return

    class_id = int(detections[0][5])
    english_label = labels[class_id]
    arabic_label = get_arabic_translation(english_label)

    print("\n YOLO detected:")  # ← إضافة
    print("---------------------")
    print(arabic_label)
    print("---------------------\n")

    speak_arabic(f"أمامك {arabic_label}")

# =====================================================
# OCR (FINAL 🔥)
# =====================================================
def run_ocr():
    img = capture_image("ocr.jpg", 1280, 720)
    results = reader.readtext(img)

    text = ""
    for (_, t, _) in results:
        text += t + " "

    text = text.strip()

    if text:
        print("\n🧾 EasyOCR قرأ النص التالي:")
        print("--------------------------------")
        print(text)
        print("--------------------------------\n")

        speak_mixed_text(text)

    else:
        print("\n❌ EasyOCR لم يتعرف على نص\n")
        speak_arabic("لم أتعرف على نص")

# =====================================================
# Register Object
# =====================================================
def register_object():
    print("🎙️ قولي اسم العنصر الآن...")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8192)
    stream.start_stream()

    recognizer = KaldiRecognizer(vosk_model, 16000)

    object_name = None

    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            object_name = result.get("text", "").strip()
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    if not object_name:
        speak_arabic("لم أسمع الاسم")
        return

    speak_arabic("جاري تسجيل العنصر، يرجى تحريك العنصر كل مرة عند سماع الصوت")

    embeddings = []

    for i in range(30):
        img_path = f"temp_{i}.jpg"
        capture_image(img_path)

        beep()

        frame = cv2.imread(img_path)
        vec = get_embedding(frame)
        embeddings.append(vec)

        time.sleep(0.5)

    avg_vec = np.mean(embeddings, axis=0).astype(np.float32)

    cursor.execute("INSERT OR REPLACE INTO personal_embeddings VALUES (?,?)",
                   (object_name, avg_vec.tobytes()))
    conn.commit()

    speak_arabic(f"تم تسجيل العنصر بنجاح باسم {object_name}")

# =====================================================
# Face Recognition + Face Description
# =====================================================

def get_face_encoding(frame):
    """Extract one face encoding using the lightweight HOG detector."""
    if frame is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return None

    # Use the largest face if more than one face is visible.
    location = max(
        locations,
        key=lambda box: max(0, box[2] - box[0]) * max(0, box[1] - box[3])
    )

    encodings = face_recognition.face_encodings(
        rgb,
        known_face_locations=[location]
    )

    return encodings[0] if encodings else None


def listen_name():
    """Listen for a person's name using the existing offline Vosk model."""
    print("🎙️ قولي اسم الشخص الآن...")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=8192
    )
    stream.start_stream()

    recognizer = KaldiRecognizer(vosk_model, 16000)
    name = None

    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                name = result.get("text", "").strip()

                if name:
                    break
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    return name


def register_person():
    """Register a person using multiple face encodings."""
    name = listen_name()

    if not name:
        speak_arabic("لم أسمع الاسم")
        return

    speak_arabic("جاري تسجيل الوجه، انظري إلى الكاميرا وحركي وجهك قليلًا")

    encodings = []

    # Collect several valid views to make recognition more stable.
    for i in range(30):
        img_path = f"face_register_{i}.jpg"
        capture_image(img_path, 960, 720)

        frame = cv2.imread(img_path)
        encoding = get_face_encoding(frame)

        if encoding is not None:
            encodings.append(encoding)
            beep()
            print(f"✅ وجه صالح {len(encodings)}/20")
        else:
            print("⚠️ لم أجد وجهًا واضحًا في هذه الصورة")

        if len(encodings) >= 20:
            break

        time.sleep(0.25)

    if len(encodings) < 5:
        speak_arabic("لم أستطع تسجيل الوجه بشكل واضح")
        return

    # Average multiple encodings for a more stable representation.
    avg_encoding = np.mean(encodings, axis=0).astype(np.float32)

    cursor.execute(
        "INSERT OR REPLACE INTO face_embeddings VALUES (?, ?)",
        (name, avg_encoding.tobytes())
    )
    conn.commit()

    speak_arabic(f"تم تسجيل {name} بنجاح")
    print(f"👤 تم تسجيل الشخص: {name}")


def recognize_person():
    """Recognize a previously registered person."""
    speak_arabic("جاري التعرف على الشخص")

    img_path = "face_recognition.jpg"
    capture_image(img_path, 960, 720)

    frame = cv2.imread(img_path)
    encoding = get_face_encoding(frame)

    if encoding is None:
        speak_arabic("لم أجد وجهًا واضحًا")
        return

    cursor.execute("SELECT name, encoding FROM face_embeddings")
    people = cursor.fetchall()

    if not people:
        speak_arabic("لا يوجد أشخاص مسجلون")
        return

    best_name = None
    best_distance = 999.0

    for name, blob in people:
        stored = np.frombuffer(blob, dtype=np.float32)

        if stored.size != encoding.size:
            continue

        distance = float(np.linalg.norm(encoding - stored))

        if distance < best_distance:
            best_distance = distance
            best_name = name

    # 0.60 is a conservative/common face-recognition threshold.
    if best_name is not None and best_distance <= 0.60:
        print(f"👤 الشخص: {best_name} | distance={best_distance:.3f}")
        speak_arabic(f"هذه {best_name}")
    else:
        print(f"❓ شخص غير معروف | distance={best_distance:.3f}")
        speak_arabic("هذا الشخص غير مسجل")


def _describe_eye_color(image_bgr, eye_points):
    if not eye_points:
        return None

    xs = [int(p[0]) for p in eye_points]
    ys = [int(p[1]) for p in eye_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if max_x - min_x < 4 or max_y - min_y < 3:
        return None

    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    rx = max(2, int((max_x - min_x) * 0.18))
    ry = max(2, int((max_y - min_y) * 0.35))

    h, w = image_bgr.shape[:2]
    x1, x2 = max(0, cx-rx), min(w, cx+rx+1)
    y1, y2 = max(0, cy-ry), min(h, cy+ry+1)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hval = hsv[:, :, 0].reshape(-1)
    sval = hsv[:, :, 1].reshape(-1)
    vval = hsv[:, :, 2].reshape(-1)

    valid = (vval > 25) & (vval < 245) & (sval > 20)

    if np.sum(valid) < 5:
        return "داكنة"

    hue = float(np.median(hval[valid]))
    sat = float(np.median(sval[valid]))
    val = float(np.median(vval[valid]))

    if sat < 45:
        return "داكنة" if val < 85 else "فاتحة"

    if hue < 10 or hue >= 170:
        return "بنية داكنة"
    if hue < 25:
        return "بنية"
    if hue < 38:
        return "عسلية"
    if hue < 85:
        return "خضراء"
    if hue < 130:
        return "زرقاء"

    return "داكنة"


def _describe_hair_color(image_bgr, face_box):
    top, right, bottom, left = face_box
    h, w = image_bgr.shape[:2]

    fw = max(1, right-left)
    fh = max(1, bottom-top)

    x1 = max(0, left + int(fw*0.10))
    x2 = min(w, right - int(fw*0.10))
    y1 = max(0, top - int(fh*0.55))
    y2 = max(0, top + int(fh*0.10))

    if x2 <= x1 or y2 <= y1:
        return None

    region = image_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hval = hsv[:, :, 0].reshape(-1)
    sval = hsv[:, :, 1].reshape(-1)
    vval = hsv[:, :, 2].reshape(-1)

    dark_ratio = float(np.mean(vval < 85))
    medium_ratio = float(np.mean(vval < 130))

    if medium_ratio < 0.25:
        return None

    valid = (sval > 25) & (vval > 25) & (vval < 220)

    if np.sum(valid) < 20:
        return "أسود أو داكن" if dark_ratio > 0.45 else None

    hue = float(np.median(hval[valid]))
    val = float(np.median(vval[valid]))

    if dark_ratio > 0.60 and val < 90:
        return "أسود"
    if hue < 12 or hue >= 170:
        return "بني داكن"
    if hue < 28:
        return "بني"
    if hue < 40:
        return "بني فاتح"
    if hue < 85:
        return "أشقر أو بني فاتح"
    if hue < 130:
        return "داكن مائل للرمادي"

    return "داكن"


def _describe_hair_length(image_bgr, face_box):
    top, right, bottom, left = face_box
    h, w = image_bgr.shape[:2]

    fw = max(1, right-left)
    fh = max(1, bottom-top)

    x1 = max(0, left - int(fw*0.35))
    x2 = min(w, right + int(fw*0.35))
    y1 = max(0, top)
    y2 = min(h, bottom + int(fh*1.55))

    if x2 <= x1 or y2 <= y1:
        return None

    region = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    mask = (gray < 80).astype(np.uint8) * 255
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    lower = mask[int(mask.shape[0]*0.45):]
    if lower.size == 0:
        return None

    ratio = float(np.mean(lower > 0))

    if ratio > 0.38:
        return "طويل"
    if ratio > 0.18:
        return "متوسط"
    if ratio > 0.07:
        return "قصير"

    return None


def _describe_glasses(image_bgr, landmarks):
    if not landmarks:
        return None

    scores = []

    for key in ("left_eye", "right_eye"):
        points = landmarks.get(key)
        if not points:
            continue

        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]

        x1 = max(0, min(xs)-12)
        x2 = min(image_bgr.shape[1], max(xs)+12)
        y1 = max(0, min(ys)-10)
        y2 = min(image_bgr.shape[0], max(ys)+10)

        if x2 <= x1 or y2 <= y1:
            continue

        region = image_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        scores.append(float(np.mean(edges > 0)))

    if len(scores) < 2:
        return None

    score = float(np.mean(scores))

    if score > 0.115:
        return True
    if score < 0.065:
        return False

    return None


def _describe_face_direction(landmarks):
    nose = landmarks.get("nose_bridge")
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")

    if not nose or not left_eye or not right_eye:
        return None

    nose_x = float(np.mean([p[0] for p in nose]))
    left_x = float(np.mean([p[0] for p in left_eye]))
    right_x = float(np.mean([p[0] for p in right_eye]))

    center = (left_x + right_x) / 2
    distance = abs(right_x-left_x)

    if distance < 1:
        return None

    ratio = (nose_x-center)/distance

    if ratio < -0.12:
        return "إلى اليمين"
    if ratio > 0.12:
        return "إلى اليسار"

    return "للأمام"


def describe_face():
    """Describe visible facial features. It does not identify the person."""
    print("\n🙂 جاري تحليل ملامح الوجه...\n")
    speak_arabic("ثواني، جاري تحليل ملامح الوجه")

    img_path = "face_description.jpg"
    capture_image(img_path, 960, 720)

    frame = cv2.imread(img_path)

    if frame is None:
        speak_arabic("تعذر التقاط الصورة")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb, model="hog")

    if not locations:
        speak_arabic("لم أجد وجهًا واضحًا أمام الكاميرا")
        return

    face_box = max(
        locations,
        key=lambda box: max(0, box[2]-box[0]) *
                        max(0, box[1]-box[3])
    )

    landmark_list = face_recognition.face_landmarks(
        rgb,
        face_locations=[face_box]
    )

    if not landmark_list:
        speak_arabic("وجدت وجهًا لكن لم أستطع تحليل ملامحه")
        return

    landmarks = landmark_list[0]

    top, right, bottom, left = face_box
    fw = max(1, right-left)
    fh = max(1, bottom-top)

    # Approximate face shape.
    ratio = fw / fh
    chin = landmarks.get("chin", [])

    if ratio > 0.92:
        face_shape = "دائري"
    elif ratio < 0.73:
        face_shape = "طويل"
    elif len(chin) >= 10 and (max(p[0] for p in chin)-min(p[0] for p in chin)) < fw*0.68:
        face_shape = "بيضاوي"
    else:
        face_shape = "بيضاوي"

    glasses = _describe_glasses(frame, landmarks)

    eye_colors = []
    for key in ("left_eye", "right_eye"):
        color = _describe_eye_color(frame, landmarks.get(key))
        if color:
            eye_colors.append(color)

    eye_color = None
    if eye_colors:
        eye_color = max(set(eye_colors), key=eye_colors.count)

    hair_color = _describe_hair_color(frame, face_box)
    hair_length = _describe_hair_length(frame, face_box)
    direction = _describe_face_direction(landmarks)

    parts = [f"شكل الوجه {face_shape}"]

    if glasses is True:
        parts.append("وتلبس نظارة")
    elif glasses is False:
        parts.append("ولا تظهر نظارة")

    if hair_length:
        parts.append(f"والشعر {hair_length}")

    if hair_color:
        parts.append(f"ولون الشعر {hair_color}")

    if eye_color:
        parts.append(f"ولون العينين {eye_color}")

    if direction:
        parts.append(f"والوجه متجه {direction}")

    description = "، ".join(parts) + "."

    print("📝 وصف الوجه:")
    print(description)

    speak_arabic(description)


def find_keyboard_device():
    """
    Find a real keyboard/input device that exposes the E key.
    This works without a desktop screen and does not require the
    'keyboard' Python package or sudo, provided the basira user can
    read /dev/input/event*.
    """
    if InputDevice is None or ecodes is None:
        return None

    candidates = []

    for path in list_devices():
        try:
            dev = InputDevice(path)
            caps = dev.capabilities()
            key_caps = caps.get(ecodes.EV_KEY, [])

            # KEY_E is 18 in Linux input-event codes, but use ecodes.
            if ecodes.KEY_E in key_caps:
                name = (dev.name or "").lower()

                # Prefer devices whose name looks like a keyboard.
                priority = 0
                if "keyboard" in name:
                    priority += 10
                if "kbd" in name:
                    priority += 5

                candidates.append((priority, dev))
            else:
                dev.close()
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def handle_face_button():
    """
    E behavior:
      - Quick E press/release -> recognize person.
      - Hold E continuously for 3 seconds -> register new person.

    Uses Linux evdev press/release events, so the hold is based on the
    actual key being held. It does not depend on keyboard autorepeat,
    and it works without a graphical desktop/screen.
    """
    print("\n[E] ضغطة سريعة = التعرف، هولد E لمدة 3 ثواني = تسجيل شخص")

    if InputDevice is None:
        speak_arabic("مكتبة التحكم بالكيبورد غير مثبتة")
        print("❌ evdev غير مثبت. ثبتيه بالأمر: sudo apt install python3-evdev")
        return

    dev = find_keyboard_device()

    if dev is None:
        speak_arabic("لم أجد لوحة المفاتيح")
        print("❌ لم يتم العثور على جهاز كيبورد يحتوي على زر E.")
        return

    try:
        start = None
        hold_triggered = False

        # We are called after read_key() has consumed the initial E press.
        # Wait for the corresponding key-release/auto-repeat events.
        for event in dev.read_loop():
            if event.type != ecodes.EV_KEY:
                continue

            key_event = event

            if key_event.code != ecodes.KEY_E:
                continue

            # value=1 -> key down, value=0 -> key up, value=2 -> repeat
            if key_event.value == 1:
                if start is None:
                    start = time.monotonic()

            elif key_event.value == 2:
                # Autorepeat is ignored for timing; actual release determines
                # the duration. If repeat is available, it can still trigger
                # registration at 3 seconds while the key remains held.
                if start is not None and not hold_triggered:
                    if time.monotonic() - start >= 3.0:
                        hold_triggered = True
                        register_person()
                        return

            elif key_event.value == 0:
                if start is None:
                    recognize_person()
                else:
                    duration = time.monotonic() - start

                    if duration >= 3.0:
                        if not hold_triggered:
                            register_person()
                    else:
                        recognize_person()

                    return

    except PermissionError:
        speak_arabic("لا توجد صلاحية للوصول إلى لوحة المفاتيح")
        print("❌ لا توجد صلاحية لقراءة /dev/input/event*.")
        print("أضيفي المستخدم basira إلى مجموعة input ثم أعيدي تسجيل الدخول:")
        print("sudo usermod -aG input basira")

    except Exception as e:
        print(f"❌ خطأ في قراءة الكيبورد: {e}")
        speak_arabic("حدث خطأ في زر التعرف")

    finally:
        try:
            dev.close()
        except Exception:
            pass


# =====================================================
# Read Key
# =====================================================
def read_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch.lower()

# =====================================================
# Main Loop
# =====================================================

print("""
==============================
النظام جاهز

a = كشف عنصر
b = قراءة نص
c = تسجيل عنصر
d = الساعة
e = التعرف على شخص
   ضغط E لمدة 3 ثوانٍ = تسجيل شخص
f = وصف ملامح الوجه
q = خروج
==============================
""")

while True:
    key = read_key()

    if key == 'a':
        run_detection()

    elif key == 'b':
        run_ocr()

    elif key == 'c':
        register_object()

    elif key == 'd':
        speak_time()

    elif key == 'e':
        handle_face_button()

    elif key == 'f':
        describe_face()

    elif key == 'q':
        speak_arabic("إلى اللقاء")
        break

    print("\n✅ جاهز للزر التالي\n")
