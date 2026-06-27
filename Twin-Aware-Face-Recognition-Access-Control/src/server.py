
import face_recognition
import numpy as np
import cv2
import pickle
import requests
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH         = "face_model.pkl"
CAMERA_INDEX       = 0               # 0 = default PC webcam

# Heltec LoRa32 IP — check its Serial Monitor after it connects to WiFi
LORA32_IP          = "192.168.1.11"   # <-- change this to your LoRa32's IP
LORA32_PORT        = 80
LORA32_URL         = f"http://{LORA32_IP}:{LORA32_PORT}/result"

# How many seconds between each photo capture
CAPTURE_INTERVAL_SEC = 5             # takes a photo every 5 seconds

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 55)
print("  Loading face_model.pkl ...")
print("=" * 55)

try:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
except FileNotFoundError:
    print(f"\n❌ '{MODEL_PATH}' not found!")
    print("   Run  train_model.py  first.\n")
    exit(1)

encodeListKnown   = bundle["encodeListKnown"]
nameListKnown     = bundle["nameListKnown"]
all_people        = bundle["all_people"]
twin_pairs        = bundle["twin_pairs"]
twin_lda_models   = bundle["twin_lda_models"]
cfg               = bundle["config"]

TOLERANCE          = cfg["TOLERANCE"]
TWIN_GAP_THRESHOLD = cfg["TWIN_GAP_THRESHOLD"]
LDA_ENABLED        = cfg["LDA_ENABLED"]

print(f"  ✅ Model loaded")
print(f"     People    : {all_people}")
print(f"     Encodings : {len(encodeListKnown)}")
print(f"     LDA models: {len(twin_lda_models)}")


# ============================================================
# RECOGNITION FUNCTION
# ============================================================

def recognize_face(face_encoding):
    matches        = face_recognition.compare_faces(encodeListKnown, face_encoding, tolerance=TOLERANCE)
    face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)

    vote_counts   = defaultdict(int)
    best_distance = defaultdict(lambda: 1.0)

    for i, (match, dist) in enumerate(zip(matches, face_distances)):
        name = nameListKnown[i]
        if match:
            vote_counts[name] += 1
            best_distance[name] = min(best_distance[name], dist)

    if not vote_counts:
        return "UNKNOWN", 1.0, "no_match"

    winner      = max(vote_counts, key=lambda n: (vote_counts[n], -best_distance[n]))
    winner_dist = best_distance[winner]

    # Check twin ambiguity
    sorted_people = sorted(best_distance.items(), key=lambda x: x[1])
    is_ambiguous  = False
    rival         = None

    if len(sorted_people) >= 2:
        top1_name, top1_dist = sorted_people[0]
        top2_name, top2_dist = sorted_people[1]
        if abs(top1_dist - top2_dist) < TWIN_GAP_THRESHOLD and top1_dist < TOLERANCE:
            is_ambiguous = True
            winner       = top1_name
            rival        = top2_name

    method = "voting"

    if is_ambiguous and LDA_ENABLED:
        key = tuple(sorted([winner, rival]))
        if key in twin_lda_models:
            model    = twin_lda_models[key]
            proba    = model['lda'].predict_proba([face_encoding])[0]
            pred_idx = np.argmax(proba)
            winner   = model['le'].inverse_transform([pred_idx])[0]
            lda_conf = proba[pred_idx]
            method   = f"LDA {lda_conf*100:.0f}%"

    return winner, winner_dist, method


# ============================================================
# SEND RESULT TO LORA32
# ============================================================

def send_to_lora32(name, allowed):
    """
    Sends a POST request to the LoRa32's web server.
    Payload: { "name": "harini", "allowed": true }
    The LoRa32 displays this on its OLED.
    """
    payload = {
        "name"   : name,
        "allowed": allowed
    }
    try:
        resp = requests.post(LORA32_URL, json=payload, timeout=3)
        print(f"  📡 Sent to LoRa32 → {payload} | HTTP {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"  ⚠  Could not reach LoRa32 at {LORA32_URL} — is it connected to WiFi?")
    except requests.exceptions.Timeout:
        print(f"  ⚠  LoRa32 did not respond in time")


# ============================================================
# DRAW FACE BOX ON FRAME
# ============================================================

def draw_face(frame, faceLoc, name, confidence, method):
    y1, x2, y2, x1 = faceLoc

    color = (0, 0, 255) if name == "UNKNOWN" else (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{name}  [{method}]  d={confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 8, y2 + th + 10), color, -1)
    cv2.putText(frame, label, (x1 + 4, y2 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return frame


# ============================================================
# PICK BEST RESULT FROM MULTIPLE FACES
# Priority: known person > UNKNOWN
# If multiple known people, pick the one with lowest distance
# ============================================================

def pick_best_result(results):
    """
    results: list of (scaledLoc, name, conf, method)
    Rule: if ANY face is a known person, return that.
          If multiple known, return the one with best (lowest) distance.
          Only return UNKNOWN if ALL faces are unknown.
    """
    known = [(loc, name, conf, method)
             for (loc, name, conf, method) in results
             if name != "UNKNOWN"]

    if known:
        # Pick known face with lowest confidence distance
        best = min(known, key=lambda x: x[2])
        return best
    elif results:
        # All unknown — return first
        return results[0]
    return None


# ============================================================
# MAIN CAMERA LOOP — photo capture mode
# ============================================================

def run():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"❌ Cannot open webcam (index {CAMERA_INDEX})")
        return

    print("\n✅ Webcam started — Photo mode.")
    print(f"   Taking a photo every {CAPTURE_INTERVAL_SEC} seconds.")
    print("   Press  Q  to quit\n")

    last_capture_time = 0
    last_results      = []    # results from last photo — drawn on live feed
    countdown         = CAPTURE_INTERVAL_SEC

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        now = time.time()
        time_since_last = now - last_capture_time
        seconds_left    = max(0, int(CAPTURE_INTERVAL_SEC - time_since_last))

        # ── CAPTURE PHOTO every CAPTURE_INTERVAL_SEC seconds ─
        if time_since_last >= CAPTURE_INTERVAL_SEC:
            last_capture_time = now
            print(f"\n📸 Capturing photo...")

            # Use current frame as the photo
            photo = frame.copy()
            small     = cv2.resize(photo, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locs = face_recognition.face_locations(rgb_small, model="hog")
            face_encs = face_recognition.face_encodings(rgb_small, face_locs)

            last_results = []
            for enc, loc in zip(face_encs, face_locs):
                name, conf, method = recognize_face(enc)
                y1, x2, y2, x1    = loc
                scaled_loc         = (y1*2, x2*2, y2*2, x1*2)
                last_results.append((scaled_loc, name, conf, method))
                print(f"   Face detected → {name} | dist={conf:.3f} | {method}")

            # ── Pick best result & send to LoRa32 ────────────
            if last_results:
                best = pick_best_result(last_results)
                _, best_name, best_conf, best_method = best
                allowed = (best_name != "UNKNOWN")
                print(f"   ✅ Best result: {best_name} | allowed={allowed}")
                send_to_lora32(best_name, allowed)
            else:
                print("   👤 No face detected in photo.")
                send_to_lora32("NONE", False)

        # ── Draw last results on live frame ───────────────────
        for (faceLoc, name, conf, method) in last_results:
            frame = draw_face(frame, faceLoc, name, conf, method)

        # ── HUD: countdown timer ──────────────────────────────
        hud = f"Next capture in: {seconds_left}s | Faces: {len(last_results)} | Q=quit"
        cv2.putText(frame, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Face Recognition — Photo Mode", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
