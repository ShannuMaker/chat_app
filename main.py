import os
import uuid
import base64
import urllib.request
import numpy as np
import cv2
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

# Model file paths & working mirror URLs
GENDER_PROTO_FILE = "gender_deploy.prototxt"
GENDER_MODEL_FILE = "gender_net.caffemodel"

PROTO_URL = "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/gender_deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/gender_net.caffemodel"

def download_file(url: str, filename: str):
    """Downloads files with custom headers to prevent HTTP 403/404 errors."""
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
        out_file.write(response.read())

# Auto-download pre-trained Caffe Gender Detection Model if missing
if not os.path.exists(GENDER_PROTO_FILE):
    print("Downloading gender_deploy.prototxt...")
    download_file(PROTO_URL, GENDER_PROTO_FILE)

if not os.path.exists(GENDER_MODEL_FILE):
    print("Downloading gender_net.caffemodel (approx 43MB)...")
    download_file(MODEL_URL, GENDER_MODEL_FILE)

# Load OpenCV Cascades & Neural Network
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
gender_net = cv2.dnn.readNet(GENDER_MODEL_FILE, GENDER_PROTO_FILE)

# Queues and Room Tracking
waiting_males: List[WebSocket] = []
waiting_females: List[WebSocket] = []
active_rooms: Dict[str, List[WebSocket]] = {}
user_rooms: Dict[WebSocket, str] = {}
banned_users: set = set()

# Moderation Rules
PROHIBITED_WORDS = ["abuse", "spam", "hate", "scam", "badword"]

def decode_base64_image(base64_str: str):
    """Converts a base64 frame into an OpenCV BGR image matrix."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def detect_face_and_gender(img):
    """
    Detects face and classifies gender ('male' or 'female').
    Returns (True, "male"/"female") if detected, or (False, None) if no face found.
    """
    if img is None:
        return False, None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        
        if len(faces) == 0:
            return False, None

        # Select largest detected face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Crop face with padding
        padding = 20
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding)
        
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return False, None

        # Preprocess face blob for Caffe Model
        blob = cv2.dnn.blobFromImage(
            face_crop, 
            scalefactor=1.0, 
            size=(227, 227), 
            mean=(78.4263377603, 87.7689143744, 114.895847746), 
            swapRB=False
        )
        gender_net.setInput(blob)
        preds = gender_net.forward()
        
        # Classify Output: Index 0 = Male, Index 1 = Female
        gender_list = ["male", "female"]
        predicted_gender = gender_list[preds[0].argmax()]
        
        return True, predicted_gender
    except Exception as e:
        print(f"Gender classification error: {e}")
        return False, None

def detect_image_violation(img) -> bool:
    """Analyzes HSV skin ratios to filter out inappropriate content."""
    if img is None:
        return True
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])
        return skin_ratio > 0.68
    except Exception:
        return False

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Error: index.html not found!</h1>", status_code=404)

@app.websocket("/ws/chat/{gender}")
async def websocket_chat(websocket: WebSocket, gender: str):
    client_ip = websocket.client.host
    if client_ip in banned_users:
        await websocket.accept()
        await websocket.send_json({"type": "error", "payload": "⛔ Access Denied: Device/IP banned due to safety violations."})
        await websocket.close()
        return

    await websocket.accept()
    user_gender = gender.lower()
    room_id = None
    is_verified = False

    try:
        # Step 1: AI Camera Face & Gender Verification
        init_data = await websocket.receive_json()
        if init_data.get("type") == "verify":
            img = decode_base64_image(init_data.get("image", ""))
            
            face_found, detected_gender = detect_face_and_gender(img)

            if not face_found:
                await websocket.send_json({"type": "error", "payload": "❌ Verification Failed: No human face detected in camera snapshot."})
                await websocket.close()
                return

            if detected_gender != user_gender:
                await websocket.send_json({
                    "type": "error", 
                    "payload": f"❌ Gender Verification Failed: You selected '{user_gender.capitalize()}', but AI camera detected '{detected_gender.capitalize()}'."
                })
                await websocket.close()
                return

            if detect_image_violation(img):
                banned_users.add(client_ip)
                await websocket.send_json({"type": "error", "payload": "⛔ Banned: Explicit content detected during verification."})
                await websocket.close()
                return

            is_verified = True
            await websocket.send_json({"type": "status", "payload": f"✅ Verified as {user_gender.capitalize()}. Joining matchmaking queue..."})

        if not is_verified:
            await websocket.close()
            return

        # Step 2: Matchmaking
        if user_gender == "male":
            if waiting_females:
                partner_ws = waiting_females.pop(0)
                room_id = str(uuid.uuid4())
                active_rooms[room_id] = [websocket, partner_ws]
                user_rooms[websocket] = room_id
                user_rooms[partner_ws] = room_id

                await websocket.send_json({"type": "match_start", "role": "initiator", "partner_gender": "Female"})
                await partner_ws.send_json({"type": "match_start", "role": "receiver", "partner_gender": "Male"})
            else:
                waiting_males.append(websocket)
                await websocket.send_json({"type": "status", "payload": "Searching for a user..."})

        elif user_gender == "female":
            if waiting_males:
                partner_ws = waiting_males.pop(0)
                room_id = str(uuid.uuid4())
                active_rooms[room_id] = [partner_ws, websocket]
                user_rooms[websocket] = room_id
                user_rooms[partner_ws] = room_id

                await partner_ws.send_json({"type": "match_start", "role": "initiator", "partner_gender": "Female"})
                await websocket.send_json({"type": "match_start", "role": "receiver", "partner_gender": "Male"})
            else:
                waiting_females.append(websocket)
                await websocket.send_json({"type": "status", "payload": "Searching for a male user..."})

        # Step 3: WebRTC Signaling & Chat Loop
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            current_room = user_rooms.get(websocket)

            if not current_room or current_room not in active_rooms:
                continue

            if msg_type in ["offer", "answer", "candidate"]:
                for client in active_rooms[current_room]:
                    if client != websocket:
                        await client.send_json(data)

            elif msg_type == "text":
                text_payload = data.get("payload", "")
                if any(word in text_payload.lower() for word in PROHIBITED_WORDS):
                    await websocket.send_json({"type": "system", "payload": "⚠️ Message blocked: Content policy violation."})
                    continue

                for client in active_rooms[current_room]:
                    if client != websocket:
                        await client.send_json({"type": "message", "payload": text_payload})

            elif msg_type == "camera_frame":
                img = decode_base64_image(data.get("image", ""))
                if detect_image_violation(img):
                    banned_users.add(client_ip)
                    await websocket.send_json({"type": "error", "payload": "⛔ Banned for video safety violations."})
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        if websocket in waiting_males:
            waiting_males.remove(websocket)
        if websocket in waiting_females:
            waiting_females.remove(websocket)

        current_room = user_rooms.pop(websocket, None)
        if current_room and current_room in active_rooms:
            partners = active_rooms.pop(current_room)
            for client in partners:
                if client != websocket:
                    user_rooms.pop(client, None)
                    try:
                        await client.send_json({"type": "peer_disconnected"})
                        await client.close()
                    except Exception:
                        pass