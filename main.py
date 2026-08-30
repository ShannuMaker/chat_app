import os
import uuid
import base64
import urllib.request
import numpy as np
import cv2
import datetime
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse

app = FastAPI()

# Model file paths & working mirror URLs
GENDER_PROTO_FILE = "gender_deploy.prototxt"
GENDER_MODEL_FILE = "gender_net.caffemodel"
AGE_PROTO_FILE = "age_deploy.prototxt"
AGE_MODEL_FILE = "age_net.caffemodel"

BASE_MODEL_URL = "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/"

def download_file(url: str, filename: str):
    req = urllib.request.Request(
        url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
        out_file.write(response.read())

# Auto-download pre-trained Caffe Models
for file_name in [GENDER_PROTO_FILE, GENDER_MODEL_FILE, AGE_PROTO_FILE, AGE_MODEL_FILE]:
    if not os.path.exists(file_name):
        print(f"Downloading {file_name}...")
        download_file(BASE_MODEL_URL + file_name, file_name)

# Load OpenCV Cascades & Neural Networks
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
gender_net = cv2.dnn.readNet(GENDER_MODEL_FILE, GENDER_PROTO_FILE)
age_net = cv2.dnn.readNet(AGE_MODEL_FILE, AGE_PROTO_FILE)

# Queues and Room Tracking
waiting_males: List[WebSocket] = []
waiting_females: List[WebSocket] = []
active_rooms: Dict[str, List[WebSocket]] = {}
user_rooms: Dict[WebSocket, str] = {}
client_ips: Dict[WebSocket, str] = {}
banned_users: set = set()

# Moderation Rules (Scam & Abuse)
SCAM_WORDS = ["crypto", "invest", "cashapp", "venmo", "telegram", "whatsapp", "paypal", "bitcoin", "scam", "hack"]

def decode_base64_image(base64_str: str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def detect_attributes(img):
    if img is None:
        return False, None, False
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        
        if len(faces) == 0:
            return False, None, False

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        padding = 20
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding)
        
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return False, None, False

        blob = cv2.dnn.blobFromImage(
            face_crop, scalefactor=1.0, size=(227, 227), 
            mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False
        )
        
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        predicted_gender = ["male", "female"][gender_preds[0].argmax()]
        
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        predicted_age = age_list[age_preds[0].argmax()]
        
        is_kid = predicted_age in ['(0-2)', '(4-6)', '(8-12)']
        return True, predicted_gender, is_kid
    except Exception:
        return False, None, False

def detect_image_violation(img) -> bool:
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

def log_policy_acceptance(ip_address: str):
    """Logs the user IP and timestamp when they accept the legal policies."""
    with open("policy_agreements.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] IP: {ip_address} agreed to TOS, Privacy Policy, and Community Guidelines.\n")

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Error: index.html not found!</h1>", status_code=404)

@app.head("/")
async def health_check():
    return Response(status_code=200)

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.websocket("/ws/chat/{gender}")
async def websocket_chat(websocket: WebSocket, gender: str):
    client_ip = websocket.client.host
    if client_ip in banned_users:
        await websocket.accept()
        await websocket.send_json({"type": "error", "payload": "⛔ Banned: Your IP address is permanently banned from this platform."})
        await websocket.close()
        return

    await websocket.accept()
    client_ips[websocket] = client_ip
    user_gender = gender.lower()
    room_id = None
    is_verified = False

    try:
        init_data = await websocket.receive_json()
        if init_data.get("type") == "verify":
            
            # Policy Acceptance Enforcement
            if not init_data.get("policy_accepted"):
                await websocket.send_json({"type": "error", "payload": "❌ Verification Failed: You must accept the policies to use this service."})
                await websocket.close()
                return
            
            log_policy_acceptance(client_ip)
            
            img = decode_base64_image(init_data.get("image", ""))
            face_found, detected_gender, is_kid = detect_attributes(img)

            if not face_found:
                await websocket.send_json({"type": "error", "payload": "❌ Verification Failed: No human face detected."})
                await websocket.close()
                return

            if is_kid:
                banned_users.add(client_ip)
                await websocket.send_json({"type": "error", "payload": "⛔ Banned: Minors are strictly prohibited."})
                await websocket.close()
                return

            if detected_gender != user_gender:
                await websocket.send_json({"type": "error", "payload": f"❌ Verification Failed: Detected '{detected_gender.capitalize()}'."})
                await websocket.close()
                return

            if detect_image_violation(img):
                banned_users.add(client_ip)
                await websocket.send_json({"type": "error", "payload": "⛔ Banned: Explicit/Nudity content detected."})
                await websocket.close()
                return

            is_verified = True
            await websocket.send_json({"type": "status", "payload": f"✅ Verified. Joining matchmaking queue..."})

        if not is_verified:
            await websocket.close()
            return

        # Matchmaking
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

        # Chat Loop
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
                text_payload = data.get("payload", "").lower()
                if any(word in text_payload for word in SCAM_WORDS):
                    banned_users.add(client_ip)
                    await websocket.send_json({"type": "error", "payload": "⛔ Banned for Scam/Spam violations."})
                    for client in active_rooms[current_room]:
                        if client != websocket:
                            await client.send_json({"type": "system", "payload": "Stranger was banned for scamming."})
                            await client.send_json({"type": "peer_disconnected"})
                            await client.close()
                    await websocket.close()
                    break

                for client in active_rooms[current_room]:
                    if client != websocket:
                        await client.send_json({"type": "message", "payload": data.get("payload")})
            
            elif msg_type == "report":
                for client in active_rooms[current_room]:
                    if client != websocket:
                        banned_users.add(client_ips.get(client))
                        await client.send_json({"type": "error", "payload": "⛔ You have been reported and banned."})
                        await websocket.send_json({"type": "system", "payload": "User reported and IP banned. Disconnected."})
                        await client.close()
                await websocket.send_json({"type": "peer_disconnected"})
                break

            elif msg_type == "camera_frame":
                img = decode_base64_image(data.get("image", ""))
                if detect_image_violation(img):
                    banned_users.add(client_ip)
                    await websocket.send_json({"type": "error", "payload": "⛔ Banned for video safety violations."})
                    for client in active_rooms[current_room]:
                        if client != websocket:
                            await client.send_json({"type": "system", "payload": "Stranger was banned for safety violations."})
                            await client.send_json({"type": "peer_disconnected"})
                            await client.close()
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        if websocket in waiting_males:
            waiting_males.remove(websocket)
        if websocket in waiting_females:
            waiting_females.remove(websocket)
        client_ips.pop(websocket, None)

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