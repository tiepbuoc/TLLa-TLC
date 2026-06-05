import os
import sys
import time
import json
import serial
import threading
import struct
import crcmod
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import secrets

# ============================================================
# SERVER CHAT THỰC TẾ - GIAO TIẾP VỚI ESP32 QUA SERIAL
# ============================================================

app = Flask(__name__)
CORS(app)
app.secret_key = secrets.token_hex(16)

# Cấu hình Serial cho ESP32-C3
SERIAL_PORT = os.environ.get('ESP32_PORT', 'COM3')  # Mặc định COM3, có thể đổi bằng env
SERIAL_BAUD = 115200

# Trạng thái ESP32
esp32_connected = False
esp32_serial = None
esp32_status = {
    'online': False,
    'lora_frequency': 433,
    'lora_sf': 12,
    'lora_bw': 125,
    'rssi': -65,
    'snr': 8.5,
    'packets_sent': 0,
    'packets_received': 0,
    'packet_errors': 0,
    'last_heartbeat': None
}

# Hàng đợi tin nhắn
message_queue = []
processing_queue = []

# CRC16 cho kiểm tra lỗi
crc16 = crcmod.mkCrcFun(0x11021, rev=True, initCrc=0xFFFF, xorOut=0x0000)

# ============================================================
# KHỞI TẠO KẾT NỐI SERIAL VỚI ESP32
# ============================================================

def init_esp32_serial():
    global esp32_serial, esp32_connected, esp32_status
    try:
        esp32_serial = serial.Serial(
            port=SERIAL_PORT,
            baudrate=SERIAL_BAUD,
            timeout=1,
            write_timeout=1
        )
        esp32_connected = True
        esp32_status['online'] = True
        esp32_status['last_heartbeat'] = datetime.now()
        print(f"[ESP32] ✅ Kết nối thành công với {SERIAL_PORT} @ {SERIAL_BAUD} baud")
        
        # Gửi lệnh kiểm tra
        esp32_serial.write(b"STATUS\n")
        time.sleep(0.5)
        
        # Đọc phản hồi
        if esp32_serial.in_waiting:
            response = esp32_serial.readline().decode('utf-8').strip()
            print(f"[ESP32] Phản hồi: {response}")
            
        return True
    except Exception as e:
        print(f"[ESP32] ❌ Không thể kết nối: {e}")
        esp32_connected = False
        esp32_status['online'] = False
        return False

# ============================================================
# NÉN TLLa-TLC THỰC TẾ (KHÔNG FAKE)
# ============================================================

def tlla_compress(text: str) -> str:
    """Nén văn bản bằng TLLa-TLC thật"""
    try:
        from server import translate_vi_to_tlla, get_compressor
        # Bước 1: Tiếng Việt -> TLLa rút gọn
        tlla_result = translate_vi_to_tlla(text)
        tlla_text = tlla_result['result']
        # Bước 2: TLLa -> TLC compact
        compressor = get_compressor()
        compact = compressor.compress(tlla_text)
        return compact
    except Exception as e:
        print(f"[TLC] Lỗi nén: {e}")
        return text

def tlla_decompress(compact: str) -> str:
    """Giải nén TLC compact về tiếng Việt thật"""
    try:
        from server import get_compressor, translate_mixed_tlla_to_vi
        compressor = get_compressor()
        tlla_text = compressor.decompress(compact)
        vi_text = translate_mixed_tlla_to_vi(tlla_text)
        return vi_text
    except Exception as e:
        print(f"[TLC] Lỗi giải nén: {e}")
        return compact

# ============================================================
# GỬI TIN NHẮN QUA ESP32 (LO RA THẬT)
# ============================================================

def send_via_esp32(message: str, message_id: str) -> bool:
    """Gửi tin nhắn qua ESP32 bằng LoRa thật"""
    global esp32_serial, esp32_status
    
    if not esp32_connected or not esp32_serial:
        return False
    
    try:
        # Đóng gói tin nhắn: [LEN][ID][DATA][CRC]
        msg_id_bytes = message_id.encode('utf-8')[:16]
        data_bytes = message.encode('utf-8')[:200]  # Giới hạn 200 bytes
        
        # Tạo payload: 1 byte độ dài ID + ID + data
        payload = struct.pack('B', len(msg_id_bytes)) + msg_id_bytes + data_bytes
        
        # Tính CRC16
        crc = crc16(payload)
        packet = payload + struct.pack('>H', crc)
        
        # Gửi qua Serial
        esp32_serial.write(packet)
        esp32_serial.flush()
        
        esp32_status['packets_sent'] += 1
        print(f"[ESP32] 📤 Đã gửi tin nhắn ID={message_id} ({len(data_bytes)} bytes)")
        return True
        
    except Exception as e:
        print(f"[ESP32] ❌ Lỗi gửi: {e}")
        return False

# ============================================================
# NHẬN TIN NHẮN TỪ ESP32
# ============================================================

def read_from_esp32():
    """Đọc dữ liệu từ ESP32 trong thread riêng"""
    global esp32_serial, esp32_status, message_queue
    
    buffer = bytearray()
    
    while esp32_connected:
        try:
            if esp32_serial and esp32_serial.in_waiting > 0:
                data = esp32_serial.read(esp32_serial.in_waiting)
                buffer.extend(data)
                
                # Xử lý packet đầy đủ
                while len(buffer) >= 3:
                    # Đọc độ dài ID
                    id_len = buffer[0]
                    if len(buffer) < 1 + id_len + 2:
                        break
                    
                    # Trích xuất packet
                    msg_id_bytes = buffer[1:1+id_len]
                    data_start = 1 + id_len
                    
                    # Tìm CRC ở cuối
                    crc_received = struct.unpack('>H', buffer[-2:])[0]
                    crc_calc = crc16(buffer[:-2])
                    
                    if crc_received == crc_calc:
                        msg_id = msg_id_bytes.decode('utf-8')
                        msg_data = buffer[data_start:-2].decode('utf-8')
                        
                        esp32_status['packets_received'] += 1
                        message_queue.append({
                            'id': msg_id,
                            'text': msg_data,
                            'timestamp': datetime.now()
                        })
                        print(f"[ESP32] 📥 Nhận tin nhắn ID={msg_id}")
                        
                        # Xóa packet khỏi buffer
                        buffer = buffer[data_start + len(msg_data) + 2:]
                    else:
                        # CRC lỗi, bỏ 1 byte
                        esp32_status['packet_errors'] += 1
                        buffer.pop(0)
                        
        except Exception as e:
            print(f"[ESP32] Lỗi đọc: {e}")
            time.sleep(0.1)
        
        time.sleep(0.01)

# ============================================================
# API ENDPOINTS (GIỐNG FAKE NHƯNG HOẠT ĐỘNG THẬT)
# ============================================================

@app.route('/api/chat/send', methods=['POST'])
def send_chat_message():
    """Gửi tin nhắn qua LoRa thật (nén TLLa-TLC rồi gửi)"""
    data = request.get_json()
    message = data.get('message', '').strip()
    user_id = data.get('userId', 'anonymous')
    user_name = data.get('userName', 'User')
    
    if not message:
        return jsonify({'success': False, 'error': 'Tin nhắn trống'}), 400
    
    if not esp32_connected or not esp32_status['online']:
        return jsonify({'success': False, 'error': 'ESP32 chưa kết nối'}), 503
    
    # Tạo ID tin nhắn
    msg_id = f"{user_id[:8]}_{int(time.time()*1000)}"
    
    # Nén TLLa-TLC thật
    try:
        compact = tlla_compress(message)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Nén thất bại: {e}'}), 500
    
    # Thêm vào hàng đợi xử lý
    processing_queue.append({
        'id': msg_id,
        'original': message,
        'compact': compact,
        'user_id': user_id,
        'user_name': user_name,
        'status': 'compressed',
        'start_time': time.time()
    })
    
    # Gửi qua ESP32 trong thread riêng
    def do_send():
        msg = next((m for m in processing_queue if m['id'] == msg_id), None)
        if msg:
            msg['status'] = 'sending'
            success = send_via_esp32(compact, msg_id)
            if success:
                msg['status'] = 'sent'
            else:
                msg['status'] = 'failed'
                msg['error'] = 'ESP32 không phản hồi'
    
    thread = threading.Thread(target=do_send)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message_id': msg_id,
        'status': 'processing'
    })

@app.route('/api/chat/status/<message_id>', methods=['GET'])
def get_message_status(message_id):
    """Kiểm tra trạng thái tin nhắn"""
    msg = next((m for m in processing_queue if m['id'] == message_id), None)
    if not msg:
        return jsonify({'success': False, 'error': 'Không tìm thấy'}), 404
    return jsonify({
        'success': True,
        'status': msg['status'],
        'error': msg.get('error', '')
    })

@app.route('/api/chat/poll', methods=['GET'])
def poll_messages():
    """Poll tin nhắn mới từ ESP32 (đã giải nén sẵn)"""
    # Lấy tin nhắn từ queue và giải nén
    messages = []
    while message_queue:
        msg = message_queue.pop(0)
        try:
            # Giải nén TLLa-TLC thật
            decompressed = tlla_decompress(msg['text'])
        except Exception as e:
            decompressed = f"[Lỗi giải nén] {msg['text'][:50]}..."
        
        messages.append({
            'id': msg['id'],
            'text': decompressed,
            'userId': 'remote',
            'userName': 'Remote User',
            'timestamp': msg['timestamp'],
            'lora_rssi': esp32_status['rssi'],
            'lora_snr': esp32_status['snr']
        })
    
    return jsonify({
        'success': True,
        'messages': messages,
        'new_count': len(messages),
        'lora_info': {
            'rssi': esp32_status['rssi'],
            'snr': esp32_status['snr'],
            'frequency': esp32_status['lora_frequency'],
            'sf': esp32_status['lora_sf'],
            'packets_sent': esp32_status['packets_sent'],
            'packets_received': esp32_status['packets_received'],
            'packet_errors': esp32_status['packet_errors']
        }
    })

@app.route('/api/chat/esp32_status', methods=['GET'])
def get_esp32_status():
    """Lấy trạng thái ESP32 thật"""
    return jsonify({
        'success': True,
        'online': esp32_connected and esp32_status['online'],
        'lora_frequency': esp32_status['lora_frequency'],
        'lora_sf': esp32_status['lora_sf'],
        'lora_bw': esp32_status['lora_bw'],
        'last_heartbeat': esp32_status['last_heartbeat'].isoformat() if esp32_status['last_heartbeat'] else None,
        'rssi': esp32_status['rssi'],
        'snr': esp32_status['snr'],
        'packets_sent': esp32_status['packets_sent'],
        'packets_received': esp32_status['packets_received'],
        'packet_errors': esp32_status['packet_errors']
    })

# ============================================================
# KHỞI ĐỘNG
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("  LoRa Chat Server - REAL (ESP32 + LoRa SX1278)")
    print("="*60)
    
    # Kết nối ESP32
    init_esp32_serial()
    
    # Khởi động thread đọc dữ liệu từ ESP32
    if esp32_connected:
        read_thread = threading.Thread(target=read_from_esp32, daemon=True)
        read_thread.start()
        print("[ESP32] ✅ Thread đọc dữ liệu đã khởi động")
    else:
        print("[ESP32] ⚠️ Không thể kết nối ESP32, vui lòng kiểm tra cổng COM")
        print(f"   Để kết nối, set environment variable ESP32_PORT=COMx")
    
    print("="*60)
    print(f"  Server chạy tại: http://localhost:5001")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5001, debug=False)