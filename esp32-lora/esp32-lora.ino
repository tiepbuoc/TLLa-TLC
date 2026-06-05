/*
 * ESP32-C3 Super Mini + LoRa SX1278 (RA-02)
 * Giao tiếp LoRa thật - KHÔNG FAKE
 * 
 * Hardware:
 * - ESP32-C3 Super Mini
 * - LoRa RA-02 (SX1278) @ 433MHz
 * 
 * Wiring:
 * - NSS     -> GPIO 5
 * - SCK     -> GPIO 18
 * - MOSI    -> GPIO 23
 * - MISO    -> GPIO 19
 * - RST     -> GPIO 14
 * - DIO0    -> GPIO 2
 */

#include <Arduino.h>
#include <SPI.h>
#include <LoRa.h>
#include <CRC.h>

// ============================================================
// CẤU HÌNH CHÂN
// ============================================================
#define LORA_SS      5
#define LORA_SCK     18
#define LORA_MOSI    23
#define LORA_MISO    19
#define LORA_RST     14
#define LORA_DIO0    2
#define LED_BUILTIN  8

// ============================================================
// THAM SỐ LoRa
// ============================================================
#define LORA_FREQUENCY   433E6    // 433 MHz
#define LORA_SF          12       // Spreading Factor (6-12)
#define LORA_BW          125E3    // Bandwidth (125 kHz)
#define LORA_CR          8        // Coding Rate (4/8)
#define LORA_PREAMBLE    8        // Preamble length
#define LORA_TX_POWER    20       // TX power (dBm, max 20)

// ============================================================
// CẤU HÌNH SERIAL
// ============================================================
#define SERIAL_BAUD      115200

// ============================================================
// BUFFER
// ============================================================
#define MAX_PACKET_SIZE  256
uint8_t rxBuffer[MAX_PACKET_SIZE];
uint8_t txBuffer[MAX_PACKET_SIZE];

// ============================================================
// THỐNG KÊ
// ============================================================
volatile uint32_t packetsSent = 0;
volatile uint32_t packetsReceived = 0;
volatile uint32_t packetErrors = 0;
volatile int32_t lastRSSI = -100;
volatile float lastSNR = 0;
uint32_t lastHeartbeat = 0;
bool loraInitialized = false;

// CRC16 tính toán
uint16_t calculateCRC16(const uint8_t* data, size_t len) {
  return CRC::crc16(data, len);
}

// ============================================================
// KHỞI TẠO LoRa
// ============================================================
bool initLoRa() {
  Serial.println("[LoRa] Đang khởi tạo module...");
  
  // Cấu hình SPI
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_SS);
  
  // Khởi tạo LoRa
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  
  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("[LoRa] ❌ Khởi tạo thất bại!");
    Serial.println("   Kiểm tra wiring và nguồn");
    return false;
  }
  
  // Cấu hình tham số
  LoRa.setSpreadingFactor(LORA_SF);
  LoRa.setSignalBandwidth(LORA_BW);
  LoRa.setCodingRate4(LORA_CR);
  LoRa.setPreambleLength(LORA_PREAMBLE);
  LoRa.setTxPower(LORA_TX_POWER);
  LoRa.enableCrc();  // Bật CRC hardware
  
  Serial.println("[LoRa] ✅ Khởi tạo thành công!");
  Serial.print("   Tần số: ");
  Serial.print(LORA_FREQUENCY / 1E6);
  Serial.println(" MHz");
  Serial.print("   Spreading Factor: ");
  Serial.println(LORA_SF);
  Serial.print("   Bandwidth: ");
  Serial.print(LORA_BW / 1E3);
  Serial.println(" kHz");
  
  return true;
}

// ============================================================
// GỬI DỮ LIỆU QUA LoRa
// ============================================================
bool sendLoRaPacket(const uint8_t* data, size_t len) {
  if (!loraInitialized) return false;
  
  // Bắt đầu gói tin
  LoRa.beginPacket();
  LoRa.write(data, len);
  
  // Kết thúc và gửi
  LoRa.endPacket();
  
  packetsSent++;
  Serial.printf("[LoRa] 📤 Đã gửi %d bytes\n", len);
  
  return true;
}

// ============================================================
// NHẬN DỮ LIỆU QUA LoRa
// ============================================================
void onLoRaReceive(int packetSize) {
  if (packetSize == 0) return;
  
  // Đọc RSSI và SNR
  lastRSSI = LoRa.packetRssi();
  lastSNR = LoRa.packetSnr();
  
  // Đọc dữ liệu
  int i = 0;
  while (LoRa.available() && i < MAX_PACKET_SIZE - 1) {
    rxBuffer[i++] = LoRa.read();
  }
  
  // Kiểm tra CRC (hardware đã check, nhưng vẫn kiểm tra thêm)
  if (LoRa.crc()) {
    packetErrors++;
    Serial.println("[LoRa] ❌ CRC error");
    return;
  }
  
  packetsReceived++;
  
  // Tính toán RSSI/SNR indicators
  int quality = 0;
  if (lastSNR > 10) quality = 100;
  else if (lastSNR > 5) quality = 80;
  else if (lastSNR > 0) quality = 60;
  else if (lastSNR > -5) quality = 40;
  else quality = 20;
  
  Serial.printf("[LoRa] 📥 Nhận %d bytes | RSSI: %d dBm | SNR: %.1f dB | Quality: %d%%\n",
                i, lastRSSI, lastSNR, quality);
  
  // Gửi lên Serial để server đọc
  Serial.write(rxBuffer, i);
  Serial.println(); // Kết thúc dòng
}

// ============================================================
// GỬI HEARTBEAT
// ============================================================
void sendHeartbeat() {
  if (!loraInitialized) return;
  
  char hbMsg[64];
  snprintf(hbMsg, sizeof(hbMsg), "HB:%lu|SF:%d|FREQ:%.0f|P:%lu|R:%lu|E:%lu",
           millis(), LORA_SF, LORA_FREQUENCY / 1E6,
           packetsSent, packetsReceived, packetErrors);
  
  LoRa.beginPacket();
  LoRa.print(hbMsg);
  LoRa.endPacket();
  
  Serial.printf("[LoRa] 💓 Heartbeat sent: %s\n", hbMsg);
}

// ============================================================
// XỬ LÝ LỆNH TỪ SERIAL (từ server)
// ============================================================
void processSerialCommand() {
  if (!Serial.available()) return;
  
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  
  if (cmd == "STATUS") {
    Serial.println("ESP32-C3 + LoRa SX1278");
    Serial.printf("Online: %s\n", loraInitialized ? "YES" : "NO");
    Serial.printf("LoRa Freq: %.0f MHz\n", LORA_FREQUENCY / 1E6);
    Serial.printf("SF: %d, BW: %.0f kHz\n", LORA_SF, LORA_BW / 1E3);
    Serial.printf("Packets sent: %lu\n", packetsSent);
    Serial.printf("Packets received: %lu\n", packetsReceived);
    Serial.printf("Packet errors: %lu\n", packetErrors);
    Serial.printf("Last RSSI: %d dBm\n", lastRSSI);
    Serial.printf("Last SNR: %.1f dB\n", lastSNR);
  }
  else if (cmd == "RESET") {
    Serial.println("Resetting counters...");
    packetsSent = 0;
    packetsReceived = 0;
    packetErrors = 0;
  }
  else if (cmd.startsWith("SEND:")) {
    // Gửi tin nhắn qua LoRa
    String msg = cmd.substring(5);
    LoRa.beginPacket();
    LoRa.print(msg);
    LoRa.endPacket();
    packetsSent++;
    Serial.printf("[LoRa] 📤 Sent: %s\n", msg.c_str());
  }
}

// ============================================================
// CẬP NHẬT LED
// ============================================================
void updateLED() {
  static unsigned long lastBlink = 0;
  static bool ledState = false;
  
  // Blink theo tốc độ dựa trên chất lượng tín hiệu
  unsigned long interval = 5000; // Default 5s
  
  if (loraInitialized && lastSNR > 0) {
    interval = 2000;  // Tín hiệu tốt -> blink nhanh
  } else if (loraInitialized && lastSNR > -10) {
    interval = 3000;  // Tín hiệu trung bình
  } else {
    interval = 5000;  // Tín hiệu yếu hoặc không có
  }
  
  if (millis() - lastBlink > interval) {
    ledState = !ledState;
    digitalWrite(LED_BUILTIN, ledState);
    lastBlink = millis();
  }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  // Khởi tạo Serial
  Serial.begin(SERIAL_BAUD);
  delay(1000);
  
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  
  Serial.println("\n==========================================");
  Serial.println("  ESP32-C3 Super Mini + LoRa SX1278");
  Serial.println("  REAL LoRa Communication");
  Serial.println("==========================================\n");
  
  // Khởi tạo LoRa
  loraInitialized = initLoRa();
  
  if (loraInitialized) {
    // Đăng ký callback nhận dữ liệu
    LoRa.onReceive(onLoRaReceive);
    LoRa.receive();  // Chuyển sang chế độ nhận
    
    // Blink LED báo hiệu sẵn sàng
    for (int i = 0; i < 3; i++) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(200);
      digitalWrite(LED_BUILTIN, LOW);
      delay(200);
    }
    
    Serial.println("[ESP32] ✅ Ready to send/receive via LoRa");
  } else {
    Serial.println("[ESP32] ❌ LoRa initialization failed");
  }
  
  lastHeartbeat = millis();
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  // Xử lý lệnh từ Serial (từ server)
  processSerialCommand();
  
  // Gửi heartbeat mỗi 30 giây
  if (loraInitialized && (millis() - lastHeartbeat > 30000)) {
    sendHeartbeat();
    lastHeartbeat = millis();
  }
  
  // Cập nhật LED
  updateLED();
  
  // Đợi ngắt LoRa
  delay(10);
}