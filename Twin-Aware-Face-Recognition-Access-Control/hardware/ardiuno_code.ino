#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ── WiFi ──────────────────────────────────────────────────────
const char* WIFI_SSID     = "wifi_name";
const char* WIFI_PASSWORD = "wifi_password";

// ── Right-side pins ───────────────────────────────────────────
#define LED_GREEN   26    // Green LED  (ALLOWED)
#define LED_RED     27    // Red LED    (NOT ALLOWED)
#define BUZZER_PIN  25    // Active buzzer
#define MOTOR_IN1   32    // L293D IN1
#define MOTOR_IN2   33    // L293D IN2

// ── I2C LCD (left-side pins 21=SDA, 22=SCL — ESP32 default) ──
// I2C address is usually 0x27 or 0x3F — check your module
// Change 0x27 to 0x3F if LCD shows nothing
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ── Duration ──────────────────────────────────────────────────
#define ACTION_DURATION 5000   // 5 seconds

// ── Web server on port 80 ─────────────────────────────────────
WebServer server(80);


// ══════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════

void allOff() {
  digitalWrite(LED_GREEN,  LOW);
  digitalWrite(LED_RED,    LOW);
  digitalWrite(MOTOR_IN1,  LOW);
  digitalWrite(MOTOR_IN2,  LOW);
  digitalWrite(BUZZER_PIN, LOW);
}

void lcdPrint(String line1, String line2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.setCursor(0, 1);
  lcd.print(line2);
}

void grantAccess(String name) {
  Serial.println(">>> ALLOWED: " + name);

  // LCD — row 1: ALLOWED, row 2: person name (max 16 chars)
  String displayName = name;
  if (displayName.length() > 16) displayName = displayName.substring(0, 16);
  lcdPrint("  ACCESS GRANTED", displayName);

  // Outputs ON
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(MOTOR_IN1, HIGH);
  digitalWrite(MOTOR_IN2, LOW);

  delay(ACTION_DURATION);           // 5 seconds

  // Outputs OFF
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(MOTOR_IN1, LOW);
  Serial.println("    Done.");
}

void denyAccess() {
  Serial.println(">>> NOT ALLOWED");

  // LCD — row 1: NOT ALLOWED, row 2: blank name area
  lcdPrint(" ACCESS DENIED  ", " Unknown Person ");

  // Outputs ON
  digitalWrite(LED_RED,    HIGH);
  digitalWrite(BUZZER_PIN, HIGH);

  delay(ACTION_DURATION);           // 5 seconds

  // Outputs OFF
  digitalWrite(LED_RED,    LOW);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("    Done.");
}

void lcdIdle() {
  lcdPrint("  Face Recog.   ", "   Waiting...   ");
}


// ══════════════════════════════════════════════════════════════
// /result — called by camera_server.py on PC
// Body: { "name": "harini", "allowed": true }
// ══════════════════════════════════════════════════════════════

void handleResult() {
  if (!server.hasArg("plain")) {
    server.send(400, "application/json", "{\"error\":\"No body\"}");
    return;
  }

  String body = server.arg("plain");
  Serial.println("\nReceived: " + body);

  StaticJsonDocument<128> doc;
  DeserializationError err = deserializeJson(doc, body);

  if (err) {
    Serial.println("JSON parse error!");
    server.send(400, "application/json", "{\"error\":\"Bad JSON\"}");
    return;
  }

  const char* name    = doc["name"]    | "UNKNOWN";
  bool        allowed = doc["allowed"] | false;

  allOff();

  if (String(name) == "NONE") {
    Serial.println(">>> No face — idle");
    lcdIdle();

  } else if (allowed) {
    grantAccess(String(name));

  } else {
    denyAccess();
  }

  // Return to idle display after action
  lcdIdle();

  server.send(200, "application/json", "{\"status\":\"ok\"}");
}


// ── /ping — PC checks if ESP32 is reachable ──────────────────
void handlePing() {
  server.send(200, "application/json", "{\"status\":\"ok\"}");
}


// ══════════════════════════════════════════════════════════════
// SETUP
// ══════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  delay(500);

  // GPIO setup
  pinMode(LED_GREEN,  OUTPUT);
  pinMode(LED_RED,    OUTPUT);
  pinMode(MOTOR_IN1,  OUTPUT);
  pinMode(MOTOR_IN2,  OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  allOff();

  // LCD setup
  Wire.begin(13, 14);       // SDA=21, SCL=22 (ESP32 default I2C)
  lcd.init();
  lcd.backlight();
  lcdPrint("  Initialising  ", "   Please wait  ");
  Serial.println("\n=== ESP32 Face Recognition ===");

  // Boot test — fire each output once briefly
  Serial.println("Testing outputs...");
  digitalWrite(LED_GREEN,  HIGH); delay(300); digitalWrite(LED_GREEN,  LOW);
  digitalWrite(LED_RED,    HIGH); delay(300); digitalWrite(LED_RED,    LOW);
  digitalWrite(MOTOR_IN1,  HIGH); delay(300); digitalWrite(MOTOR_IN1,  LOW);
  digitalWrite(BUZZER_PIN, HIGH); delay(300); digitalWrite(BUZZER_PIN, LOW);
  Serial.println("Test done.");

  // Connect WiFi
  lcdPrint(" Connecting WiFi", WIFI_SSID);
  Serial.printf("Connecting to WiFi: %s\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  String ip = WiFi.localIP().toString();
  Serial.println("\nWiFi connected!");
  Serial.println("ESP32 IP: " + ip);
  Serial.println(">>> Copy this IP into camera_server.py as LORA32_IP <<<\n");

  // Show IP on LCD for 3 seconds so you can copy it
  lcdPrint("  WiFi Connected", ip);
  delay(3000);

  // Register endpoints
  server.on("/result", HTTP_POST, handleResult);
  server.on("/ping",   HTTP_GET,  handlePing);
  server.begin();

  Serial.println("Web server started. Waiting for results...\n");
  lcdIdle();
}


// ══════════════════════════════════════════════════════════════
// LOOP
// ══════════════════════════════════════════════════════════════

void loop() {
  server.handleClient();
}
