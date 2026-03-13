# CRS API Integration Guide

Everything you need to call the College Recommendation System API from any frontend or tool.

---

## Quick Reference

| | |
|---|---|
| **Live URL** | `https://api.rabiaryal.com.np` |
| **Local URL** | `http://localhost:8000` |
| **Auth header** | `x-api-key: demo-secret-2026` |
| **Chat endpoint** | `POST /api/v1/chat` |
| **Health endpoint** | `GET /api/v1/health` |
| **Docs (Swagger)** | `http://localhost:8000/docs` |

---

## Authentication

Every request to `/api/v1/chat` must include this header:

```
x-api-key: demo-secret-2026
```

Missing or wrong key returns:
```json
HTTP 401 Unauthorized
{ "detail": "Missing or incorrect x-api-key" }
```

> To change the key, edit `DEMO_API_KEY` in `backend/app/api/auth.py` and restart the server.
> Or set the env var: `export DEMO_API_KEY="your-new-key"`

---

## Endpoints

### POST `/api/v1/chat` — Send a message

**Required headers:**

| Header | Value |
|---|---|
| `x-api-key` | `demo-secret-2026` |
| `Content-Type` | `application/json` |

**Request body:**

```json
{
  "session_id": "abc-123",
  "message": "show me colleges under 5 lakh"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✅ Yes | The user's message (1–1000 chars) |
| `session_id` | string | ⚠️ Recommended | Keeps conversation context across turns. Omit it and every message starts fresh with no memory. |

**Response body:**

```json
{
  "message": "Here are colleges within your budget of Rs. 5,00,000...",
  "session_id": "abc-123",
  "intent": "recommend_with_constraints",
  "entities": {
    "budget": 500000
  },
  "confidence": 0.91,
  "timestamp": "2026-03-11T10:30:00",
  "debug_info": {}
}
```

| Field | Type | Description |
|---|---|---|
| `message` | string | Chatbot reply (may contain Markdown) |
| `session_id` | string | Echo of your session ID — save and reuse for follow-ups |
| `intent` | string | Detected intent label |
| `entities` | object | Extracted slots (budget, rank, course, college_name, etc.) |
| `confidence` | float | Intent confidence score (0.0–1.0) |
| `timestamp` | string | ISO 8601 datetime |
| `debug_info` | object | Internal pipeline info (can be ignored) |

---

### GET `/api/v1/health` — Basic health check

No auth required.

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2026-03-11T10:30:00"
}
```

### GET `/api/v1/health/detailed` — Full system status

No auth required. Returns DB connection, model load status, and uptime.

---

## Session Management

The `session_id` is how the server remembers what a user said previously.

```
Turn 1:  "show me colleges under 5 lakh"   →  bot asks: "which course?"
Turn 2:  "computer"                         →  bot returns recommendations

Both turns MUST use the same session_id.
```

**Rules:**
- Generate `session_id` **once** when the user opens the chat (e.g. `crypto.randomUUID()` in JS).
- Send that **same ID** with every message in the conversation.
- To reset, either send `"message": "clear"` or generate a new `session_id`.

---

## Usage Examples

### cURL

```bash
# First message
curl -X POST https://api.rabiaryal.com.np/api/v1/chat \
     -H "x-api-key: demo-secret-2026" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my-session-1", "message": "show me colleges under 5 lakh"}'

# Follow-up — same session_id so the bot remembers budget
curl -X POST https://api.rabiaryal.com.np/api/v1/chat \
     -H "x-api-key: demo-secret-2026" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my-session-1", "message": "computer"}'

# Reset the conversation
curl -X POST https://api.rabiaryal.com.np/api/v1/chat \
     -H "x-api-key: demo-secret-2026" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my-session-1", "message": "clear"}'
```

---

### Postman

1. **Method:** `POST`
2. **URL:** `https://api.rabiaryal.com.np/api/v1/chat`
3. **Headers tab:**

   | Key | Value |
   |---|---|
   | `x-api-key` | `demo-secret-2026` |
   | `Content-Type` | `application/json` |

4. **Body tab** → raw → JSON:
   ```json
   {
     "session_id": "test-session-1",
     "message": "compare KEC and IOE"
   }
   ```

**Auto-capture `session_id` between requests** — paste this in the **Tests** tab of your first request:
```javascript
var res = pm.response.json();
pm.environment.set("session_id", res.session_id);
```

Then every subsequent body can use:4
```json
{
  "session_id": "{{session_id}}",
  "message": "which one has hostel?"
}
```

---

### React / JavaScript (fetch)

**`src/chatApi.js`** — copy this into your project:

```javascript
const API_BASE = "https://api.rabiaryal.com.np";
const API_KEY  = "demo-secret-2026";

// Call once when the chat window opens
export function createSessionId() {
  return crypto.randomUUID();
}

export async function sendMessage(sessionId, message) {
  const response = await fetch(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
    }),
  });

  if (response.status === 401) throw new Error("Invalid API key");
  if (!response.ok) throw new Error(`Server error: ${response.status}`);

  return response.json(); // returns full ChatResponse object
}
```

**React chat component:**

```jsx
import { useState, useRef } from "react";
import { createSessionId, sendMessage } from "./chatApi";

export default function ChatWidget() {
  const sessionId = useRef(createSessionId()); // fixed for this tab session
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);

  async function handleSend() {
    if (!input.trim()) return;
    const userMsg = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", text: userMsg }]);
    setLoading(true);

    try {
      const data = await sendMessage(sessionId.current, userMsg);
      setMessages(prev => [...prev, { role: "bot", text: data.message }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: "bot", text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div>
        {messages.map((m, i) => (
          <p key={i}><strong>{m.role}:</strong> {m.text}</p>
        ))}
        {loading && <p>Thinking...</p>}
      </div>
      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === "Enter" && handleSend()}
        placeholder="Ask about colleges..."
      />
      <button onClick={handleSend}>Send</button>
    </div>
  );
}
```

---

### Python (requests)

```python
import requests
import uuid

API_BASE = "https://api.rabiaryal.com.np"
API_KEY  = "demo-secret-2026"
HEADERS  = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json",
}

session_id = str(uuid.uuid4())  # generate once per conversation

def chat(message: str) -> str:
    response = requests.post(
        f"{API_BASE}/api/v1/chat",
        headers=HEADERS,
        json={"session_id": session_id, "message": message},
    )
    response.raise_for_status()
    return response.json()["message"]

# Multi-turn example
print(chat("show me colleges under 5 lakh"))  # bot asks for course
print(chat("computer"))                        # bot returns results
print(chat("which ones have hostel?"))         # bot filters from previous results
```

---

### Next.js / Node.js (server-side proxy)

```javascript
// pages/api/chat.js  (Pages Router)
// or app/api/chat/route.js  (App Router)

const API_BASE = "https://api.rabiaryal.com.np";
const API_KEY  = "demo-secret-2026"; // stays server-side — never sent to browser

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  const { session_id, message } = req.body;

  const upstream = await fetch(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
    },
    body: JSON.stringify({ session_id, message }),
  });

  const data = await upstream.json();
  res.status(upstream.status).json(data);
}
```

> The API key never reaches the browser — all requests go through your Next.js server.

---

### Flutter / Dart

Add to `pubspec.yaml`:
```yaml
dependencies:
  http: ^1.2.0
  uuid: ^4.3.3
```

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:uuid/uuid.dart';

const String _apiBase = 'https://api.rabiaryal.com.np';
const String _apiKey  = 'demo-secret-2026';

class CrsApiService {
  final String sessionId = const Uuid().v4(); // one per conversation

  Future<String> sendMessage(String message) async {
    final response = await http.post(
      Uri.parse('$_apiBase/api/v1/chat'),
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': _apiKey,
      },
      body: jsonEncode({
        'session_id': sessionId,
        'message': message,
      }),
    );

    if (response.statusCode == 401) throw Exception('Invalid API key');
    if (response.statusCode != 200) throw Exception('Server error ${response.statusCode}');

    final data = jsonDecode(response.body);
    return data['message'] as String;
  }
}
```

---

## What the Bot Can Answer

| Example message | Intent triggered |
|---|---|
| `hello` / `hi` | `greeting` |
| `tell me about KEC` | `college_details` |
| `compare IOE and KU` | `compare_colleges` |
| `show colleges in Kathmandu` | `search_college` |
| `top rated colleges` | `best_items_search` |
| `recommend colleges for rank 500, budget 6 lakh` | `personalized_recommendation` |
| `colleges under 5 lakh for computer` | `recommend_with_constraints` |
| `does KEC have hostel?` | `hostel_query` |
| `contact number of Sagarmatha college` | `contact_query` |
| `what is the fee of KU?` | `college_attribute_query` |
| `how to get admission in KEC?` | `admission_process` |
| `clear` | resets session context |

---

## Intent & Entity Reference

**12 Intent labels:**
`greeting`, `goodbye`, `college_details`, `compare_colleges`, `search_college`,
`best_items_search`, `personalized_recommendation`, `recommend_with_constraints`,
`hostel_query`, `contact_query`, `college_attribute_query`, `admission_process`

**9 Entity types:**

| Entity | Example input |
|---|---|
| `COLLEGE_NAME` | "KEC", "Kathmandu University" |
| `COURSE` | "Computer", "Civil", "BE Computer" |
| `LOCATION` | "Kathmandu", "Lalitpur" |
| `BUDGET` | "5 lakh", "600000" |
| `RANK` | "500", "rank 1200" |
| `COLLEGE_TYPE` | "government", "private" |
| `ATTRIBUTE` | "fee", "rating", "hostel" |
| `COLLEGE_NAME_1` | first college in a comparison query |
| `COLLEGE_NAME_2` | second college in a comparison query |

**Budget normalization:**

| User input | Interpreted as |
|---|---|
| `"5 lakh"` | `500000` |
| `"600000"` | `600000` |
| `"7"` *(bare number < 1000)* | `700000` |
| `"1.5 lakh"` | `150000` |

---

## Running the Backend

```bash
conda activate bctproject
cd backend
python -m app.main
```

Swagger UI: `http://localhost:8000/docs`

**With Cloudflare Tunnel** (separate terminal):
```bash
cloudflared tunnel run
```

Both must run simultaneously for `https://api.rabiaryal.com.np` to work.

---

## Changing the API Key

**Option 1 — Edit the file** (`backend/app/api/auth.py`):
```python
DEMO_API_KEY = os.getenv("DEMO_API_KEY", "your-new-key-here")
```

**Option 2 — Environment variable** (no file edit needed):
```bash
export DEMO_API_KEY="your-new-key"
python -m app.main
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `401 Unauthorized` | Check `x-api-key` value — must be `demo-secret-2026` |
| Follow-up loses context | Send the same `session_id` in every request |
| Bot treats follow-up as new conversation | You likely omitted or changed `session_id` |
| Backend not reachable via tunnel | Ensure both `python -m app.main` and `cloudflared tunnel run` are running |
| CORS error from browser | Backend allows `localhost:3000` and `localhost:3001` by default; update `config.py` for other origins |
