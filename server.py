"""
server.py -- Backend της Linda για το Hebden Bridge Town Hall kiosk.

Τρέχει στο Render (cloud), ΟΧΙ στο iPad. Το iPad μιλάει σε αυτό μέσω
WebSocket -- στέλνει ό,τι είπε/έγραψε ο επισκέπτης, παίρνει πίσω την
απάντηση της Linda σε κείμενο (το iPad κάνει το speech-to-text /
text-to-speech μέσα στον browser, όχι εδώ).

Ρύθμιση στο Render:
  - Build Command:  pip install -r requirements.txt
  - Start Command:  uvicorn server:app --host 0.0.0.0 --port $PORT

Environment variables που χρειάζεται (τα βάζεις στο Render dashboard):
  OPENAI_API_KEY          (υποχρεωτικό)
  STAFF_PHONE_NUMBER      (προαιρετικό -- SMS escalation)
  TWILIO_ACCOUNT_SID      (προαιρετικό)
  TWILIO_AUTH_TOKEN       (προαιρετικό)
  TWILIO_FROM_NUMBER      (προαιρετικό)
"""

import os
import json
import time
from collections import defaultdict, deque

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

app = FastAPI()

# ------------------------------------------------------------------
# Απλό rate limiting -- προστασία από κατάχρηση/υπερβολικό κόστος OpenAI
# (in-memory, αρκετό για ένα kiosk· δεν επιβιώνει restart, ok για τη χρήση μας)
# ------------------------------------------------------------------
RATE_LIMIT_WINDOW = 60          # δευτερόλεπτα
RATE_LIMIT_MAX_CONVERSE = 15    # ανά IP ανά λεπτό -- αρκετό για κανονική χρήση kiosk
RATE_LIMIT_MAX_SETMODE = 5

_request_log = defaultdict(deque)


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(key: str, max_requests: int) -> bool:
    now = time.time()
    log = _request_log[key]
    while log and now - log[0] > RATE_LIMIT_WINDOW:
        log.popleft()
    if len(log) >= max_requests:
        return False
    log.append(now)
    return True

ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(BASE_DIR, "static", "photos")

os.makedirs(PHOTOS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# ΣΗΜΕΙΩΣΗ: αυτό μηδενίζεται σε "linda" αν το δωρεάν instance του Render
# κάνει restart (π.χ. μετά από αδράνεια) -- αν το slideshow mode χρειάζεται
# να "κρατάει" μόνιμα, θα χρειαστεί persistence αργότερα.
current_mode = {"mode": "linda"}

# ------------------------------------------------------------------
# ΡΥΘΜΙΣΕΙΣ
# ------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CHATGPT_MODEL = "gpt-4o-mini"
OPENAI_SEARCH_MODEL = "gpt-4o-mini"

BUILDING_NAME = "Hebden Bridge Town Hall"
USER_LOCATION_CITY = "Hebden Bridge"
USER_LOCATION_COUNTRY = "GB"

STAFF_PHONE_NUMBER = os.environ.get("STAFF_PHONE_NUMBER", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "")

_twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    from twilio.rest import Client as TwilioClient
    _twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

client = OpenAI(api_key=OPENAI_API_KEY)

BUILDING_INFO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "building_info.txt")


def load_building_info() -> str:
    """Ίδια λογική με πριν -- διαβάζει το αρχείο ΦΡΕΣΚΟ σε κάθε νέα συνομιλία."""
    try:
        with open(BUILDING_INFO_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[building_info] Δεν διαβάστηκε το αρχείο ({e})")
        return "(No building information file found.)"


def build_system_prompt() -> str:
    building_info = load_building_info()
    return (
        f"You are Linda, the friendly voice information assistant at {BUILDING_NAME}, a public "
        "building in Hebden Bridge, West Yorkshire, UK. You speak to visitors who walk up "
        "to a kiosk screen -- including elderly visitors and tourists who may be unfamiliar "
        "with the building or the area. "
        "Speak warmly, clearly, and briefly -- like a helpful receptionist, not a robot. "
        "No markdown, no lists -- you're speaking out loud. "
        "\n\n"
        "STRICT RULE about the Town Hall building itself (toilets, reception, opening hours, "
        "cafe, lift, council offices, wifi, any notices): use ONLY the information below. "
        "NEVER search the internet or guess about the building -- if something isn't covered "
        "here, say you're not sure and offer to get a member of staff.\n\n"
        f"{building_info}\n\n"
        "\n\n"
        "For EVERYTHING ELSE about Hebden Bridge and the surrounding area -- pubs, cafes "
        "(other than the Town Hall's own), shops, events, attractions, directions, "
        "recommendations, weather, anything current or specific -- use the web_search tool "
        "to get real, up to date information, then answer concisely in your own words. Don't "
        "rely on your own general knowledge for these, it may be outdated or wrong. "
        "For directions/maps to places OUTSIDE the building, use open_maps. For pictures, use "
        "open_image_search. "
        "If a visitor explicitly asks to speak to a person, seems confused, distressed, or has "
        "a need you genuinely cannot help with, use the request_human_assistance tool -- and "
        "reassure them warmly that someone is on the way."
    )


# ------------------------------------------------------------------
# Εργαλεία -- SMS escalation + web search γίνονται ΕΔΩ (στο server).
# open_maps/open_image_search ΔΕΝ μπορούν να ανοίξουν browser tab από το
# server -- επιστρέφονται σαν "action" στο iPad, που τα ανοίγει το ίδιο.
# ------------------------------------------------------------------

CLIENT_ACTIONS = {"open_maps", "open_image_search"}


def action_request_human_assistance(visitor_description: str) -> str:
    if not _twilio_client or not STAFF_PHONE_NUMBER or not TWILIO_FROM_NUMBER:
        print(f"[ESCALATION -- SMS not configured] {visitor_description}")
        return (
            "I've noted that you need in-person help, but I'm not yet able to send an "
            "alert -- someone should be along shortly. Please take a seat if you can."
        )
    try:
        _twilio_client.messages.create(
            body=f"[Linda @ {BUILDING_NAME}] Visitor needs assistance: {visitor_description}",
            from_=TWILIO_FROM_NUMBER,
            to=STAFF_PHONE_NUMBER,
        )
        return (
            "I've just sent a message to a member of staff -- someone should be with "
            "you shortly. Thank you for your patience!"
        )
    except Exception as e:
        print(f"[SMS escalation error]: {type(e).__name__}: {e}")
        return (
            "I tried to alert a member of staff but something went wrong on my end. "
            "Please take a seat if you can, and someone should notice you shortly."
        )


def action_web_search(query: str) -> str:
    if not OPENAI_API_KEY:
        return "SEARCH_TOOL_ERROR: OPENAI_API_KEY is not set."
    web_search_tool = {
        "type": "web_search_preview",
        "user_location": {
            "type": "approximate",
            "approximate": {"city": USER_LOCATION_CITY, "country": USER_LOCATION_COUNTRY},
        },
    }
    try:
        response = client.responses.create(
            model=OPENAI_SEARCH_MODEL, tools=[web_search_tool], input=query,
        )
        return getattr(response, "output_text", None) or "SEARCH_TOOL_ERROR: empty response."
    except Exception as e:
        print(f"[web_search tool error]: {type(e).__name__}: {e}")
        try:
            response = client.responses.create(
                model=OPENAI_SEARCH_MODEL, tools=[{"type": "web_search_preview"}], input=query,
            )
            return getattr(response, "output_text", None) or "SEARCH_TOOL_ERROR: empty response."
        except Exception as e2:
            print(f"[web_search tool error, retry]: {type(e2).__name__}: {e2}")
            return f"SEARCH_TOOL_ERROR: {type(e2).__name__}: {e2}"


ACTION_DISPATCH = {
    "request_human_assistance": lambda inp: action_request_human_assistance(inp["visitor_description"]),
    "web_search": lambda inp: action_web_search(inp["query"]),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "request_human_assistance",
            "description": "Alerts a member of staff by SMS that a visitor needs in-person help.",
            "parameters": {
                "type": "object",
                "properties": {"visitor_description": {"type": "string"}},
                "required": ["visitor_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_maps",
            "description": "Opens Google Maps for a location OUTSIDE the building.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Real internet search for GENERAL Hebden Bridge info (pubs, shops, "
                            "events) -- NEVER for the Town Hall building itself.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_image_search",
            "description": "Opens Google Images for a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def ask_gpt(user_text: str, history: list):
    """ΣΗΜΕΙΩΣΗ: το `history` ΔΕΝ περιέχει ακόμα το μήνυμα του χρήστη -- το προσθέτουμε εδώ."""
    history.append({"role": "user", "content": user_text})
    client_actions = []

    while True:
        response = client.chat.completions.create(
            model=CHATGPT_MODEL, messages=history, tools=TOOLS, tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            history.append({"role": "assistant", "content": msg.content or "I'm here to help."})
            return msg.content or "I'm here to help.", client_actions

        history.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name in CLIENT_ACTIONS:
                client_actions.append({"action": name, "args": args})
                result = "(Will be shown to the visitor on their screen.)"
            else:
                fn = ACTION_DISPATCH.get(name)
                try:
                    result = fn(args) if fn else f"Unknown tool: {name}"
                except Exception as e:
                    result = f"Error while executing: {e}"

            history.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})


# ------------------------------------------------------------------
# WebSocket endpoint -- το iPad συνδέεται εδώ
# ------------------------------------------------------------------

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "linda-backend"}


@app.get("/kiosk")
async def kiosk_page():
    """Αυτή τη διεύθυνση ανοίγει το iPad στο Safari."""
    html_path = os.path.join(BASE_DIR, "linda_orb.html")
    return FileResponse(html_path)


@app.get("/admin")
async def admin_page():
    """Πάνελ ελέγχου -- εναλλαγή Linda / Slideshow mode εξ αποστάσεως."""
    html_path = os.path.join(BASE_DIR, "admin.html")
    return FileResponse(html_path)


@app.get("/mode")
async def get_mode():
    return JSONResponse(
        content=current_mode,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.post("/set-mode")
async def set_mode(request: Request, mode: str = Form(...), key: str = Form(...)):
    client_ip = _get_client_ip(request)
    if not _check_rate_limit(f"setmode:{client_ip}", RATE_LIMIT_MAX_SETMODE):
        return JSONResponse(status_code=429, content={"error": "rate_limited"})

    if not ADMIN_KEY or key != ADMIN_KEY:
        return {"error": "unauthorized"}
    if mode not in ("linda", "slideshow"):
        return {"error": "invalid mode"}
    current_mode["mode"] = mode
    return {"status": "ok", "mode": mode}


@app.get("/photos")
async def list_photos():
    try:
        files = sorted(
            f for f in os.listdir(PHOTOS_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
    except FileNotFoundError:
        files = []
    return {"photos": files}


@app.post("/converse")
async def converse(
    request: Request,
    history: str = Form("[]"),
    text: str = Form(None),
    audio: UploadFile = File(None),
):
    """Κύριο endpoint -- το iPad στέλνει είτε ηχητικό (audio) είτε γραπτό κείμενο
    (text, στο accessibility mode), μαζί με το ιστορικό της συνομιλίας μέχρι
    τώρα (το κρατάει ο browser, όχι ο server -- stateless). Επιστρέφει την
    απάντηση σε κείμενο + το ενημερωμένο ιστορικό."""
    client_ip = _get_client_ip(request)
    if not _check_rate_limit(f"converse:{client_ip}", RATE_LIMIT_MAX_CONVERSE):
        return JSONResponse(status_code=429, content={"error": "rate_limited"})

    try:
        prior_history = json.loads(history)
    except json.JSONDecodeError:
        prior_history = []

    if audio is not None:
        audio_bytes = await audio.read()
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=(audio.filename or "audio.webm", audio_bytes, audio.content_type or "audio/webm"),
            )
            user_text = (transcript.text or "").strip()
        except Exception as e:
            print(f"[transcription error]: {type(e).__name__}: {e}")
            return {"user_text": "", "reply_text": "", "history": prior_history, "actions": [], "error": "transcription_failed"}
    else:
        user_text = (text or "").strip()

    if not user_text:
        return {"user_text": "", "reply_text": "", "history": prior_history, "actions": []}

    full_history = [{"role": "system", "content": build_system_prompt()}] + prior_history
    reply_text, actions = ask_gpt(user_text, full_history)
    new_history = full_history[1:]  # αφαιρούμε το system message πριν το στείλουμε πίσω

    return {
        "user_text": user_text,
        "reply_text": reply_text,
        "history": new_history,
        "actions": actions,
    }