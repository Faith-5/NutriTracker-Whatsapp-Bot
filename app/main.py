from fastapi import FastAPI, Request
from app.whatsapp import send_whatsapp_message
from app.bot.chain import run_bot
from app.config import VERIFY_TOKEN

app = FastAPI()

@app.get("/")
async def verify(request: Request):
    params = dict(request.query_params)

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ WEBHOOK VERIFIED")
        return int(challenge)

    return {"error": "Invalid token"}, 403

@app.post("/")
async def webhook(request: Request):
    data = await request.json()
    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        user_text = message["text"]["body"]
        user_number = message["from"]  # This becomes session_id

        bot_reply = run_bot(user_text, session_id=user_number)
        send_whatsapp_message(user_number, bot_reply)

    except Exception as e:
        print("Error:", e)

    return {"status": "ok"}

