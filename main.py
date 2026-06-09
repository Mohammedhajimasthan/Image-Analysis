from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import os

# ======================
# Load ENV
# ======================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=API_KEY)

# ======================
# App Init
# ======================
app = FastAPI(title="Gemini Image Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Static Files
# ======================
app.mount("/static", StaticFiles(directory="."), name="static")

# ======================
# ✅ WORKING VISION MODEL
# ======================
model = genai.GenerativeModel("gemini-1.0-pro-vision")

# ======================
# GET → UI
# ======================
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# ======================
# POST → Analyze Image
# ======================
@app.post("/analyze")
async def analyze_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        image_bytes = await image.read()

        response = model.generate_content([
            prompt,
            {
                "mime_type": image.content_type,
                "data": image_bytes
            }
        ])

        description = response.text or "No response generated."

        encoded_image = base64.b64encode(image_bytes).decode()

        return JSONResponse({
            "description": description,
            "image": encoded_image
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"description": f"Error: {str(e)}"}
        )

# ======================
# Run Server
# ======================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)