import re
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from litellm import completion
from pydantic import BaseModel

load_dotenv()

# --- Config ---

MODEL = "vertex_ai/gemini-2.0-flash-lite"

MENU = {
    "chicken_teriyaki_bowl": {
        "tags": ["high_protein", "filling", "quick", "mild", "rice"],
    },
    "spicy_chicken_rice_bowl": {
        "tags": ["high_protein", "filling", "quick", "spicy", "rice"],
    },
    "beef_bibimbap": {
        "tags": ["high_protein", "filling", "medium", "spicy", "rice", "korean"],
    },
    "tofu_veggie_stir_fry": {
        "tags": ["vegetarian", "quick", "light", "rice", "vegan_option"],
    },
    "lentil_soup": {
        "tags": ["vegetarian", "light", "medium", "comfort"],
    },
    "turkey_chili": {
        "tags": ["high_protein", "filling", "spicy", "medium", "comfort"],
    },
    "egg_fried_rice": {
        "tags": ["quick", "filling", "rice", "mild", "vegetarian_option"],
    },
    "pesto_pasta": {
        "tags": ["comfort", "filling", "medium", "italian"],
    },
    "chicken_caesar_salad": {
        "tags": ["high_protein", "light", "quick", "salad"],
    },
    "salmon_rice_bowl": {
        "tags": ["high_protein", "quick", "mild", "rice", "seafood"],
    },
    "shrimp_tacos": {
        "tags": ["quick", "filling", "seafood", "mexican_option"],
    },
    "veggie_burrito_bowl": {
        "tags": ["vegetarian", "filling", "quick", "mexican_option", "rice"],
    },
}

SYSTEM_PROMPT = """\
<role>
You are DishDecider, a dish recommendation assistant. Your job is to help the user choose exactly one dish from a fixed catalog of dishes.
You speak naturally like a helpful friend. Keep questions short and conversational.
</role>

<scope_positive>
You provide these services:
1) Ask up to 3 short clarifying questions total across the conversation to learn preferences (diet/allergies, craving, protein preference, base like rice/pasta/soup/salad, budget, ingredients-on-hand).
2) Recommend exactly one dish from the catalog by selecting a dish_id from the provided list.
3) Give a brief explanation (1–3 sentences) tied to the user’s preferences.
</scope_positive>

<decision_policy>
- Use DECISION: ASK when the request is in-scope but you need more information to choose a dish.
- Use DECISION: RECOMMEND when you have enough information to pick one dish confidently.
- Use DECISION: REFUSE only when the request is out-of-scope or unsafe, not when you merely need more information.
- Breakfast/lunch/dinner/snack requests are all in-scope. Treat them as meal context, not scope restrictions.
</decision_policy>

<out_of_scope_handling_positive>
When a request is outside your services, respond helpfully by doing one of these allowed actions:
A) For medical or clinical nutrition requests: explain you can only help pick from the catalog using general preferences, and invite the user to share non-medical preferences.
B) For restaurant search, ordering, delivery, prices, or real-time availability: explain you can only recommend a dish idea from the catalog, and invite the user to share preferences.
C) For step-by-step recipes or cooking instructions: explain you can recommend a dish from the catalog and give a short reason, but not detailed instructions.
D) For exact macro/calorie calculation: explain you can recommend based on general goals (lighter/filling, higher protein) but do not compute exact numbers.
</out_of_scope_handling_positive>

<cuisine_handling>
If the user names a cuisine that is not explicitly represented in the catalog (for example Japanese or Chinese), do not refuse.
Instead, acknowledge the preference briefly and recommend the closest option available in the catalog based on tags (spicy/mild, rice/pasta/soup/salad, filling/light, protein preference).
</cuisine_handling>

<uncertainty_escape_hatch>
If you do not have enough information to choose a dish, ask 1–2 short questions.
If you still cannot choose after a total of 3 questions in the conversation, do NOT refuse.
Instead, make your best-effort recommendation from the catalog and explain any assumptions briefly.
</uncertainty_escape_hatch>

<format>
Always respond in this plain-text mini protocol and nothing else.

If you need more info:
DECISION: ASK
Q1: <question>
Q2: <question>   (optional; maximum 2 questions)

If you can recommend:
DECISION: RECOMMEND
DISH: <one dish_id from the catalog>
WHY: <1–3 sentences referencing the user's preferences>

If the request is outside your services:
DECISION: REFUSE
WHY: <1–2 sentences>
ALT: <what the user can provide so you can recommend from the catalog>
</format>

<hard_rules>
- Ask at most 2 questions per response.
- Recommend exactly one dish when you recommend.
- Stop asking once you have enough information and recommend.
- Ask only what you need. As soon as you can pick a dish confidently, recommend immediately.
- Only use dish_ids from the provided catalog.
- When a cuisine is unavailable, choose the closest match instead.
- Never use REFUSE just because information is missing.
</hard_rules>

<normalization>
If the user says:
- "no ingredients", "no ingredients on hand", "got no food" -> treat as ingredients_on_hand = none
- "no budget", "budget is fine", "any budget" -> treat as budget_level = no_limit
- Only ask follow-up questions about those items if they are not already provided.
</normalization>
"""

FEW_SHOT_EXAMPLES = [
    {
        "user": "What should I eat for dinner?",
        "assistant": (
            "DECISION: ASK\n"
            "Q1: Any dietary restrictions or allergies?\n"
            "Q2: How much time do you have, and do you want something light or filling?"
        ),
    },
    {
        "user": "Suggest me dinner menu.",
        "assistant": (
            "DECISION: ASK\n"
            "Q1: Any dietary restrictions or allergies?\n"
            "Q2: Do you want something spicy or mild?"
        ),
    },
    {
        "user": "I have no allergies, I want something meat-based, and I have a lot of time.",
        "assistant": (
            "DECISION: ASK\n"
            "Q1: What kind of cuisine do you prefer (Korean, Italian, Mexican, or no preference)?\n"
            "Q2: Do you want rice, pasta, salad, or soup as the base?"
        ),
    },
    {
        "user": "I have 15 minutes, want high protein, no seafood, and I feel like something spicy. I prefer rice.",
        "assistant": (
            "DECISION: RECOMMEND\n"
            "DISH: spicy_chicken_rice_bowl\n"
            "WHY: You want a quick, spicy, high-protein dinner with rice and no seafood, so this matches your preferences and time limit."
        ),
    },
    {
        "user": "No allergies. Any cuisine. No ingredients on hand. No budget. I want something meat-based.",
        "assistant": (
            "DECISION: RECOMMEND\n"
            "DISH: beef_bibimbap\n"
            "WHY: You want a meat-based dinner with no ingredient constraints and no budget limit, so beef bibimbap is a filling, satisfying option from the catalog."
        ),
    },
    {
        "user": "What should I eat for dinner if I have diabetes? Give me a strict meal plan and exact macros.",
        "assistant": (
            "DECISION: REFUSE\n"
            "WHY: I cannot provide medical or condition-specific nutrition advice or strict macro targets.\n"
            "ALT: If you share general preferences like cuisine, time, and ingredients to avoid, I can recommend one dish from the catalog."
        ),
    },
    {
        "user": "No allergies, and I want Chinese.",
        "assistant": (
            "DECISION: RECOMMEND\n"
            "DISH: spicy_chicken_rice_bowl\n"
            "WHY: We do not have Chinese cuisine explicitly in the catalog, but you want something flavorful and filling, and this spicy chicken rice bowl is the closest match available."
        ),
    },
]


def build_initial_messages() -> list[dict]:
    menu_text = build_menu_catalog_text()
    system_with_catalog = (
        SYSTEM_PROMPT
        + "\n\n<catalog>\nAllowed dish_ids:\n"
        + menu_text
        + "\n</catalog>\n"
    )

    messages = [{"role": "system", "content": system_with_catalog}]
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    return messages


def build_menu_catalog_text() -> str:
    lines = []
    for dish_id, meta in MENU.items():
        tags = ", ".join(meta.get("tags", []))
        lines.append(f"- {dish_id} (tags: {tags})")
    return "\n".join(lines)

def pretty_dish_name(dish_id: str) -> str:
    return dish_id.replace("_", " ").title()



def generate_response(messages: list[dict]) -> str:
    """Generate a response using LiteLLM."""
    try:
        response = completion(model=MODEL, messages=messages, temperature=0.2)
        return response.choices[0].message.content
    except Exception as e:
        return (
            "DECISION: REFUSE\n"
            f"WHY: Model call failed: {str(e)}\n"
            "ALT: Please share your dinner preferences (diet/allergies, time, and what you feel like), and I will recommend one dish from the catalog."
        )

# --- Backstop to catch misses ---

DECISION_RE = re.compile(r"^DECISION:\s*(ASK|RECOMMEND|REFUSE)\s*$", re.MULTILINE)
DISH_RE = re.compile(r"^DISH:\s*([a-z0-9_]+)\s*$", re.MULTILINE)
Q_RE = re.compile(r"^Q[1-9]:\s*(.+)$", re.MULTILINE)
ALT_RE = re.compile(r"^ALT:\s*(.+)$", re.MULTILINE)

def apply_backstop(text: str) -> tuple[bool, str, str]:
    """Validate/repair model output.

    Returns (ok, final_text, error_reason). If not ok, `final_text` is a safe
    ASK response to keep the conversation going.
    """
    m = DECISION_RE.search(text.strip())
    if not m:
        return (
            False,
            "DECISION: ASK\n"
            "Q1: Any dietary restrictions or allergies?\n"
            "Q2: How much time do you have, and do you want something light or filling?",
            "Missing DECISION header."
        )

    decision = m.group(1)

    if decision == "ASK":
        qs = [q.strip() for q in Q_RE.findall(text)]
        if not qs:
            return (
                False,
                "DECISION: ASK\n"
                "Q1: Any dietary restrictions or allergies?\n"
                "Q2: How much time do you have, and do you want something light or filling?",
                "ASK missing Q lines."
            )
        if len(qs) > 2:
            return (
                False,
                "DECISION: ASK\n"
                f"Q1: {qs[0]}\n"
                f"Q2: {qs[1]}",
                "Too many questions."
            )
        return (True, text, "")

    if decision == "RECOMMEND":
        dm = DISH_RE.search(text)
        if not dm:
            return (
                False,
                "DECISION: ASK\n"
                "Q1: Do you want something spicy or mild?\n"
                "Q2: Do you prefer rice, pasta, salad, or soup?",
                "RECOMMEND missing DISH."
            )
        dish_id = dm.group(1)
        if dish_id not in MENU:
            return (
                False,
                "DECISION: ASK\n"
                "Q1: Do you prefer chicken, beef, turkey, tofu, or vegetarian?\n"
                "Q2: Do you want something spicy or mild?",
                "Unknown dish_id."
            )
        return (True, text, "")

    if decision == "REFUSE":
        # Optional: require ALT line for deterministic evaluation
        if not ALT_RE.search(text):
            # Repair by adding ALT
            repaired = text.strip() + "\nALT: If you share non-medical preferences (time, cuisine, allergies), I can recommend one dish from the catalog."
            return (True, repaired, "")
        return (True, text, "")

    return (False, text, "Unknown decision.")

OOS_MEDICAL = {
    "diabetes", "pregnant", "pregnancy", "kidney", "cholesterol",
    "hypertension", "blood pressure", "cancer", "pcos",
}
OOS_ORDERING = {"near me", "delivery", "doordash", "ubereats", "order", "restaurant", "open now"}
OOS_RECIPE = {"recipe", "how to cook", "instructions", "step by step", "cook it", "make it"}
OOS_MACROS = {"macros", "calories", "kcal", "protein grams", "carbs grams", "fat grams"}

SAFETY_DISTRESS = {"suicide", "kill myself", "self harm", "hurt myself", "want to die", "can't go on"}
EATING_DISORDER = {"anorexia", "bulimia", "purge", "binge", "starve", "laxative"}

def detect_oos_or_safety(text: str) -> str | None:
    t = (text or "").lower()

    if any(k in t for k in SAFETY_DISTRESS) or any(k in t for k in EATING_DISORDER):
        return "safety"
    if any(k in t for k in OOS_MEDICAL):
        return "medical"
    if any(k in t for k in OOS_ORDERING):
        return "ordering"
    if any(k in t for k in OOS_RECIPE):
        return "recipe"
    if any(k in t for k in OOS_MACROS):
        return "macros"
    return None

def oos_response(flag: str) -> str:
    if flag == "safety":
        return (
            "DECISION: REFUSE\n"
            "WHY: I cannot help with requests involving self-harm or eating-disorder behaviors.\n"
            "ALT: If you are in immediate danger, call your local emergency number. If you want, tell me what you're feeling and I can help you find support resources. For dinner, share general preferences (time, allergies, light vs filling) and I can recommend one dish from the catalog."
        )
    if flag == "medical":
        return (
            "DECISION: REFUSE\n"
            "WHY: I cannot provide medical or condition-specific nutrition advice.\n"
            "ALT: Share non-medical preferences (allergies, time, spicy vs mild, rice/pasta/soup/salad) and I will recommend one dish from the catalog."
        )
    if flag == "ordering":
        return (
            "DECISION: REFUSE\n"
            "WHY: I cannot search restaurants, prices, or delivery availability.\n"
            "ALT: Tell me your preferences (time, allergies, spicy vs mild, filling vs light) and I will recommend one dish idea from the catalog."
        )
    if flag == "recipe":
        return (
            "DECISION: REFUSE\n"
            "WHY: I cannot provide step-by-step cooking instructions.\n"
            "ALT: Tell me your preferences and I will recommend one dish from the catalog with a short reason."
        )
    if flag == "macros":
        return (
            "DECISION: REFUSE\n"
            "WHY: I cannot compute exact calories or macros.\n"
            "ALT: Tell me your general goal (lighter vs filling, higher protein) plus time and allergies, and I will recommend one dish from the catalog."
        )
    return (
        "DECISION: REFUSE\n"
        "WHY: I cannot help with that request.\n"
        "ALT: Share your dinner preferences and I will recommend one dish from the catalog."
    )


# --- Session Management ---

# Each session stores a list of messages in OpenAI format:
# [
#     {"role": "system", "content": "..."},
#     {"role": "user", "content": "Hello!"},
#     {"role": "assistant", "content": "Hi there!"},
#     ...
# ]
sessions: dict[str, list[dict]] = {}


# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.get("/")
def index():
    return FileResponse("index.html")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = build_initial_messages()

    # Add user message
    sessions[session_id].append({"role": "user", "content": request.message})

    # -------- Python out-of-scope + safety backstop --------
    flag = detect_oos_or_safety(request.message)
    if flag is not None:
        final_text = oos_response(flag)
        _, final_text, _ = apply_backstop(final_text)

        sessions[session_id].append({"role": "assistant", "content": final_text})
        return ChatResponse(response=final_text, session_id=session_id)

    # -------- Normal LLM path --------
    raw_text = generate_response(sessions[session_id])
    _, final_text, _ = apply_backstop(raw_text)

    # -------- Only prettify dish name (nothing else) --------
    def replace_dish(match):
        dish_id = match.group(1)
        return f"DISH: {pretty_dish_name(dish_id)}"

    final_text = re.sub(
        r"DISH:\s*([a-z0-9_]+)",
        replace_dish,
        final_text
    )

    sessions[session_id].append({"role": "assistant", "content": final_text})
    return ChatResponse(response=final_text, session_id=session_id)


@app.post("/clear")
def clear(session_id: str | None = None):
    if session_id and session_id in sessions:
        del sessions[session_id]
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
