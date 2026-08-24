#!/usr/bin/env python3
"""
اتوماسیون تولید و انتشار روزانه مقاله سئو برای پارت‌زون (partzune.ir)

هرروز یک بار اجرا می‌شود (از طریق GitHub Actions):
  1. یک موضوع جدید (که قبلا منتشر نشده) از بین دسته‌های محصولات سایت انتخاب می‌کند
  2. یک مقاله کامل و سئو-محور برای آن موضوع با Google Gemini API (رایگان) تولید می‌کند
  3. مقاله را به‌صورت خودکار در بخش بلاگ سایت (از طریق API میکسین) منتشر می‌کند
  4. موضوع منتشر شده را در state/used_topics.json ذخیره می‌کند تا تکراری نشود
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# تنظیمات
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MIXIN_API_KEY = os.environ.get("MIXIN_API_KEY")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")
SITE_BASE_URL = os.environ.get("SITE_BASE_URL", "https://partzune.ir").rstrip("/")

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "state", "used_topics.json")

# مدل رایگان گوگل جمینای - نیازی به کارت اعتباری نداره.
# اگه گوگل اسم مدل رو عوض کرد، از aistudio.google.com مدل جدید رایگان رو چک کن و اینجا جایگزین کن.
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# عکس‌های استوک واقعی و کاملا رایگان (بدون کارت بانکی، بدون ریسک شارژ اتفاقی)
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

# دسته‌های محصولاتی که سایت روشون کار می‌کنه - AI هرروز از همینا یه موضوع مشخص و خاص انتخاب می‌کنه
PRODUCT_CATEGORIES = [
    "لنت ترمز",
    "قالپاق ماشین",
    "قالپاق اسپرت",
    "سرکمک / سردنده",
    "سردنده اسپرت",
    "سراگزوز",
    "سراگزوز اسپرت",
    "سراگزوز دولول",
    "شمع ماشین",
    "اسپری پنچرگیری",
    "سایر لوازم یدکی و اکسسوری اسپرت خودرو",
]

# اولین مقاله رو طبق خواسته صریح صاحب سایت هاردکد می‌کنیم؛ از روز دوم به بعد
# انتخاب موضوع کاملا خودکار و توسط AI انجام میشه.
FIRST_TOPIC = {
    "category": "لنت ترمز",
    "topic": "بررسی و راهنمای خرید لنت ترمز چیان (Chian)",
    "primary_keyword": "لنت ترمز چیان",
}


# ---------------------------------------------------------------------------
# ابزارهای کمکی
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def fail(msg: str) -> None:
    log(f"❌ خطا: {msg}")
    sys.exit(1)


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {"published": []}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def extract_json(text: str) -> dict:
    """پاسخ کلود رو حتی اگه توی ```json فنس پیچیده شده باشه، پارس می‌کنه."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def call_gemini(system: str, user: str, max_tokens: int = 4000) -> str:
    if not GEMINI_API_KEY:
        fail("GEMINI_API_KEY تنظیم نشده")

    resp = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers={"content-type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.8,
            },
        },
        timeout=120,
    )
    if resp.status_code != 200:
        fail(f"خطای Gemini API ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    try:
        candidate = data["candidates"][0]
        parts = candidate["content"]["parts"]
        return "\n".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError):
        fail(f"پاسخ غیرمنتظره از Gemini: {json.dumps(data, ensure_ascii=False)[:500]}")


def get_pexels_image(query: str):
    """یه عکس استوک واقعی و رایگان از Pexels برای کوئری داده‌شده پیدا می‌کنه."""
    if not PEXELS_API_KEY:
        log("⚠️ PEXELS_API_KEY تنظیم نشده - این عکس رد میشه")
        return None

    try:
        resp = requests.get(
            PEXELS_SEARCH_URL,
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 3, "orientation": "landscape"},
            timeout=30,
        )
        if resp.status_code != 200:
            log(f"⚠️ خطای Pexels ({resp.status_code}) برای کوئری «{query}» - رد میشه")
            return None

        photos = resp.json().get("photos", [])
        if not photos:
            log(f"⚠️ برای کوئری «{query}» عکسی پیدا نشد - رد میشه")
            return None

        p = photos[0]
        return {
            "url": p["src"]["large"],
            "photographer": p.get("photographer", "Pexels"),
            "photographer_url": p.get("photographer_url", "https://www.pexels.com"),
            "pexels_url": p.get("url", "https://www.pexels.com"),
        }
    except requests.RequestException as e:
        log(f"⚠️ خطای شبکه توی Pexels: {e} - این عکس رد میشه")
        return None


def build_image_html(photo: dict, alt_text: str) -> str:
    return (
        '<figure style="margin:24px 0;text-align:center;">'
        f'<img src="{photo["url"]}" alt="{alt_text}" '
        'style="max-width:100%;height:auto;border-radius:10px;" loading="lazy" />'
        '<figcaption style="font-size:12px;color:#888;margin-top:6px;">'
        f'عکس: <a href="{photo["photographer_url"]}" target="_blank" rel="noopener">{photo["photographer"]}</a>'
        f' / <a href="{photo["pexels_url"]}" target="_blank" rel="noopener">Pexels</a>'
        "</figcaption></figure>"
    )


# ---------------------------------------------------------------------------
# مرحله ۱: انتخاب موضوع
# ---------------------------------------------------------------------------

def pick_topic(state: dict) -> dict:
    if not state["published"]:
        log("این اولین اجرای اتوماسیونه -> موضوع اول هاردکد شده: لنت ترمز چیان")
        return FIRST_TOPIC

    used_list = "\n".join(
        f"- {item['topic']} (دسته: {item['category']})" for item in state["published"]
    )

    system = (
        "تو یک استراتژیست سئوی حرفه‌ای برای یک فروشگاه اینترنتی ایرانی لوازم یدکی و "
        "اکسسوری اسپرت خودرو به نام «پارت‌زون» (partzune.ir) هستی. کارت انتخاب دقیق‌ترین "
        "و مشخص‌ترین موضوع ممکن برای مقاله بعدی وبلاگ است، نه یک موضوع کلی."
    )
    user = f"""دسته‌های محصولات سایت:
{chr(10).join(f"- {c}" for c in PRODUCT_CATEGORIES)}

موضوعاتی که قبلا منتشر شدن (این‌ها رو دیگه تکرار نکن، حتی مشابهش رو هم انتخاب نکن):
{used_list if used_list else "(هنوز هیچی منتشر نشده)"}

یک موضوع جدید، خیلی مشخص و کاربردی (نه کلی) برای مقاله بعدی وبلاگ انتخاب کن. ترجیحا موضوع رو
به یک برند خاص، یک مدل خاص خودرو ایرانی (مثل پراید، پژو ۲۰۶، پژو پارس، سمند، تیبا، دنا، رانا،
شاهین، کوییک و امثالش)، یا یک مسئله رایج مشتری (مثل «چطور تشخیص بدیم لنت ترمز باید تعویض بشه»)
گره بزن تا برای سئوی کم‌رقابت مناسب باشه.

فقط و فقط این JSON رو خروجی بده، بدون هیچ توضیح اضافه و بدون فنس مارک‌داون:
{{
  "category": "یکی از دسته‌های بالا",
  "topic": "عنوان کامل و مشخص موضوع مقاله",
  "primary_keyword": "کلیدواژه اصلی سئو برای این مقاله"
}}"""

    raw = call_gemini(system, user, max_tokens=500)
    try:
        topic = extract_json(raw)
    except Exception as e:
        fail(f"پارس نشدن موضوع انتخابی: {e}\nپاسخ خام: {raw[:300]}")

    for key in ("category", "topic", "primary_keyword"):
        if key not in topic:
            fail(f"فیلد {key} توی موضوع انتخابی نیست: {topic}")

    log(f"موضوع امروز: {topic['topic']}  (دسته: {topic['category']})")
    return topic


# ---------------------------------------------------------------------------
# مرحله ۲: تولید مقاله
# ---------------------------------------------------------------------------

def generate_article(topic: dict) -> dict:
    system = (
        "تو یک کپی‌رایتر و متخصص سئوی حرفه‌ای فارسی‌زبان هستی که برای فروشگاه اینترنتی "
        "لوازم یدکی و اکسسوری اسپرت خودرو «پارت‌زون» مقاله وبلاگ می‌نویسی. لحن مقالات باید "
        "قابل‌اعتماد، دقیق فنی ولی قابل‌فهم برای مشتری عادی باشه."
    )
    user = f"""یک مقاله وبلاگ کامل و سئو-محور به زبان فارسی درباره موضوع زیر بنویس:

موضوع: {topic['topic']}
دسته محصول مرتبط: {topic['category']}
کلیدواژه اصلی: {topic['primary_keyword']}

قوانین سئو که باید رعایت بشه:
- کلیدواژه اصلی باید توی تیتر H1، پاراگراف اول، حداقل یکی از H2 ها، و seo_description بیاد
- طول مقاله بین ۹۰۰ تا ۱۴۰۰ کلمه
- ساختار: مقدمه کوتاه جذاب -> چند بخش با تیترهای H2/H3 -> یک بخش «سوالات متداول» با حداقل ۳ سوال
  (با تگ‌های h3 برای سوال) -> نتیجه‌گیری کوتاه با یک call-to-action ملایم برای خرید از پارت‌زون
- زبان طبیعی و روان فارسی، از تکرار بیش از حد کلیدواژه (کیورد استافینگ) خودداری کن
- seo_title حداکثر ۶۰ کاراکتر، seo_description حداکثر ۱۵۵ کاراکتر
- slug باید فقط انگلیسی، حروف کوچک، با خط تیره بین کلمات باشه (مثلا: chian-brake-pad-guide)
- ۴ تا ۶ تگ مرتبط برای دسته‌بندی محتوا پیشنهاد بده

قوانین فرمت‌بندی و ظاهر مقاله (خیلی مهمه، مقاله باید حرفه‌ای و آبرومند به نظر برسه، نه یه متن خشک):
- بدنه باید HTML معتبر باشه، نه Markdown
- نکات کلیدی، اسم برندها، و هشدارهای مهم رو حتما با <strong> بولد کن (نه بیش از حد، فقط جاهای مهم)
- حتما یک جدول (<table>) مفید بساز -- مثلا جدول مقایسه مشخصات، جدول علائم خرابی و راه‌حل،
  یا جدول قیمت/مدل. جدول باید این استایل رو داشته باشه دقیقا:
  <table style="width:100%;border-collapse:collapse;margin:20px 0;">
  <thead><tr style="background:#f5f5f5;">
  <th style="border:1px solid #ddd;padding:10px;text-align:right;">ستون۱</th>...
  </tr></thead>
  <tbody><tr><td style="border:1px solid #ddd;padding:10px;">مقدار</td>...</tr></tbody></table>
- از لیست‌های <ul><li> برای نکات چندتایی استفاده کن
- تیترها: H2 برای بخش‌های اصلی، H3 برای زیربخش‌ها و سوالات متداول (اندازه فونت با همین تگ‌ها مشخص میشه)

قوانین عکس (خیلی مهمه):
- دقیقا ۳ جای مشخص توی مقاله باید عکس بذاری: یکی زود بعد از مقدمه، یکی وسط مقاله نزدیک
  مهم‌ترین بخش فنی، و یکی نزدیک انتها قبل از نتیجه‌گیری
- توی متن body دقیقا این پلیس‌هولدرها رو بذار: {{{{IMAGE_1}}}} و {{{{IMAGE_2}}}} و {{{{IMAGE_3}}}}
  (این‌ها بعدا با عکس واقعی جایگزین میشن، خودت تگ img ننویس)
- برای هر پلیس‌هولدر، توی فیلد جداگانه "images" یک کوئری جستجوی عکس به زبان انگلیسی (برای یه بانک
  عکس استوک بین‌المللی) و یک alt text فارسی مرتبط با اون بخش از مقاله بده. کوئری باید ساده،
  واقع‌گرایانه و قابل پیدا شدن توی بانک عکس استوک باشه (مثلا: "car brake pad mechanic repair"،
  "car wheel rim closeup"، "spark plug engine")، نه اسم برند ایرانی چون تو بانک عکس خارجی پیدا نمیشه

فقط و فقط این JSON رو خروجی بده، بدون هیچ توضیح اضافه، بدون فنس مارک‌داون، بدون متن قبل یا بعدش:
{{
  "title": "تیتر اصلی مقاله (H1)",
  "slug": "english-url-slug",
  "seo_title": "عنوان سئو حداکثر ۶۰ کاراکتر",
  "seo_description": "توضیحات متا حداکثر ۱۵۵ کاراکتر",
  "description": "خلاصه کوتاه یک یا دو خطی از مقاله برای نمایش در لیست بلاگ",
  "tags": ["تگ۱", "تگ۲", "تگ۳"],
  "images": [
    {{"placeholder": "IMAGE_1", "query": "english stock photo search query", "alt": "متن جایگزین فارسی"}},
    {{"placeholder": "IMAGE_2", "query": "english stock photo search query", "alt": "متن جایگزین فارسی"}},
    {{"placeholder": "IMAGE_3", "query": "english stock photo search query", "alt": "متن جایگزین فارسی"}}
  ],
  "body": "<p>...</p>{{{{IMAGE_1}}}}<h2>...</h2>...<table>...</table>...{{{{IMAGE_2}}}}...{{{{IMAGE_3}}}}..."
}}"""

    raw = call_gemini(system, user, max_tokens=7000)
    try:
        article = extract_json(raw)
    except Exception as e:
        fail(f"پارس نشدن مقاله تولیدشده: {e}\nپاسخ خام (۵۰۰ کاراکتر اول): {raw[:500]}")

    required = ["title", "slug", "seo_title", "seo_description", "description", "tags", "body", "images"]
    for key in required:
        if key not in article or (key != "body" and not article[key]):
            fail(f"فیلد {key} توی مقاله تولیدشده نیست یا خالیه")

    log(f"مقاله تولید شد: {article['title']}  (slug: {article['slug']})")

    # جایگزینی پلیس‌هولدرهای عکس با عکس واقعی از Pexels
    body = article["body"]
    for img_spec in article.get("images", []):
        placeholder = "{{" + img_spec.get("placeholder", "") + "}}"
        if placeholder not in body:
            continue
        photo = get_pexels_image(img_spec.get("query", topic["primary_keyword"]))
        if photo:
            html = build_image_html(photo, img_spec.get("alt", article["title"]))
            body = body.replace(placeholder, html)
            log(f"🖼️  عکس برای «{img_spec.get('query')}» جاسازی شد")
        else:
            body = body.replace(placeholder, "")  # اگه عکس پیدا نشد، جای خالی حذف میشه

    article["body"] = body
    return article


# ---------------------------------------------------------------------------
# مرحله ۳: انتشار در میکسین
# ---------------------------------------------------------------------------

def publish_to_mixin(topic: dict, article: dict) -> dict:
    if not MIXIN_API_KEY:
        fail("MIXIN_API_KEY تنظیم نشده")

    payload = {
        "title": article["title"],
        "slug": article["slug"],
        "body": article["body"],
        "description": article["description"],
        "seo_title": article["seo_title"],
        "seo_description": article["seo_description"],
        "tags": article["tags"],
        "post_type": "blog",
        "category_name": topic["category"],
        "is_active": True,
        "top_menu_show": False,
        "footer_show": False,
        "accept_reviews": True,
    }

    resp = requests.post(
        f"{SITE_BASE_URL}/api/v4/pages/",
        headers={
            "Authorization": f"Api-Key {MIXIN_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if resp.status_code not in (200, 201):
        fail(f"خطای انتشار در میکسین ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    log(f"✅ مقاله با موفقیت منتشر شد. آدرس/شناسه: {data.get('data', {}).get('url') or data.get('data', {}).get('id')}")
    return data


# ---------------------------------------------------------------------------
# اجرای اصلی
# ---------------------------------------------------------------------------

def main() -> None:
    log("شروع اتوماسیون تولید و انتشار مقاله روزانه پارت‌زون")

    state = load_state()
    topic = pick_topic(state)
    article = generate_article(topic)
    result = publish_to_mixin(topic, article)

    state["published"].append(
        {
            "topic": topic["topic"],
            "category": topic["category"],
            "primary_keyword": topic["primary_keyword"],
            "slug": article["slug"],
            "title": article["title"],
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
    )
    save_state(state)
    log("✅ state/used_topics.json آپدیت شد")
    log("پایان اتوماسیون")


if __name__ == "__main__":
    main()
