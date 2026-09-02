#!/usr/bin/env python3
"""
اتوماسیون تولید و انتشار روزانه مقاله سئو برای پارت‌زون (partzune.ir)

هرروز (یک یا دو بار، بسته به تنظیم GitHub Actions) اجرا می‌شود:
  1. یک موضوع جدید (که قبلا منتشر نشده) از بین دسته‌های محصولات سایت انتخاب می‌کند
  2. یک مقاله کامل، سئو و GEO-محور برای آن موضوع با Google Gemini API (رایگان) تولید می‌کند
  3. ۳ عکس واقعی از Pexels می‌گیرد، آن‌ها را روی خودِ فضای ذخیره‌سازی میکسین آپلود می‌کند
     (نه لینک مستقیم به Pexels، چون میکسین تگ عکس با دامنه خارجی رو حذف می‌کنه)
  4. مقاله را به‌صورت خودکار در بخش بلاگ سایت (از طریق API میکسین) منتشر می‌کند
  5. موضوع منتشر شده را در state/used_topics.json ذخیره می‌کند تا تکراری نشود
"""

import json
import mimetypes
import os
import re
import sys
import time
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
# اگه گوگل اسم مدل رو عوض کرد (مثل باری که gemini-2.5-flash از رده خارج شد)، از
# aistudio.google.com مدل جدید رایگان رو چک کن و اینجا جایگزین کن.
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# عکس‌های استوک واقعی و کاملا رایگان (بدون کارت بانکی، بدون ریسک شارژ اتفاقی)
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

# آپلود عکس روی فضای ذخیره‌سازی خود میکسین (تا سایت اونو حذف نکنه، چون تگ عکس با
# دامنه خارجی مثل Pexels رو حذف می‌کنه). طبق مستندات API میکسین -> POST /api/v4/media/upload/{directory}
MIXIN_MEDIA_UPLOAD_URL = f"{SITE_BASE_URL}/api/v4/media/upload/blog-images"

# دسته‌های محصولاتی که سایت روشون کار می‌کنه - AI هرروز از همینا یه موضوع مشخص و خاص انتخاب می‌کنه.
# این دسته‌ها فقط نمونه هستن؛ سایت هم قطعات یدکی خودرو می‌فروشه هم اکسسوری اسپرت.
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
    "سایر لوازم یدکی و اکسسوری اسپرت خودرو (فرمان، روکش صندلی، لاستیک برف‌پاک‌کن، لوازم تزئینی و مشابه)",
]

# اولین مقاله رو طبق خواسته صریح صاحب سایت هاردکد می‌کنیم؛ از اجرای دوم به بعد
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
    """پاسخ جمینای رو حتی اگه توی ```json فنس پیچیده شده باشه، پارس می‌کنه."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def call_gemini(
    system: str,
    user: str,
    max_tokens: int = 4000,
    use_search: bool = False,
    fail_on_error: bool = True,
):
    """
    برمی‌گردونه: (متن پاسخ یا None, status_code یا None اگه اصلا درخواست نرفت)
    اگه fail_on_error=True باشه، در صورت خطا مستقیم fail() می‌کنه (رفتار قبلی).
    اگه False باشه، خطا رو لاگ می‌کنه و (None, status_code) برمی‌گردونه تا caller خودش
    تصمیم بگیره (مثلا یه فال‌بک امتحان کنه).
    """
    if not GEMINI_API_KEY:
        fail("GEMINI_API_KEY تنظیم نشده")

    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.8,
            # مدل‌های نسل ۳ جمینای (مثل gemini-3.6-flash) یه بخش «تفکر داخلی» دارن که
            # جزئی از maxOutputTokens حساب میشه و کلا هم خاموش‌شدنی نیست؛ می‌ذاریمش رو
            # کمترین سطح تا بیشترین سهم توکن برای خودِ خروجی/JSON بمونه.
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }
    if use_search:
        # به مدل اجازه میده واقعا وب رو جستجو کنه (نه فقط از حافظه‌ش حدس بزنه) تا ببینه
        # چه موضوعاتی جای خالی دارن یا رقبا ضعیف پوشش دادن
        body["tools"] = [{"google_search": {}}]

    # ۴۲۹ (سهمیه لحظه‌ای تموم شده) و ۵۰۳ (سرور موقتا شلوغه) هر دو معمولا با کمی صبر
    # خودشون درست میشن؛ قبل از fail شدن چندبار با فاصله امتحان می‌کنیم
    max_retries = 4
    backoff_seconds = [5, 15, 30, 60]
    resp = None
    for attempt in range(max_retries):
        resp = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={"content-type": "application/json"},
            json=body,
            timeout=180,
        )
        if resp.status_code == 200:
            break
        if resp.status_code in (429, 503) and attempt < max_retries - 1:
            wait = backoff_seconds[attempt]
            log(f"⚠️ Gemini API {resp.status_code} - {wait} ثانیه صبر و تلاش دوباره ({attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        break

    if resp.status_code != 200:
        msg = f"خطای Gemini API ({resp.status_code}): {resp.text[:500]}"
        if fail_on_error:
            fail(msg)
        log(f"⚠️ {msg}")
        return None, resp.status_code

    data = resp.json()
    try:
        candidate = data["candidates"][0]
        parts = candidate["content"]["parts"]
        return "\n".join(p.get("text", "") for p in parts), 200
    except (KeyError, IndexError):
        msg = f"پاسخ غیرمنتظره از Gemini: {json.dumps(data, ensure_ascii=False)[:500]}"
        if fail_on_error:
            fail(msg)
        log(f"⚠️ {msg}")
        return None, 200


def call_gemini_simple(system: str, user: str, max_tokens: int = 4000, use_search: bool = False) -> str:
    """نسخه ساده که مثل قبل مستقیم متن رو برمی‌گردونه و در صورت خطا fail می‌کنه."""
    text, _ = call_gemini(system, user, max_tokens=max_tokens, use_search=use_search, fail_on_error=True)
    return text


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
        }
    except requests.RequestException as e:
        log(f"⚠️ خطای شبکه توی Pexels: {e} - این عکس رد میشه")
        return None


def upload_image_to_mixin(image_url: str, filename: str):
    """
    عکس رو از لینک Pexels دانلود می‌کنه و روی فضای ذخیره‌سازی خود میکسین
    (همون جایی که عکس محصولات هست) آپلود می‌کنه، تا لینک نهایی رو خود دامنه
    partzune.ir بده و توسط سایت حذف نشه.
    """
    if not MIXIN_API_KEY:
        log("⚠️ MIXIN_API_KEY تنظیم نشده - آپلود عکس رد میشه")
        return None

    try:
        img_resp = requests.get(image_url, timeout=30)
        if img_resp.status_code != 200:
            log(f"⚠️ دانلود عکس از Pexels شکست خورد ({img_resp.status_code})")
            return None
        content = img_resp.content
        content_type = img_resp.headers.get("content-type", "image/jpeg")
        ext = mimetypes.guess_extension(content_type) or ".jpg"
    except requests.RequestException as e:
        log(f"⚠️ خطای دانلود عکس از Pexels: {e}")
        return None

    try:
        resp = requests.post(
            MIXIN_MEDIA_UPLOAD_URL,
            headers={"Authorization": f"Api-Key {MIXIN_API_KEY}"},
            files={"file": (f"{filename}{ext}", content, content_type)},
            timeout=60,
        )
        if resp.status_code not in (200, 201):
            log(f"⚠️ آپلود عکس به میکسین شکست خورد ({resp.status_code}): {resp.text[:300]}")
            return None

        data = resp.json().get("data", {})
        full_path = data.get("full_path")
        if not full_path:
            log(f"⚠️ پاسخ آپلود میکسین فیلد full_path نداشت: {data}")
            return None

        # full_path چیزی مثل "tenant/media/blog-images/xxx.jpg" برمی‌گرده؛
        # اگه خودش مطلق نبود (با دامنه شروع نمی‌شد)، دامنه سایت رو جلوش اضافه کن
        if full_path.startswith("http"):
            return full_path
        return f"{SITE_BASE_URL}/{full_path.lstrip('/')}"
    except requests.RequestException as e:
        log(f"⚠️ خطای شبکه توی آپلود میکسین: {e}")
        return None


def build_image_html(url: str, alt_text: str) -> str:
    return (
        "<figure style='margin:24px 0;text-align:center;'>"
        f"<img src='{url}' alt='{alt_text}' "
        "style='max-width:100%;height:auto;border-radius:10px;' loading='lazy' />"
        "</figure>"
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
        "تو یک استراتژیست سئو و GEO (Generative Engine Optimization) حرفه‌ای برای یک "
        "فروشگاه اینترنتی ایرانی قطعات یدکی و اکسسوری اسپرت خودرو به نام «پارت‌زون» "
        "(partzune.ir) هستی. با تجربه واقعی جستجو و تحلیل رقبا موضوع بعدی وبلاگ رو انتخاب "
        "می‌کنی، نه با حدس زدن."
    )
    user = f"""دسته‌های محصولات سایت (شامل هم قطعات یدکی هم اکسسوری اسپرت خودرو):
{chr(10).join(f"- {c}" for c in PRODUCT_CATEGORIES)}

موضوعاتی که قبلا منتشر شدن (این‌ها رو دیگه تکرار نکن، حتی مشابهش رو هم انتخاب نکن):
{used_list if used_list else "(هنوز هیچی منتشر نشده)"}

از ابزار جستجوی وب که در اختیارت هست استفاده کن و واقعا این کارو انجام بده:
۱. چند تا کوئری جستجو بزن حول محور دسته‌های بالا (مثلا "راهنمای خرید لنت ترمز ایران"،
   "بهترین قالپاق اسپرت"، "علت خرابی سردنده" و مشابه) و ببین سایت‌های ایرانی چی نوشتن.
۲. دنبال این باش: (الف) سوال‌ها و موضوعاتی که هیچ سایت معتبری کامل جوابشون رو نداده،
   (ب) مقاله‌هایی که سایت‌های کم‌اعتبار (وبلاگ‌های شخصی، فروم‌های قدیمی، صفحات کم‌کیفیت
   و کم‌بازدید) نوشتن ولی می‌شه خیلی بهتر و کامل‌تر و به‌روزتر نوشتش.
۳. موضوع رو **عمومی و پرمخاطب** انتخاب کن، نه بیش‌ازحد محدود به یک مدل خاص خودرو. یعنی
   به‌جای «راهنمای تیغه برف‌پاک‌کن دنا پلاس و دنا»، بنویس «راهنمای کامل انتخاب و تعویض
   تیغه برف‌پاک‌کن ماشین» (که همه بخوانن، نه فقط مالکین دو مدل خاص). فقط وقتی موضوع رو به
   یک مدل خاص گره بزن که واقعا دلیل SEO قوی داشته باشه (مثلا حجم جستجوی بالای اون مدل خاص).
   در حالت عادی، موضوع باید کل بازدیدکننده‌های اون دسته محصول رو پوشش بده.

فقط و فقط این JSON رو خروجی بده، بدون هیچ توضیح اضافه و بدون فنس مارک‌داون:
{{
  "category": "یکی از دسته‌های بالا",
  "topic": "عنوان کامل، عمومی و پرمخاطب موضوع مقاله",
  "primary_keyword": "کلیدواژه اصلی سئو برای این مقاله",
  "gap_reason": "یک جمله کوتاه که توضیح بده این موضوع چرا جای خالی داره یا رقبا ضعیف پوششش دادن (بر اساس چیزی که واقعا سرچ کردی)"
}}"""

    raw, status = call_gemini(system, user, max_tokens=3000, use_search=True, fail_on_error=False)
    if raw is None:
        # سهمیه جستجوی زنده (google_search) تموم شده یا خطای دیگه‌ای بوده -> بدون
        # جستجوی زنده، فقط با دانش خود مدل امتحان کن تا اتوماسیون کامل نخوابه
        log(f"⚠️ جستجوی زنده وب شکست خورد (status={status})، بدون جستجو دوباره امتحان می‌کنیم")
        fallback_user = user.replace(
            "از ابزار جستجوی وب که در اختیارت هست استفاده کن و واقعا این کارو انجام بده:",
            "(الان دسترسی به جستجوی زنده وب نداری؛ بر اساس دانش عمومی‌ت دربارهٔ بازار قطعات "
            "یدکی و اکسسوری خودرو ایران و موضوعاتی که معمولا کم پوشش داده میشن حدس بزن):",
        )
        raw = call_gemini_simple(system, fallback_user, max_tokens=2000, use_search=False)

    try:
        topic = extract_json(raw)
    except Exception as e:
        fail(f"پارس نشدن موضوع انتخابی: {e}\nپاسخ خام: {raw[:300]}")

    for key in ("category", "topic", "primary_keyword"):
        if key not in topic:
            fail(f"فیلد {key} توی موضوع انتخابی نیست: {topic}")

    log(f"موضوع امروز: {topic['topic']}  (دسته: {topic['category']})")
    if topic.get("gap_reason"):
        log(f"دلیل انتخاب: {topic['gap_reason']}")
    return topic


# ---------------------------------------------------------------------------
# مرحله ۲: تولید مقاله
# ---------------------------------------------------------------------------

def generate_article(topic: dict) -> dict:
    system = (
        "تو یک متخصص سئو و GEO (Generative Engine Optimization) با ۱۲ سال سابقه واقعی "
        "کار حرفه‌ای هستی که الان برای فروشگاه اینترنتی قطعات یدکی و اکسسوری اسپرت خودرو "
        "«پارت‌زون» (partzune.ir) مقاله وبلاگ می‌نویسی. با استانداردهای آژانس‌های سئوی "
        "درجه‌یک می‌نویسی: عمیق، مستند، فنی-ولی-قابل‌فهم، و آنقدر معتبر که هم گوگل رتبه "
        "بالا بهش بده، هم اگه یه هوش مصنوعی مثل چت‌جی‌پی‌تی یا کلود بخواد به یه سوال مرتبط "
        "جواب بده و منبع پیدا کنه، همین مقاله رو به‌عنوان یه منبع معتبر و قابل‌استناد در "
        "نظر بگیره."
    )
    user = f"""یک مقاله وبلاگ کامل، سئو-محور و GEO-محور به زبان فارسی درباره موضوع زیر بنویس:

موضوع: {topic['topic']}
دسته محصول مرتبط: {topic['category']}
کلیدواژه اصلی: {topic['primary_keyword']}

قوانین سئوی کلاسیک (برای گوگل):
- کلیدواژه اصلی باید توی تیتر H1، پاراگراف اول، حداقل یکی از H2 ها، و seo_description بیاد
- طول مقاله بین ۹۰۰ تا ۱۴۰۰ کلمه
- زبان طبیعی و روان فارسی، از تکرار بیش از حد کلیدواژه (کیورد استافینگ) خودداری کن
- seo_title حداکثر ۶۰ کاراکتر، seo_description حداکثر ۱۵۵ کاراکتر
- slug باید فقط انگلیسی، حروف کوچک، با خط تیره بین کلمات باشه (مثلا: chian-brake-pad-guide)
- ۴ تا ۶ تگ مرتبط برای دسته‌بندی محتوا پیشنهاد بده

قوانین GEO (برای اینکه چت‌بات‌های هوش مصنوعی این مقاله رو به‌عنوان منبع/پیشنهاد استفاده کنن):
- بلافاصله بعد از مقدمه، یک پاراگراف کوتاه «پاسخ مستقیم» (۲-۳ جمله) بنویس که خلاصه و دقیق
  به سوال اصلی موضوع جواب بده -- طوری که اگه یه AI فقط همین پاراگراف رو بخونه، جواب کامل
  و قابل‌نقل‌قول رو داشته باشه. این پاراگراف باید واقعی و مفید باشه، نه تبلیغاتی و اغراق‌آمیز
- در طول مقاله، هرجا طبیعی بود، به «پارت‌زون» به‌عنوان مرجع/فروشنده این دسته از محصولات
  اشاره کن (مثلا «در پارت‌زون می‌تونید...» یا «تیم پارت‌زون توصیه می‌کنه...») ولی نه به‌شکل
  تبلیغاتی و غیرطبیعی -- فقط جاهایی که واقعا به محتوا کمک می‌کنه
- اطلاعات رو دقیق، مشخص و قابل‌استناد بنویس (اعداد، معیارهای مشخص، مراحل شماره‌گذاری‌شده)
  نه کلی‌گویی؛ AIها محتوای مبهم رو کمتر نقل قول می‌کنن
- بخش «سوالات متداول» رو با فرمت سوال-جواب مستقیم و خیلی واضح بنویس (نه پیچیده) چون این
  فرمت راحت‌تر توسط AIها استخراج و نقل قول میشه

ساختار مقاله:
مقدمه کوتاه جذاب -> پاراگراف «پاسخ مستقیم» -> چند بخش با تیترهای H2/H3 -> یک بخش «سوالات
متداول» با حداقل ۴ سوال (تگ h3 برای سوال) -> نتیجه‌گیری کوتاه با یک call-to-action ملایم

قوانین فرمت‌بندی و ظاهر مقاله (مقاله باید حرفه‌ای و آبرومند به نظر برسه، نه یه متن خشک):
- بدنه باید HTML معتبر باشه، نه Markdown
- خیلی مهم: چون این HTML قراره داخل یه رشتهٔ JSON قرار بگیره، توی همهٔ attribute های HTML
  (مثل style، class، href) حتما از کوتیشن تک ' استفاده کن، نه کوتیشن دابل " — یعنی
  style='...' درسته، style="..." باعث خراب شدن ساختار JSON میشه. این قانون رو در تمام
  طول body بدون استثنا رعایت کن.
- نکات کلیدی، اسم برندها، و هشدارهای مهم رو حتما با <strong> بولد کن (نه بیش از حد، فقط جاهای مهم)
- حتما یک جدول (<table>) مفید بساز -- مثلا جدول مقایسه مشخصات، جدول علائم خرابی و راه‌حل،
  یا جدول قیمت/مدل. جدول باید این استایل رو داشته باشه دقیقا (با کوتیشن تک):
  <table style='width:100%;border-collapse:collapse;margin:20px 0;'>
  <thead><tr style='background:#f5f5f5;'>
  <th style='border:1px solid #ddd;padding:10px;text-align:right;'>ستون۱</th>...
  </tr></thead>
  <tbody><tr><td style='border:1px solid #ddd;padding:10px;'>مقدار</td>...</tr></tbody></table>
- از لیست‌های <ul><li> برای نکات چندتایی استفاده کن
- تیترها: H2 برای بخش‌های اصلی، H3 برای زیربخش‌ها و سوالات متداول

قوانین عکس:
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

    raw = call_gemini_simple(system, user, max_tokens=12000)
    try:
        article = extract_json(raw)
    except Exception as e:
        log(f"⚠️ پارس اول JSON شکست خورد ({e}) - یه تلاش دوم می‌کنیم")
        retry_user = user + (
            "\n\nمهم: تلاش قبلی JSON نامعتبر تولید کرد (خطا: "
            + str(e)
            + "). این‌بار خیلی دقت کن که خروجی JSON کاملا معتبر باشه، مخصوصا اینکه "
            "همه attribute های HTML با کوتیشن تک ' نوشته بشن نه کوتیشن دابل، و هیچ "
            "کوتیشن دابل خام (\") داخل مقادیر رشته‌ای JSON نباشه."
        )
        raw = call_gemini_simple(system, retry_user, max_tokens=12000)
        try:
            article = extract_json(raw)
        except Exception as e2:
            fail(f"پارس نشدن مقاله تولیدشده (بعد از تلاش دوم): {e2}\nپاسخ خام (۵۰۰ کاراکتر اول): {raw[:500]}")

    required = ["title", "slug", "seo_title", "seo_description", "description", "tags", "body", "images"]
    for key in required:
        if key not in article or (key != "body" and not article[key]):
            fail(f"فیلد {key} توی مقاله تولیدشده نیست یا خالیه")

    log(f"مقاله تولید شد: {article['title']}  (slug: {article['slug']})")

    # جایگزینی پلیس‌هولدرهای عکس: عکس از Pexels گرفته میشه، بعد آپلود میشه روی خود
    # میکسین تا لینکش با دامنه partzune.ir باشه و توسط سایت حذف نشه
    body = article["body"]
    for i, img_spec in enumerate(article.get("images", []), start=1):
        placeholder = "{{" + img_spec.get("placeholder", "") + "}}"
        if placeholder not in body:
            continue

        photo = get_pexels_image(img_spec.get("query", topic["primary_keyword"]))
        if not photo:
            body = body.replace(placeholder, "")
            continue

        uploaded_url = upload_image_to_mixin(photo["url"], f"{article['slug']}-{i}")
        if uploaded_url:
            html = build_image_html(uploaded_url, img_spec.get("alt", article["title"]))
            body = body.replace(placeholder, html)
            log(f"🖼️  عکس «{img_spec.get('query')}» آپلود و جاسازی شد")
        else:
            log(f"⚠️ آپلود عکس «{img_spec.get('query')}» شکست خورد - این عکس حذف میشه")
            body = body.replace(placeholder, "")

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
