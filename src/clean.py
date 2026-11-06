"""
clean.py  -  Phase 1, Task 4
Urdu news text ke liye EK reusable cleaning pipeline: clean_text().
Har dafa same input par same output deta hai (koi randomness nahi).

Steps ka ORDER (aur kyun):
  1. Unicode NFC           -> sab se pehle, taake ek hi harf ki do encodings ek ho jayen
  2. Urdu normalization    -> Arabic harf ko Urdu harf mein badalna, aerab hatana,
                              jure hue words ("ہےفنانس") ko alag karna
  3. Noise removal         -> HTML, URL, English, punctuation, extra spaces
  4. Case folding          -> Urdu mein capital/small nahi hota (sirf safety ke liye)
  5. Numbers -> NUM        -> har alag number alag "word" na bane
  6. Tokenization          -> whitespace par (Urdu mein words space se alag hain)
  7. Stop-word removal     -> negations (نہیں، نہ، مت) list se BAHAR rakhe hain
  8. Stemming              -> halka suffix stemmer (Urdu ka koi lemmatizer nahi)
"""
import re
import unicodedata

# ---------- 2. Urdu-specific normalization ----------
# Arabic keyboard / Arabic text se aane wale harf jo dekhne mein same hain
# lekin Unicode mein alag hain. Inko ek na kiya to "کتاب" aur "كتاب"
# do alag words ban jayenge.
URDU_CHAR_MAP = {
    "ي": "ی",  # ي Arabic yeh       -> ی Urdu yeh (farsi yeh)
    "ى": "ی",  # ى alef maksura     -> ی
    "ك": "ک",  # ك Arabic kaf       -> ک Urdu kaf (keheh)
    "ه": "ہ",  # ه Arabic heh       -> ہ Urdu heh goal
    "ۀ": "ۂ",  # ۀ                  -> ۂ
    "ة": "ۃ",  # ة teh marbuta      -> ۃ
    chr(0x200C): " ",    # zero-width non-joiner -> space
    chr(0x200D): "",     # zero-width joiner     -> hata do
    chr(0x00A0): " ",    # non-breaking space    -> normal space
}
# Urdu (۰-۹) aur Arabic-Indic (٠-٩) digits -> 0-9
for i in range(10):
    URDU_CHAR_MAP[chr(0x06F0 + i)] = str(i)
    URDU_CHAR_MAP[chr(0x0660 + i)] = str(i)
URDU_TRANS = str.maketrans(URDU_CHAR_MAP)

DIACRITICS = re.compile(r"[ً-ٰٟـ]")   # aerab + tatweel

# Urdu mein "ے" (bari yeh) aur "ں" (noon ghunna) SIRF lafz ke aakhir mein aate hain.
# Agar in ke foran baad koi Urdu harf ho, to do words jur gaye hain
# (dataset se punctuation hatne ki wajah se): "ہےفنانس" -> "ہے فنانس", "ہیںیہ" -> "ہیں یہ"
GLUED = re.compile(r"([ےں])(?=[ء-غف-يٹ-ۓ])")

# ---------- 3. Noise ----------
HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\S+@\S+\.\S+")
LATIN = re.compile(r"[A-Za-z]+")
# Urdu letters (0600-06FF), digits, aur whitespace ke ilawa sab (punctuation
# jaise ۔ ، ؟ ! " وغیرہ) ko space bana do
NON_URDU = re.compile(r"[^؀-ۿ0-9\s]")
URDU_PUNCT = re.compile(r"[،؛؟۔٪-٭]")  # ، ؛ ؟ ۔ ٪
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
SPACES = re.compile(r"\s+")

# ---------- 7. Stop words ----------
# NLTK / spaCy mein Urdu stop-word list nahi hai, is liye khud banayi.
# Sirf grammar wale words (huroof, auxiliary verbs, pronouns).
# NEGATIONS (نہیں، نہ، مت، بغیر) jaan bujh kar list mein NAHI daale,
# kyun ke "اضافہ نہیں ہوا" aur "اضافہ ہوا" ka matlab ulta hai.
URDU_STOPWORDS = set("""
کے کی کا کو میں سے نے پر اور ہے ہیں تھا تھی تھے ہو ہوں ہوا ہوئی ہوئے ہوگا ہوگی ہوں گے
یہ وہ اس ان ایک بھی تو کہ جو جس جن جنہوں انہوں انہیں اسے اسی انہی
کر کرنے کرتے کرتا کرتی کیا کیے کئے کی گئی گیا گئے گا گی گے
رہا رہی رہے لیے لئے تک کچھ اپنے اپنی اپنا ہی جب تب اب یا لیکن مگر
کوئی کسی ساتھ والے والی والا ہونے دیا دی دیے دیں جا جاتا جاتی جاتے جائے
سکتا سکتی سکتے چاہیے کہا طرف ذریعے دوران وغیرہ ابھی جبکہ کہ بلکہ پھر
ہم آپ میں نہایت بہت ایسے ایسی ایسا جیسے جیسی جیسا کیونکہ تاکہ اگر
کیلئے کیلیے کےلیے ء
""".split())
NEGATIONS = {"نہیں", "نہ", "مت", "بغیر"}
URDU_STOPWORDS -= NEGATIONS

# ---------- 8. Light stemmer ----------
# Sirf plural / oblique suffixes hataye. Lambe suffix pehle check hote hain.
# (suffix, replacement): "شہریوں" -> "شہری", "کمپنیاں" -> "کمپنی", "خاندانوں" -> "خاندان"
# Stem kam az kam 3 harf ka rehna chahiye, warna chhote words toot jate.
SUFFIX_RULES = [("یوں", "ی"), ("یاں", "ی"), ("وں", ""), ("یں", ""), ("ات", "")]
MIN_STEM = 3


def light_stem(word):
    for suf, rep in SUFFIX_RULES:
        if word.endswith(suf) and len(word) - len(suf) >= MIN_STEM:
            return word[: -len(suf)] + rep
    return word


# ===================================================================
def normalize_urdu(text):
    """Step 1 + 2: NFC + Urdu harf normalization + aerab hatana."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(URDU_TRANS)
    text = DIACRITICS.sub("", text)
    text = GLUED.sub(r"\1 ", text)      # jure hue words alag karo
    return text


def remove_noise(text):
    """Step 3: HTML, URL, email, English, punctuation, extra spaces."""
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    text = EMAIL.sub(" ", text)
    text = LATIN.sub(" ", text)
    text = URDU_PUNCT.sub(" ", text)
    text = NON_URDU.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def tokenize(text):
    """Step 6: Urdu mein words space se alag hote hain."""
    return text.split()


def clean_text(text, remove_stopwords=True, stem=True, return_tokens=False):
    """
    POORI pipeline. Phase 2 aur 3 bhi yahi function call karenge.
    Input : raw Urdu text (str)
    Output: cleaned text (space se jure tokens), ya list agar return_tokens=True
    """
    if not isinstance(text, str):
        return [] if return_tokens else ""
    text = normalize_urdu(text)          # 1, 2
    text = remove_noise(text)            # 3
    text = text.lower()                  # 4 (Urdu mein case nahi hota; no-op safety step)
    text = NUMBER.sub(" NUM ", text)     # 5 (har number -> ek hi token "NUM")
    tokens = tokenize(text)              # 6
    if remove_stopwords:                 # 7
        tokens = [t for t in tokens if t not in URDU_STOPWORDS]
    if stem:                             # 8
        tokens = [light_stem(t) for t in tokens]
    return tokens if return_tokens else " ".join(tokens)


if __name__ == "__main__":
    raw = ("اسلام آباد: عالمی بینک نے 20 لاکھ ڈالر فراہم کرنے کا اعلان کیا، "
           "جس سے متاثرہ خاندانوں کی مدد ہوگی۔ برآمدات میں ۲۳ فیصد اضافہ نہیں ہوا۔ "
           "<p>Dawn</p> https://www.dawnnews.tv/news/1/ كتاب يہ")
    print("RAW  :", raw)
    print("CLEAN:", clean_text(raw))
    for w in ["خاندانوں", "شہریوں", "منصوبوں", "کمپنیاں", "معاملات", "خواتین"]:
        print(f"  stem: {w} -> {light_stem(w)}")
