"""
regex_patterns.py  -  Phase 1, Task 3
Hamare Urdu news data ke liye useful regular expressions.
Har pattern ke upar likha hai ke woh kya dhoondta hai aur kyun.
"""
import re

# 1) NUMBERS (ASCII + Urdu/Arabic-Indic digits)
#    \d Python 3 mein Unicode digits bhi pakarta hai (۰-۹ aur ٠-٩ bhi).
#    "\d+(?:[.,]\d+)*" -> 20, 1,500, 7.67 sab ek token.
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

# 2) NUMBER + URDU UNIT  (paise / miqdaar wali news ke liye)
#    e.g. "20 لاکھ", "16 کروڑ", "3 ارب", "23 فیصد"
#    Business category ke articles mein ye bohat aate hain.
NUMBER_WITH_UNIT = re.compile(r"(\d+(?:[.,]\d+)*)\s*(ہزار|لاکھ|کروڑ|ارب|کھرب|فیصد|ڈالر|روپے)")

# 3) MERGED WORDS (dataset authors ne punctuation hata diya, to jumla khatam
#    hone wala "گا/گی/گے" agle jumle ke pehle lafz "اس/ان/یہ/وہ" se jur gaya)
#    e.g. "کرے گااس" (asal mein "کرے گا۔ اس")
#    \b ... \b se sirf poore words match hote hain, beech ke nahi.
MERGED_FUTURE = re.compile(r"\b(گا|گی|گے)(اس|ان|یہ|وہ|جس|جبکہ)\b")

# 4) URL  ->  article ID nikalna (URL column ke liye)
#    e.g. https://www.dawnnews.tv/news/1148499/  ->  1148499
URL = re.compile(r"https?://\S+|www\.\S+")
URL_ARTICLE_ID = re.compile(r"/news/(\d+)")

# 5) DATE  (YYYY-MM-DD, Date column ka format)
DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# 6) LATIN (English) letters / HTML tags - raw input mein aa sakte hain
LATIN_WORD = re.compile(r"[A-Za-z]+")
HTML_TAG = re.compile(r"<[^>]+>")

# 7) Urdu aerab (diacritics: zabar, zer, pesh, tashdeed, ...) aur tatweel
DIACRITICS = re.compile(r"[ً-ٰٟـ]")


def demo():
    text = ("اسلام باد عالمی بینک 20 لاکھ ڈالر فراہم کرے گااس منصوبے "
            "سے 23 فیصد اضافہ ہوگا <b>Dawn</b> https://www.dawnnews.tv/news/1148499/")
    print("NUMBER          :", NUMBER.findall(text))
    print("NUMBER_WITH_UNIT:", NUMBER_WITH_UNIT.findall(text))
    print("MERGED_FUTURE   :", MERGED_FUTURE.findall(text))
    print("URL             :", URL.findall(text))
    print("URL_ARTICLE_ID  :", URL_ARTICLE_ID.findall(text))
    print("HTML_TAG        :", HTML_TAG.findall(text))
    print("LATIN_WORD      :", LATIN_WORD.findall(HTML_TAG.sub(" ", URL.sub(" ", text))))
    print("DATE            :", DATE.findall("2020-12-06"))


if __name__ == "__main__":
    demo()
