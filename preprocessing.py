import re
from nltk.stem import PorterStemmer

# ek baar stemmer banao, baar baar mat banao - performance issue hota hai
stemmer = PorterStemmer()

def load_stopwords(path=r"C:\Users\Ibad Ur Rehman\Desktop\Sem 6\IR\VSM_IR\corpus\Stopword-List (1).txt"):
    stopwords = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    except FileNotFoundError:
        print(f"Error: Stopword file not found at {path}")
    return stopwords
# globally load kar lo taake har call pe file na kholni pare
STOPWORDS = load_stopwords()

def preprocess(text):
    # lowercase - TRUMP aur trump same hone chahiye
    text = text.lower()

    # sirf alphabets rakho, numbers aur punctuation hata do
    tokens = re.findall(r"[a-z']+", text)

    # stopwords hata do - 'the', 'is', 'of' wagera koi kaam ke nahi
    tokens = [t for t in tokens if t not in STOPWORDS]

    # stemming - "running", "runs", "ran" sab "run" ban jayenge
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens