import unicodedata
import re
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)if unicodedata.category(c) != 'Mn')  #NFD=  decomposition of accented characters into char + accent. Mn= accents


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z.!?,0123456789]", r" ", s)  #remove nonstandard chars
    s = re.sub(r"([.!?,])", r" \1 ", s)  #seperate punctuation
    s = re.sub(r" +", r" ", s)  #remove redundant spaces
    s = s.strip()
    return s


