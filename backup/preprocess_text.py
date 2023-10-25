import re
from textacy import preprocessing
from defang import refang

# preprocessing rules

# Whitespace
_RE_WS = re.compile(r'\s+')

# Words containing non-alphanumeric characters
UNCOMMON_CHARS = r'[^\x00-\x7F\x80-\xFF]'
_RE_UNCOMMON = re.compile(UNCOMMON_CHARS)

_RE_BTC_ADDR = re.compile(r'([13]|bc1)[A-HJ-NP-Za-km-z1-9]{27,34}')
_RE_ETH_ADDR = re.compile(r'0x[a-fA-F0-9]{40}')
_RE_LTC_ADDR = re.compile(r'(ltc1|[LM])[a-zA-HJ-NP-Z0-9]{26,40}')

_RE_LONGWORD = re.compile(r'(\b|\B)\S{38,}(\b|\B)')

# IP address regex
IPV4SEG  = r'(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
IPV4ADDR = r'(?:(?:' + IPV4SEG + r'\.){3,3}' + IPV4SEG + r')'
IPV6SEG  = r'(?:(?:[0-9a-fA-F]){1,4})'
IPV6GROUPS = (
    r'(?:' + IPV6SEG + r':){7,7}' + IPV6SEG,                  # 1:2:3:4:5:6:7:8
    r'(?:' + IPV6SEG + r':){1,7}:',                           # 1::                                 1:2:3:4:5:6:7::
    r'(?:' + IPV6SEG + r':){1,6}:' + IPV6SEG,                 # 1::8               1:2:3:4:5:6::8   1:2:3:4:5:6::8
    r'(?:' + IPV6SEG + r':){1,5}(?::' + IPV6SEG + r'){1,2}',  # 1::7:8             1:2:3:4:5::7:8   1:2:3:4:5::8
    r'(?:' + IPV6SEG + r':){1,4}(?::' + IPV6SEG + r'){1,3}',  # 1::6:7:8           1:2:3:4::6:7:8   1:2:3:4::8
    r'(?:' + IPV6SEG + r':){1,3}(?::' + IPV6SEG + r'){1,4}',  # 1::5:6:7:8         1:2:3::5:6:7:8   1:2:3::8
    r'(?:' + IPV6SEG + r':){1,2}(?::' + IPV6SEG + r'){1,5}',  # 1::4:5:6:7:8       1:2::4:5:6:7:8   1:2::8
    IPV6SEG + r':(?:(?::' + IPV6SEG + r'){1,6})',             # 1::3:4:5:6:7:8     1::3:4:5:6:7:8   1::8
    r':(?:(?::' + IPV6SEG + r'){1,7}|:)',                     # ::2:3:4:5:6:7:8    ::2:3:4:5:6:7:8  ::8       ::
    r'fe80:(?::' + IPV6SEG + r'){0,4}%[0-9a-zA-Z]{1,}',       # fe80::7:8%eth0     fe80::7:8%1  (link-local IPv6 addresses with zone index)
    r'::(?:ffff(?::0{1,4}){0,1}:){0,1}[^\s:]' + IPV4ADDR,     # ::255.255.255.255  ::ffff:255.255.255.255  ::ffff:0:255.255.255.255 (IPv4-mapped IPv6 addresses and IPv4-translated addresses)
    r'(?:' + IPV6SEG + r':){1,6}:?[^\s:]' + IPV4ADDR
)
_RE_IPV4 = re.compile(IPV4ADDR)
_RE_IPV6 = re.compile('|'.join(['(?:{})'.format(g) for g in IPV6GROUPS[::-1]]))

_RE_URL_ONION = re.compile(r'(?:https?://)?(?:www)?(\S*?\.onion)(\S*)?')

_RE_URL = re.compile(r"""
    (?xi)
    \b
    (							# Capture 1: entire matched URL
    (?:
        https?:				# URL protocol and colon
        (?:
        /{1,3}						# 1-3 slashes
        |								#   or
        [a-z0-9%]						# Single letter or digit or '%'
                                        # (Trying not to match e.g. "URI::Escape")
        )
        |							#   or
                                    # looks like domain name followed by a slash:
        [a-z0-9.\-]+[.]
        (?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj| Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)
        /
    )
    (?:							# One or more:
        [^\s()<>{}\[\]]+						# Run of non-space, non-()<>{}[]
        |								#   or
        \([^\s()]*?\([^\s()]+\)[^\s()]*?\)  # balanced parens, one level deep: (…(…)…)
        |
        \([^\s]+?\)							# balanced parens, non-recursive: (…)
    )+
    (?:							# End with:
        \([^\s()]*?\([^\s()]+\)[^\s()]*?\)  # balanced parens, one level deep: (…(…)…)
        |
        \([^\s]+?\)							# balanced parens, non-recursive: (…)
        |									#   or
        [^\s`!()\[\]{};:'".,<>?«»“”‘’]		# not a space or one of these punct chars
    )
    |					# OR, the following to match naked domains:
    (?:
        (?<!@)			# not preceded by a @, avoid matching foo@_gmail.com_
        [a-z0-9]+
        (?:[.\-][a-z0-9]+)*
        [.]
        (?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj| Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)
        \b
        /?
        (?!@)			# not succeeded by a @, avoid matching "foo.na" in "foo.na@example.com"
    )
    )
""", re.VERBOSE)

# hashtag
_RE_HASHTAG = re.compile(r'\B#([A-Za-z_][A-Za-z0-9_]*)')

def clean_hashtag(text):
    while 1: 
        m = re.search(_RE_HASHTAG, text)
        if m == None:
            break
        text = text[:m.start()] + text[m.start()+1:]
    
    return text

def remove_hashtag(text):
    return _RE_HASHTAG.sub('', text)    

    
# preprocess text
def preprocess_text(text):

    # remove non-alphanuemeric characters
    _text = re.sub(_RE_UNCOMMON, '', text)

    # refang text
    # _text = refang(_text)

    # clean hashtag: remove # if there is a hashtag
    # _text = clean_hashtag(_text)

    # replace emails and phone numbers
    _text = preprocessing.replace.emails(_text, repl="")
    _text = preprocessing.replace.phone_numbers(_text, repl="")

    # replace IP addresses
    # _text = _RE_IPV4.sub('ID_IP_ADDRESS', _text)
    # _text = _RE_IPV6.sub('ID_IP_ADDRESS', _text)

    # replace URLs
    _text = _RE_URL_ONION.sub('', _text)
    _text = _RE_URL.sub('', _text)

    # replace crypto addresses
    # _text = _RE_BTC_ADDR.sub('ID_BTC_ADDRESS', _text)
    # _text = _RE_ETH_ADDR.sub('ID_ETH_ADDRESS', _text)
    # _text = _RE_LTC_ADDR.sub('ID_LTC_ADDRESS', _text)

    # for each word in _text, if the word length is greater than 38, then replace it with ID_LONG_WORD
    # _text = _RE_LONGWORD.sub('ID_LONG_WORD', _text)

    # whitespace stripping
    # text_preprocessed = _RE_WS.sub(' ', _text).strip()


    return _text
