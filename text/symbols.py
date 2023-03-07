_pad = '_'
_punc = ";:,.!?¡¿—-…«»'“”~() "

_jamo_leads = "".join([chr(_) for _ in range(0x1100, 0x1113)])
_jamo_vowels = "".join([chr(_) for _ in range(0x1161, 0x1176)])
_jamo_tails = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])
_kor_characters = _jamo_leads + _jamo_vowels + _jamo_tails

_cmu_characters = [
    'AA', 'AE', 'AH',
    'AO', 'AW', 'AY',
    'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY',
    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH'
]


lang_to_symbols = {
    'common': [_pad] + list(_punc),
    'ko_KR': list(_kor_characters), 
    'en_US': _cmu_characters, 
}

def lang_to_dict(lang):
    symbol_lang = lang_to_symbols['common'] + lang_to_symbols[lang]
    dict_lang = {s: i for i, s in enumerate(symbol_lang)}
    return dict_lang

def lang_to_dict_inverse(lang):
    symbol_lang = lang_to_symbols['common'] + lang_to_symbols[lang]
    dict_lang = {i: s for i, s in enumerate(symbol_lang)}
    return dict_lang

def symbol_len(lang):
    symbol_lang = lang_to_symbols['common'] + lang_to_symbols[lang]
    return len(symbol_lang)
