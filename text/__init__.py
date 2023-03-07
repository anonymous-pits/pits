""" from https://github.com/keithito/tacotron """
import re
from unicodedata import normalize

from text.cleaners import collapse_whitespace
from text.symbols import lang_to_dict, lang_to_dict_inverse


def text_to_sequence(raw_text, lang):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        text: string to convert to a sequence
        lang: language of the input text
    Returns:
        List of integers corresponding to the symbols in the text
    '''

    _symbol_to_id = lang_to_dict(lang)
    text = collapse_whitespace(raw_text)

    if lang == 'ko_KR':    
        text = normalize('NFKD', text)
        sequence = [_symbol_to_id[symbol] for symbol in text]
        tone = [0 for i in sequence]

    elif lang == 'en_US':
        _curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
        sequence = []

        while len(text):
            m = _curly_re.match(text)

            if m is not None:
                ar = m.group(1)
                sequence += [_symbol_to_id[symbol] for symbol in ar]
                ar = m.group(2)
                sequence += [_symbol_to_id[symbol] for symbol in ar.split()]
                text = m.group(3)
            else:
                sequence += [_symbol_to_id[symbol] for symbol in text]
                break

        tone = [0 for i in sequence]

    else:
        raise RuntimeError('Wrong type of lang')

    assert len(sequence) == len(tone)
    return sequence, tone


def sequence_to_text(sequence, lang):
    '''Converts a sequence of IDs back to a string'''
    _id_to_symbol = lang_to_dict_inverse(lang)
    result = ''
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text
