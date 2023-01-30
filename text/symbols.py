""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
_pad        = '_'
_punctuation = '!\'(),.:;? -¡¿'
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

from text import Mandarin
_Mandarin = ['@' + s for s in Mandarin.valid_symbols]

#language_mapping = {'Uygur': 'ug','English':'en','Spanish':'es','German':'de','Franch':'fr','Japanese':'ja','Korean':'ko'}
from text import English
_English = ['@' + s for s in English.valid_symbols]

from text import Uyghur
_Uyghur = ['@' + s + '_ug' for s in Uyghur.valid_symbols]

from text import Spanish
_Spanish = ['@' + s + '_es' for s in Spanish.valid_symbols]

from text import German
_German = ['@' + s + '_de' for s in German.valid_symbols]


symbols = list(_punctuation)+list(_pad)+list(_special)+list(_letters)+list(_letters_ipa)
symbols = symbols+_Mandarin+_English+_Uyghur+_Spanish+_German

def is_language(x):
    shared_len = len(list(_punctuation)+list(_pad)+list(_special)+list(_letters)+list(_letters_ipa))
    if x<shared_len:
        return -1
    elif x<shared_len+len(_Mandarin):
        return 0
    elif x<shared_len+len(_Mandarin)+len(_English):
        return 1
    elif x<shared_len+len(_Mandarin)+len(_English)+len(_Uyghur):
        return 2
    elif x<shared_len+len(_Mandarin)+len(_English)+len(_Uyghur)+len(_Spanish):
        return 3
    elif x<shared_len+len(_Mandarin)+len(_English)+len(_Uyghur)+len(_Spanish)+len(_German):
        return 4
    else:
        raise RuntimeError('Symbols is out of range!')
