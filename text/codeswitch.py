import random
import pkuseg
from g2pc import G2pC
from g2p_en import G2p
from opencc import OpenCC

from translate import Translator
import re
from preprocessors.utils import language_mapping
import string

def read_lexicon(lex_path, language):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if language in language_mapping.keys():
                phones = [x+"_"+language_mapping[language] for x in phones]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

chinese_lexicon = read_lexicon('preprocessors/lang/lexicon.txt', 'Mandarin')
english_lexicon = read_lexicon('preprocessors/lang/librispeech-lexicon.txt', 'English')
spanish_lexicon = read_lexicon('preprocessors/lang/spanish-lexicon.txt', 'Spanish')
from text import symbols

def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def whole_chinese(string):
    for ch in string:
        if not u'\u4e00' <= ch <= u'\u9fff':
            return False
    return True

class TranTools(object):
    def __init__(self) -> None:
        super().__init__()
        self.tools = {}
        self.tools['en-ch'] = Translator(from_lang='en',to_lang="zh")
        self.tools['ch-en'] = Translator(from_lang='zh',to_lang="en")
    
    def trans(self, text, src_lang, tgt_lang):
        tools_name = src_lang+'-'+tgt_lang
        result = self.tools[tools_name].translate(text)
        result = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", result)
        return result

class G2PTools(object):
    def __init__(self) -> None:
        super().__init__()
        self.tools = {}
        self.ch_g2p = G2pC()
        self.en_g2p = G2p()
    
    def g2p(self, text, src_lang):
        if src_lang=='zh':
            phoneme = []
            #print(self.ch_g2p(text))
            pinyin = [x[3].split() for x in self.ch_g2p(text)]
            for _py in pinyin:
                for x in _py:
                    x = x.replace('u:','v') #For pinyin that uses the ü, represent it with a u followed by a colon (e.g. nu:3 ren2)
                    if x=='r5': # 儿化音
                        x='rr'
                    if x in chinese_lexicon:
                        phoneme += chinese_lexicon[x]
                        #phoneme += ['&']           
                    else:
                        phoneme += [x]
            phoneme = [x if '@'+x in symbols else x for x in phoneme ]
            return phoneme
        elif src_lang=='en':
            phoneme = self.en_g2p(text)
            phoneme = [x+'_en' if '@'+x+'_en' in symbols else x for x in phoneme ]
            #phoneme += [' ']
            #assert phoneme[0] != ' ', print(text, phoneme)
            #phoneme = ['^' if x==' ' else x for x in phoneme]
            return phoneme
        elif src_lang=="es":
            phoneme = []
            vocabs = text.split()
            for vocab in vocabs:
                if vocab in spanish_lexicon:
                    phoneme += spanish_lexicon[vocab]
                else:
                    return False
            return phoneme

class SpanishEnglishG2P():
    def __init__(self) -> None:
        super().__init__()
        self.g2p_tools = G2PTools()

    def g2p(self, x):
        x = x.strip('"#$%&\'()*+-./:<=>?@[\\]^_`{|}~…-')
        raw = re.split(r"([,;.\-\?\!\s+])", x)

        phns = []
        for xx in raw:
            if xx==' ' or xx=='':
                continue
            ph = self.g2p_tools.g2p(xx, 'es')
            if not ph:
                ph = list(filter(lambda p: p != " ", self.g2p_tools.g2p(xx,'en')))
            phns += ph

        phones = "{" + "}{".join(phns) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "} {", phones)
        phones = phones.replace("}{", " ")
        phones = phones.replace("{{", "{")
        phones = phones.replace("}}", "}")
        phones = phones.replace(" {} ", " ")

        return phones

class CodeSwitchGen(object):
    def __init__(self, cross_rate, sentence_rate=1.0, src_language=None, tgt_language=None) -> None:
        super().__init__()
        self.cross_rate = cross_rate
        self.sentence_rate = sentence_rate
        self.seg = pkuseg.pkuseg()           # 以默认配置加载模型
        self.g2p_tools = G2PTools()
        self.t2s = OpenCC('t2s')  # 繁转简
        self.worddict, self.worddict_vocab = {}, {}
        if src_language is not None and tgt_language is not None:
            src2tgt = language_mapping[src_language]+'-'+language_mapping[tgt_language]
            self.worddict[src2tgt], self.worddict_vocab[src2tgt]= getattr(self, "load_{}".format(src2tgt.replace('-','_')))('text/Panlex/{}.txt'.format(src2tgt))
            #print(self.worddict)
        else:
            self.worddict['en-zh'], self.worddict_vocab['en-zh']=self.load_en_ch('text/Panlex/en-zh.txt')
            self.worddict['zh-en'], self.worddict_vocab['zh-en']=self.load_ch_en('text/Panlex/zh-en.txt')
            self.worddict['en-es'], self.worddict_vocab['en-es']=self.load_en_ch('text/Panlex/en-es.txt')

    def load_en_zh(self, path):
        mapping = {}
        mapping_vocab = {}
        en_ch = open(path,'r').readlines()
        for line in en_ch:
            parts = line.strip().split()
            key = parts[0].lower()
            value =  self.t2s.convert(parts[1])
            if not whole_chinese(value):
                continue
            try:
                if key not in mapping:
                    mapping[key]=[]
                    mapping_vocab[key]=[]
                phoneme = self.g2p_tools.g2p(value, 'zh')
                mapping[key].append(phoneme)
                mapping_vocab[key].append(value)
            except:
                print(value)
        return mapping, mapping_vocab
    
    def load_zh_en(self, path):
        mapping = {}
        mapping_vocab = {}
        ch_en = open(path, 'r').readlines()
        for line in ch_en:
            parts = line.strip().split()
            key = self.t2s.convert(parts[0])
            value = parts[1].upper()
            if value not in english_lexicon or not whole_chinese(key):
                continue
            if key not in mapping:
                mapping[key]=[]
                mapping_vocab[key]=[]
            phoneme = english_lexicon[value]
            mapping[key].append(phoneme)
            mapping_vocab[key].append(value)
        return mapping, mapping_vocab

    def load_en_es(self, path):
        mapping = {}
        mapping_vocab = {}
        for line in open(path,'r').readlines():
            parts = line.strip().split()
            key = parts[0].lower()
            value =  parts[1].lower()
            try:
                phoneme = self.g2p_tools.g2p(value, 'es')
                if phoneme:
                    if key not in mapping:
                        mapping[key]=[]
                        mapping_vocab[key]=[]
                    mapping[key].append(phoneme)
                    mapping_vocab[key].append(value)
            except:
                print(value)
        return mapping, mapping_vocab

    def cross(self, x, worddict_name, src_lang, disable=False):
        if not disable  and (self.cross_rate >= random.random()):
            if x in self.worddict[worddict_name]:
                idx = random.randint(0,len(self.worddict[worddict_name][x]) - 1)
                return self.worddict[worddict_name][x][idx], self.worddict_vocab[worddict_name][x][idx]
            else:
                return self.g2p_tools.g2p(x, src_lang), x
        else:
            return self.g2p_tools.g2p(x, src_lang), x

    def cross_str(self, x, src_lang='en', tgt_lang='zh', disable=False):
        x = x.strip('"#$%&\'()*+-./:<=>?@[\\]^_`{|}。！~…-')
        worddict_name = src_lang+'-'+tgt_lang
        if src_lang=='zh':
            raw = self.seg.cut(x)
        else:
            raw = re.split(r"([,;.\-\?\!\s+])", x)

        phns = []
        out = ""
        for xx in raw:
            if xx==' ' or xx=='':
                continue
            ph, vocab = self.cross(xx, worddict_name, src_lang, disable)
            phns += ph
            out += vocab
            out += " "

        phones = "{" + "}{".join(phns) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "} {", phones)
        phones = phones.replace("}{", " ")
        phones = phones.replace("{{", "{")
        phones = phones.replace("}}", "}")
        return out, phones

    def cross_list(self, x, src_lang, tgt_lang):
        return self.cross_str(x, src_lang, tgt_lang, not ( self.sentence_rate >= random.random()))
