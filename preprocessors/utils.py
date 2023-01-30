import numpy as np 
import math
import re

def load_dictionary(path):
    lexicon_dict ={}
    with open(path,'r') as g:
        for line in g:
            parts = line.strip().split(' ',1)
            pinyin, phoneme = parts[0], parts[1].split()
            lexicon_dict[pinyin] = phoneme
    return lexicon_dict

def process_rr(line):
    line = line.replace('6','2')
    line = line.replace('15','5')
    line = line.replace('25','5')
    line = line.replace('35','5')
    line = line.replace('45','5')
    if line!='rr' and line[-1] not in ['1','2','3','4','5']:
        line = line.upper()
    if len(line)>=3:
        if 'er' != line[:-1] and 'r'==line[-2]:
            return [line[:-2]+line[-1],'rr']
        else:
            return [line]
    else:
        return [line]

def del_blank(string):
    while '  ' in string:
        string = string.replace('  ',' ')
    return string

chinese_dict = load_dictionary('preprocessors/lang/lexicon.txt')

def get_alignment_by_mono(_file, prosody, pinyin, sampling_rate, hop_length, return_full=False):
    prosody = prosody.replace(' ','')

    sil_phones = ["sil", "sp", "spn", '',"sp1"]
    
    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    lines = open(_file,'r').readlines()
    num = int(len(lines))
    last_time = 0
    for i in range(num):
        line = del_blank(lines[i]).split()
        s, e, p = float(line[0].strip())/1e+7, float(line[1].strip())/1e+7, line[2].strip()
        # Trimming leading silences
        if p=='er':
            p='rr'
        if phones == [] and p in sil_phones:
            start_time = e
            continue
        else:
            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append('$')
            durations.append(int(math.floor(e*sampling_rate/hop_length))-int(math.floor(s*sampling_rate/hop_length)))
        last_time = int(math.floor(e*sampling_rate/hop_length))

    # Trimming tailing silences
    phones = phones[:end_idx]
    if return_full:
        durations = durations[:end_idx]
        durations.insert(0,int(math.floor(start_time*sampling_rate/hop_length)))
        durations.append(last_time-int(math.floor(end_time*sampling_rate/hop_length)))
    else:
        durations = durations[:end_idx]

    rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
    pros = rule.sub('',prosody.strip().lower())
    pros = pros.replace('#','')
    pros = list(pros)
    pinyin = re.split(' |/',pinyin.strip())
    py = []
    for x in pinyin:
        py+=process_rr(x)
    pinyin = py
    attr_pros = [0 if x in ['1','2','3','4'] else 1 for x in pros]
    assert len(pinyin)==sum(attr_pros), print(pinyin,pros)
    rang = list(range(len(attr_pros)))
    for i in rang:
        if attr_pros[i]==0:
            pinyin.insert(i,pros[i]) 
    assert len(pinyin)==len(pros)
    pros_phones=[]

    #mean to process punctuatioin
    for x in pinyin:
        if x in ['1','2','3','4']:
            pros_phones+=[x]
        else:
            if len(x)==1 and x<='Z' and x>='A':
                tmp = chinese_dict[x]
                tmp = [y+'+zimu'for y in tmp]
                pros_phones+=tmp
            else:
                pros_phones+=chinese_dict[x]
            pros_phones+=['&']

    short_pros_index = [i+2 for i in range(len(pros_phones)) if pros_phones[i] not in ['1','2','3','4','&'] ]
   
    phones_index = [0 if x=='$' else 1 for x in phones]

    assert len(short_pros_index)==sum(phones_index)
    #print(phones)
    sp_id = [i for i in range(len(phones_index)) if phones_index[i]==0]
    rang = list(range(len(sp_id)))
    rang.reverse()
    for i in rang:
        idx = sp_id[i]-i-1
        if pros_phones[short_pros_index[idx]] in ['1','2','3','4']:
            pros_phones.insert(short_pros_index[idx]+1, '$')
        else:
            pros_phones.insert(short_pros_index[idx], '$')
    
    mask = [0 if x in ['1','2','3','4','&'] else 1 for x in pros_phones]
    assert len(phones)==sum(mask)

    _pros_phones = [ pros_phones[i] for i in range(len(mask)) if mask[i]==1 ]
    _pros_phones = [x.replace('+zimu','') for x in _pros_phones]
    assert _pros_phones == phones, print(_pros_phones,phones)

    return phones, durations, start_time, end_time, pros_phones, mask


language_mapping = {'Chinese':'zh', 'Uygur': 'ug','English':'en','Spanish':'es','German':'de','Franch':'fr','Japanese':'ja','Korean':'ko'}
def get_alignment_word_boundary(tier, word_tier, sampling_rate, hop_length, language, return_full=False):
    sil_phones = ['sil', 'sp', 'spn', '']

    phones = []
    pros_phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    last_time = 0
    word_boundary_time = [w.end_time for w in word_tier._objects]
    word_text = [w.text for w in word_tier._objects]
    assert '<unk>' not in word_text
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == [] and p in sil_phones:
            start_time = e
            continue
        else:
            if p not in sil_phones:
                phones.append(p)
                pros_phones.append(p)
                if e in word_boundary_time:
                    pros_phones.append('^')
                end_time = e
                end_idx = len(phones)
                end_idx_pros = len(pros_phones)
            else:
                phones.append('$')
                pros_phones.append('$')
            durations.append(int(e*sampling_rate/hop_length)-int(s*sampling_rate/hop_length))
        last_time = int(math.floor(e*sampling_rate/hop_length))

    # Trimming tailing silences
    phones = phones[:end_idx]
    pros_phones = pros_phones[:end_idx_pros]
    if return_full:
        durations = durations[:end_idx]
        durations.insert(0,int(math.floor(start_time*sampling_rate/hop_length)))
        durations.append(last_time-int(math.floor(end_time*sampling_rate/hop_length)))
    else:
        durations = durations[:end_idx]
    
    mask = [0 if x in ['1','2','3','4','5','^'] else 1 for x in pros_phones]
    assert len(phones)==sum(mask)

    #add language identifier
    phones = [x+'_'+language_mapping[language] if x!='$' else x  for x in phones]
    pros_phones = [x+'_'+language_mapping[language] if x!='$' else x for x in pros_phones]
    return phones, durations, start_time, end_time, pros_phones, mask


def is_outlier(x, p25, p75):
    """Check if value is an outlier."""
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    return x <= lower or x >= upper


def remove_outlier(x, p_bottom: int = 25, p_top: int = 75):
    """Remove outlier from x."""
    p_bottom = np.percentile(x, p_bottom)
    p_top = np.percentile(x, p_top)

    indices_of_outliers = []
    for ind, value in enumerate(x):
        if is_outlier(value, p_bottom, p_top):
            indices_of_outliers.append(ind)

    x[indices_of_outliers] = 0.0
    x[indices_of_outliers] = np.max(x)
    return x
    
def average_by_duration(x, durs):
    length = sum(durs)
    durs_cum = np.cumsum(np.pad(durs, (1, 0), mode='constant'))

    # calculate charactor f0/energy
    if len(x.shape) == 2:
        x_char = np.zeros((durs.shape[0], x.shape[1]), dtype=np.float32)
    else:
        x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(length), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values, axis=0) if len(values) > 0 else 0.0  # np.mean([]) = nan.

    return x_char.astype(np.float32)


 

