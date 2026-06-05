import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from flask import Flask, render_template, request, jsonify
from tlc_compress import get_compressor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from imethod.tllai import AIInputMethod
from models import Triplet
from utils.vietnamese import Vietnamese

app = Flask(__name__, template_folder="templates")

# Thread pool dùng chung cho tất cả requests (dịch các câu song song)
_EXECUTOR = ThreadPoolExecutor(max_workers=8)

GUIDANCE = (
    "TLLa là cách nhập tiếng Việt viết tắt bằng ký tự và số.\n"
    "- Mỗi từ kết thúc bằng số tông: 1/2/3/4/5/6/7/0.\n"
    "- Không cần dấu phụ, AI sẽ dịch thành tiếng Việt có dấu.\n"
    "- Ví dụ: x0chao2 -> xin chào.\n"
    "- Dùng dấu '.' để đánh dấu cuối vần khi cần phân biệt.\n"
    "- Nhập nhiều từ liền nhau như: minh2yeu5\n"
)

input_method = None


def get_input_method():
    global input_method
    if input_method is None:
        input_method = AIInputMethod(model='base', verbose=0)
    return input_method


@app.route('/')
def index():
    return render_template('index.html')


# ─── Helpers ───────────────────────────────────────────────────────────────────

def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize('NFD', text)
    stripped = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')
    return stripped.replace('đ', 'd').replace('Đ', 'D')


def normalize_compare(s: str) -> str:
    """Chuẩn hoá để so sánh: strip dấu, lowercase, collapse space."""
    return re.sub(r'\s+', ' ', strip_accents(s.lower())).strip()


TLLa_TOKEN = re.compile(r'[a-zA-Z]+(?:\.[a-zA-Z]+)*\d')


def is_tlla_token(word: str) -> bool:
    return bool(TLLa_TOKEN.fullmatch(word))


def parse_tlla_string(tlla_str: str) -> list:
    """
    Tách chuỗi TLLa liền nhau thành list các từ TLLa.
    Ví dụ: 'xin0chao2ban5' -> ['xin0', 'chao2', 'ban5']
    """
    return re.findall(r'[a-zA-Z]+(?:\.[a-zA-Z]+)*\d', tlla_str)


def add_spaces_to_tlla(tlla_block: str) -> str:
    """
    Thêm dấu cách vào chuỗi TLLa liền để AI có thể đọc.
    Ví dụ: 'x0ch2b5m2l2' -> 'x0 ch2 b5 m2 l2'
    """
    tokens = parse_tlla_string(tlla_block)
    return ' '.join(tokens)


# ─── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    """
    Tách text thành list các câu, giữ lại dấu câu cuối câu (. ! ?).
    Ví dụ: "Xin chào. Tạm biệt!" -> ["Xin chào.", "Tạm biệt!"]
    """
    # Tách theo dấu câu kết thúc, giữ lại dấu câu trong kết quả
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def join_sentences(sentences: list, is_tlla: bool = False) -> str:
    """
    Ghép các câu lại.
    - is_tlla=False (tiếng Việt): dùng dấu cách thông thường.
    - is_tlla=True (chuỗi TLLa): nếu câu trước kết thúc bằng dấu câu cuối câu
      (. ! ?) thì KHÔNG thêm space, để dấu câu gắn liền vào block TLLa tiếp.
      Ví dụ: ["x0ch2.", "m2l2 Tuấn"] -> "x0ch2.m2l2 Tuấn"
    """
    parts = [s for s in sentences if s]
    if not parts:
        return ""
    if not is_tlla:
        return ' '.join(parts)
    result = parts[0]
    for s in parts[1:]:
        if re.search(r'[.!?]$', result):
            result = result + s
        else:
            result = result + ' ' + s
    return result


# ─── Punctuation helpers ───────────────────────────────────────────────────────

def extract_punctuation_from_vietnamese(text: str) -> dict:
    """
    Trích xuất dấu câu từ văn bản tiếng Việt.

    Trả về:
      {
        'cleaned': str,          # văn bản đã bỏ dấu câu inline
        'punct_map': list,       # [{'position': int, 'punct': str, 'after': bool}]
        'trailing': str,         # dấu câu cuối câu (nếu có)
      }

    Dấu câu inline: , ; : ( ) " "
    Dấu câu cuối câu: . ! ?
    """
    # Tách từ (giữ whitespace để tái tạo sau)
    words = re.split(r'(\s+)', text)
    word_tokens = [w for w in words if not w.isspace() and w]

    punct_map = []
    cleaned_words = []

    for idx, word in enumerate(word_tokens):
        # Dấu câu inline (gắn vào cuối từ): , ; : ) "
        m_after = re.match(r'^(.*?)([,;:)\""]+)$', word)
        # Dấu câu cuối câu: . ! ?
        m_end = re.match(r'^(.*?)([.!?]+)$', word)

        if m_end and m_end.group(2) in ('.', '!', '?', '...'):
            core = m_end.group(1)
            punct = m_end.group(2)
            cleaned_words.append(core if core else word)
            if core:  # chỉ ghi nhận nếu có từ trước dấu
                punct_map.append({'position': idx, 'punct': punct, 'after': True})
        elif m_after and m_after.group(2):
            core = m_after.group(1)
            punct = m_after.group(2)
            cleaned_words.append(core if core else word)
            if core:
                punct_map.append({'position': idx, 'punct': punct, 'after': True})
        else:
            cleaned_words.append(word)

    cleaned = ' '.join(w for w in cleaned_words if w)
    # Tách trailing punctuation ở cuối câu
    trailing = ''
    m_trail = re.search(r'([.!?]+)\s*$', cleaned)
    if m_trail:
        trailing = m_trail.group(1)
        cleaned = cleaned[:m_trail.start()].strip()

    return {
        'cleaned': cleaned,
        'punct_map': punct_map,
        'trailing': trailing,
    }


def insert_punctuation_into_tlla_string(tlla_str: str, punct_map: list, trailing: str) -> str:
    """
    Chèn dấu câu vào chuỗi TLLa đã rút gọn.

    Ánh xạ position (index trong vi_words gốc) → token TLLa tương ứng,
    kể cả khi nhiều TLLa token đã được gộp vào cùng một block liền.
    Dấu câu được chèn ngay sau token TLLa tương ứng BÊN TRONG block.

    Ví dụ: tlla_str="x0ch2b5m2l2 Tuấn 123", punct_map=[{position:2, punct:","}]
    → vi token 0=xin(x0), 1=chào(ch2), 2=bạn(b5), 3=mình(m2), 4=là(l2)
    → chèn "," sau b5 → "x0ch2b5,m2l2 Tuấn 123"
    """
    if not punct_map and not trailing:
        return tlla_str

    parts = [p for p in tlla_str.split(' ') if p]

    # Xác định mỗi part cover bao nhiêu vi_word và là TLLa block hay không
    part_info = []
    vi_idx = 0
    for part in parts:
        tlla_toks = parse_tlla_string(part)
        rebuilt = ''.join(tlla_toks)
        is_tlla_block = (rebuilt == part and bool(tlla_toks))
        count = len(tlla_toks) if is_tlla_block else 1
        part_info.append({
            'part': part,
            'start': vi_idx,
            'end': vi_idx + count - 1,
            'is_tlla': is_tlla_block,
            'tokens': tlla_toks if is_tlla_block else None,
        })
        vi_idx += count

    # Lookup: vi_position → punct
    punct_lookup = {e['position']: e['punct'] for e in punct_map if e.get('after', True)}

    result_parts = []
    for info in part_info:
        if info['is_tlla']:
            # Chèn dấu câu sau đúng token bên trong block
            rebuilt = ''
            for ti, tok in enumerate(info['tokens']):
                rebuilt += tok
                vi_pos = info['start'] + ti
                if vi_pos in punct_lookup:
                    rebuilt += punct_lookup[vi_pos]
            result_parts.append(rebuilt)
        else:
            part_str = info['part']
            if info['end'] in punct_lookup:
                part_str = part_str + punct_lookup[info['end']]
            result_parts.append(part_str)

    joined = ' '.join(result_parts)
    if trailing:
        joined = joined.rstrip() + trailing
    return joined


def insert_punctuation_into_vi_words(vi_words: list, punct_map: list, trailing: str) -> str:
    """Chèn dấu câu vào list tiếng Việt (dùng cho chiều TLLa→VI)."""
    result = list(vi_words)
    for entry in sorted(punct_map, key=lambda x: x['position'], reverse=True):
        pos = entry['position']
        if pos < len(result) and entry.get('after', True):
            result[pos] = result[pos] + entry['punct']
    joined = ' '.join(result)
    if trailing:
        joined = joined.rstrip() + trailing
    return joined


def extract_punctuation_from_tlla(text: str) -> dict:
    """
    Tách dấu câu khỏi chuỗi TLLa hỗn hợp.

    Xử lý:
    - Dấu câu cuối chuỗi (trailing): . ! ? ở cuối sau phần non-TLLa hoặc số
    - Dấu câu inline trong TLLa block: , ; : nằm giữa các token TLLa
    - Dấu câu phân câu trong TLLa block: . ! ? nằm GIỮA các token TLLa
      Ví dụ: "x0ch2.m2l2" → tách thành tokens [x0ch2, m2l2] với punct . sau token 0

    Trả về:
      {
        'cleaned': str,      # TLLa đã bỏ dấu câu
        'punct_map': list,   # [{'position': int, 'punct': str, 'after': bool}]
        'trailing': str,     # dấu câu cuối chuỗi
      }
    """
    # Tách trailing punctuation ở cuối chuỗi (chỉ khi đứng sau ký tự không phải TLLa token)
    trailing = ''
    # Trailing chỉ có nếu chuỗi kết thúc bằng dấu câu sau phần non-TLLa/số hoặc ở cuối hẳn
    # Không tách trailing nếu dấu câu nằm ngay giữa block TLLa
    m_trail = re.search(r'([.!?]+)\s*$', text)
    if m_trail:
        before = text[:m_trail.start()]
        # Chỉ là trailing nếu trước dấu không phải TLLa token kết thúc bằng số
        # Tức là: sau một non-TLLa word, hoặc sau khoảng trắng
        if not re.search(r'[a-zA-Z]\d$', before.rstrip()):
            trailing = m_trail.group(1)
            text = before.strip()

    punct_map = []
    parts = text.split(' ')
    cleaned_parts = []
    global_tok_idx = 0

    for part in parts:
        if not part:
            continue

        # Tách TLLa token, dấu câu (bao gồm cả . ! ? giữa token)
        # Pattern: TLLa token HOẶC dấu câu , ; : . ! ?
        tlla_and_punct = re.findall(r'([a-zA-Z]+\d)|([,;:.!?]+)', part)

        if tlla_and_punct:
            tok_sequence = []
            for tllatok, punct in tlla_and_punct:
                if tllatok:
                    tok_sequence.append(('tlla', tllatok))
                elif punct:
                    tok_sequence.append(('punct', punct))

            tlla_tokens_in_part = [t[1] for t in tok_sequence if t[0] == 'tlla']

            if tlla_tokens_in_part:
                local_tlla_idx = 0
                for kind, val in tok_sequence:
                    if kind == 'tlla':
                        local_tlla_idx += 1
                    elif kind == 'punct':
                        punct_map.append({
                            'position': global_tok_idx + local_tlla_idx - 1,
                            'punct': val,
                            'after': True,
                        })

                global_tok_idx += len(tlla_tokens_in_part)
                cleaned_parts.append(''.join(tlla_tokens_in_part))
            else:
                # Chỉ có dấu câu, không có TLLa token
                cleaned_parts.append(part)
                global_tok_idx += 1
        else:
            # Non-TLLa thuần: kiểm tra dấu câu gắn cuối
            m = re.match(r'^(.*?)([,;:]+)$', part)
            if m and m.group(1):
                cleaned_parts.append(m.group(1))
                punct_map.append({
                    'position': global_tok_idx,
                    'punct': m.group(2),
                    'after': True,
                })
            else:
                cleaned_parts.append(part)
            global_tok_idx += 1

    cleaned = ' '.join(p for p in cleaned_parts if p)
    return {
        'cleaned': cleaned,
        'punct_map': punct_map,
        'trailing': trailing,
    }


# ─── TLLa predict helpers ────────────────────────────────────────────────────────

# ─── Predict cache ────────────────────────────────────────────────────────────
# Cache keyed trên chuỗi token đã join bằng space (sau khi tách TLLa).
# maxsize=4096: ~4k entry, mỗi entry ~100B → ~400KB RAM, đổi lấy tốc độ.
_predict_cache: dict = {}


def predict_tlla(tlla_text: str) -> str:
    """Dịch một chuỗi TLLa (có thể liền nhau) sang tiếng Việt, có cache."""
    words = parse_tlla_string(tlla_text)
    if not words:
        return tlla_text
    joined = ' '.join(words)
    if joined in _predict_cache:
        return _predict_cache[joined]
    im = get_input_method()
    try:
        preds = im.predict(joined, "", vni_tones=True)
        result = preds[0] if preds else tlla_text
    except Exception:
        result = tlla_text
    _predict_cache[joined] = result
    return result


def predict_tlla_batch(token_list: list[str]) -> list[str]:
    """
    Dịch nhiều TLLa token riêng lẻ cùng lúc bằng 1 lần gọi im.predict.
    Gọi im.predict với toàn bộ tokens ghép bằng ' | ' làm separator,
    rồi tách kết quả theo ' | '.
    Nếu model không hỗ trợ format đó, fallback sang predict từng token.

    Dùng cho bước 1b3 (_align_and_verify_tokens).
    """
    if not token_list:
        return []

    # Kiểm tra cache trước — nếu tất cả đã có cache thì không cần gọi AI
    results = [_predict_cache.get(t) for t in token_list]
    if all(r is not None for r in results):
        return results

    # Thử batch: ghép tất cả token thành 1 chuỗi để dịch 1 lần
    # Dùng newline làm separator vì model xử lý từng dòng độc lập
    im = get_input_method()
    joined_all = "\n".join(token_list)
    try:
        preds = im.predict(joined_all, "", vni_tones=True)
        if preds and isinstance(preds[0], str):
            batch_result = preds[0].split("\n")
            if len(batch_result) == len(token_list):
                for tok, res in zip(token_list, batch_result):
                    _predict_cache[tok] = res
                return batch_result
    except Exception:
        pass

    # Fallback: predict từng token, dùng cache
    out = []
    for tok in token_list:
        if tok in _predict_cache:
            out.append(_predict_cache[tok])
        else:
            try:
                preds = im.predict(tok, "", vni_tones=True)
                res = preds[0] if preds else tok
            except Exception:
                res = tok
            _predict_cache[tok] = res
            out.append(res)
    return out


def predict_vi_to_tlla_full(text: str) -> str:
    """
    Dịch TOÀN BỘ câu tiếng Việt sang TLLa bằng AI.
    Trả về chuỗi TLLa có space giữa các token.
    """
    im = get_input_method()
    try:
        result = im.predict_vi_to_tlla(text)
        if result:
            return result if isinstance(result, str) else ' '.join(result)
        # Fallback: dịch từng từ nếu AI không hỗ trợ chiều ngược
        return _fallback_vi_to_tlla_word_by_word(text)
    except (AttributeError, Exception):
        return _fallback_vi_to_tlla_word_by_word(text)


def _fallback_vi_to_tlla_word_by_word(text: str) -> str:
    """Fallback: dịch từng từ tiếng Việt sang TLLa."""
    words = text.split()
    result_tokens = []
    for word in words:
        triplet = Vietnamese.analyze(word.lower())
        if isinstance(triplet, Triplet):
            consonant, rhyme, tone = triplet.unpack()
            tlla = ("" if consonant == '0' else consonant) + rhyme + str(tone)
            result_tokens.append(tlla)
        else:
            result_tokens.append(word)
    return ' '.join(result_tokens)


# ─── Chiều TLLa → Tiếng Việt ────────────────────────────────────────────────────

def translate_tlla_sentence_to_vi(sentence: str) -> str:
    """
    Dịch một câu TLLa (có thể hỗn hợp) sang tiếng Việt.

    Pipeline:
      2a. Tách dấu câu khỏi TLLa
      2b1. Thêm space vào block TLLa liền
      2b2. Dịch toàn bộ câu (theo từng segment)
      2b3. Ghép với phần non-TLLa
      2c. Chèn lại dấu câu
    """
    if not sentence.strip():
        return sentence

    # Bước 2a: Tách dấu câu
    punct_info = extract_punctuation_from_tlla(sentence)
    cleaned = punct_info['cleaned']
    punct_map = punct_info['punct_map']
    trailing = punct_info['trailing']

    # Phân tách segments (TLLa block vs non-TLLa)
    # Space-separated parts
    parts = cleaned.split(' ')
    vi_words = []

    for part in parts:
        if not part:
            continue
        tlla_tokens = parse_tlla_string(part)
        rebuilt = ''.join(tlla_tokens)
        if rebuilt == part and tlla_tokens:
            # Bước 2b1: thêm space
            spaced = add_spaces_to_tlla(part)
            # Bước 2b2: dịch toàn block
            translated = predict_tlla(spaced)
            vi_words.extend(translated.split())
        else:
            # Non-TLLa: giữ nguyên
            vi_words.append(part)

    # Bước 2c: Chèn lại dấu câu
    result = insert_punctuation_into_vi_words(vi_words, punct_map, trailing)
    return re.sub(r'\s+', ' ', result).strip()


def translate_mixed_tlla_to_vi(text: str) -> str:
    """
    Dịch text TLLa hỗn hợp sang tiếng Việt thuần.
    Tách câu → dịch CÁC CÂU SONG SONG → ghép lại theo thứ tự gốc.
    """
    if not text or not text.strip():
        return text

    sentences = split_sentences(text)
    if len(sentences) == 1:
        return translate_tlla_sentence_to_vi(sentences[0])

    futures = {
        _EXECUTOR.submit(translate_tlla_sentence_to_vi, sent): idx
        for idx, sent in enumerate(sentences)
    }
    results_by_idx = {}
    for future in as_completed(futures):
        idx = futures[future]
        results_by_idx[idx] = future.result()

    translated = [results_by_idx[i] for i in range(len(sentences))]
    return join_sentences(translated)


# ─── Chiều Tiếng Việt → TLLa ────────────────────────────────────────────────────

def translate_vi_sentence_to_tlla(sentence: str) -> dict:
    """
    Dịch một câu tiếng Việt sang TLLa theo pipeline mới.

    Pipeline:
      1a. Trích xuất dấu câu
      1b1. Dịch TOÀN BỘ cleaned text sang TLLa bằng AI
      1b2. Kiểm tra ngược
      1b3. Thay thế token bị lỗi bằng bản gốc (non-TLLa)
      1b4. Kiểm tra lại
      1c. Gộp TLLa liên tiếp thành block
      1d. Rút gọn tối đa
      1e. Chèn dấu câu

    Trả về dict với log từng bước.
    """
    if not sentence.strip():
        return {'result': sentence, 'steps': []}

    steps = []

    # Bước 1a: Trích xuất dấu câu
    punct_info = extract_punctuation_from_vietnamese(sentence)
    cleaned_vi = punct_info['cleaned']
    punct_map = punct_info['punct_map']
    trailing = punct_info['trailing']

    steps.append({
        'step': '1a',
        'desc': 'Trích xuất dấu câu',
        'cleaned': cleaned_vi,
        'punct_map': punct_map,
        'trailing': trailing,
    })

    vi_words = cleaned_vi.split()

    # Bước 1b1: Dịch TOÀN BỘ câu sang TLLa
    tlla_full_str = predict_vi_to_tlla_full(cleaned_vi)
    tlla_tokens = tlla_full_str.split()

    steps.append({
        'step': '1b1',
        'desc': 'Dịch toàn bộ câu sang TLLa',
        'input': cleaned_vi,
        'output': tlla_full_str,
    })

    # Bước 1b2: Kiểm tra ngược toàn câu
    back_translation = predict_tlla(' '.join(
        t for t in tlla_tokens if is_tlla_token(t)
    ) if any(is_tlla_token(t) for t in tlla_tokens) else tlla_full_str)

    steps.append({
        'step': '1b2',
        'desc': 'Kiểm tra ngược',
        'back': back_translation,
        'match': normalize_compare(back_translation) == normalize_compare(cleaned_vi),
    })

    # Bước 1b3: Thay thế từng token bị lỗi
    # Căn chỉnh tlla_tokens với vi_words
    final_tokens = _align_and_verify_tokens(vi_words, tlla_tokens)

    steps.append({
        'step': '1b3',
        'desc': 'Thay thế token lỗi bằng bản gốc',
        'tokens': final_tokens,
    })

    # Bước 1b4: Kiểm tra lại toàn câu
    # (đã được xử lý trong _align_and_verify_tokens)

    # Bước 1c: Gộp TLLa liên tiếp thành block
    step2 = _tokens_to_tlla_string(final_tokens)

    steps.append({
        'step': '1c',
        'desc': 'Gộp TLLa liên tiếp thành block',
        'result': step2,
    })

    # Bước 1d: Rút gọn tối đa
    minimize_result = minimize_tlla(step2, cleaned_vi)

    steps.append({
        'step': '1d',
        'desc': 'Rút gọn tối đa',
        'result': minimize_result['result'],
        'minimize_steps': minimize_result['steps'],
    })

    minimized = minimize_result['result']

    # Bước 1e: Chèn dấu câu vào TLLa
    # Dùng insert_punctuation_into_tlla_string để ánh xạ đúng vi_word index
    # kể cả khi nhiều token đã được gộp thành block liền.
    final_with_punct = insert_punctuation_into_tlla_string(minimized, punct_map, trailing)

    steps.append({
        'step': '1e',
        'desc': 'Chèn dấu câu',
        'result': final_with_punct,
    })

    return {
        'result': final_with_punct,
        'steps': steps,
        'step2': step2,
        'minimize_steps': minimize_result['steps'],
    }


def _align_and_verify_tokens(vi_words: list, tlla_tokens: list) -> list:
    """
    Căn chỉnh tlla_tokens với vi_words và kiểm tra từng token.
    Nếu token TLLa dịch ngược không khớp với từ gốc → giữ nguyên từ gốc.

    Tối ưu: gom tất cả TLLa token cần kiểm tra → 1 lần gọi predict_tlla_batch
    thay vì N lần gọi riêng lẻ.

    Trả về list: [{'original': str, 'tlla': str|None, 'verified': bool}]
    """
    n_vi = len(vi_words)
    n_tlla = len(tlla_tokens)

    # Bước 1: phân loại sơ bộ, thu thập các token cần batch-verify
    items = []          # (vi_word, tlla_tok_or_None, need_verify)
    to_verify = []      # (idx_in_items, tlla_tok) — cần gọi AI

    for i, vi_word in enumerate(vi_words):
        tlla_tok = tlla_tokens[i] if i < n_tlla else None
        if tlla_tok and is_tlla_token(tlla_tok):
            items.append((vi_word, tlla_tok, True))
            to_verify.append((len(items) - 1, tlla_tok))
        else:
            items.append((vi_word, None, False))

    # Bước 2: batch predict — 1 lần AI call cho tất cả token cần verify
    if to_verify:
        toks = [tlla_tok for _, tlla_tok in to_verify]
        backs = predict_tlla_batch(toks)
        for (item_idx, tlla_tok), back in zip(to_verify, backs):
            vi_word, _, _ = items[item_idx]
            ok = normalize_compare(back) == normalize_compare(vi_word)
            items[item_idx] = (vi_word, tlla_tok if ok else None, ok)

    # Bước 3: build result
    return [
        {'original': vi_word, 'tlla': tlla_tok, 'verified': bool(tlla_tok)}
        for vi_word, tlla_tok, _ in items
    ]


def _tokens_to_tlla_string(token_list: list) -> str:
    """
    Từ list token dạng [{'tlla': str|None, 'original': str}],
    gộp TLLa liên tiếp thành block liền, non-TLLa giữ nguyên.
    """
    parts = []
    tlla_run = []

    def flush_run():
        if tlla_run:
            parts.append(''.join(tlla_run))
            tlla_run.clear()

    for tok in token_list:
        if tok.get('tlla'):
            tlla_run.append(tok['tlla'])
        else:
            flush_run()
            parts.append(tok['original'])

    flush_run()
    return ' '.join(parts)


def translate_vi_to_tlla(text: str) -> dict:
    """
    Dịch toàn bộ văn bản tiếng Việt sang TLLa.
    Tách câu → dịch CÁC CÂU SONG SONG → ghép lại theo thứ tự gốc.

    Tối ưu: dùng ThreadPoolExecutor để các câu chạy đồng thời,
    giúp tận dụng I/O wait khi AI inference không bị GIL block.
    """
    sentences = split_sentences(text)

    if len(sentences) == 1:
        # Không cần thread overhead cho 1 câu
        sent_result = translate_vi_sentence_to_tlla(sentences[0])
        return {
            'result': sent_result['result'],
            'sentence_steps': [{
                'sentence': sentences[0],
                'result': sent_result['result'],
                'steps': sent_result['steps'],
            }],
        }

    # Nhiều câu: submit tất cả song song
    futures = {
        _EXECUTOR.submit(translate_vi_sentence_to_tlla, sent): idx
        for idx, sent in enumerate(sentences)
    }

    results_by_idx = {}
    for future in as_completed(futures):
        idx = futures[future]
        results_by_idx[idx] = future.result()

    all_results = []
    sentence_steps = []
    for idx, sent in enumerate(sentences):
        sent_result = results_by_idx[idx]
        all_results.append(sent_result['result'])
        sentence_steps.append({
            'sentence': sent,
            'result': sent_result['result'],
            'steps': sent_result['steps'],
        })

    final = join_sentences(all_results, is_tlla=True)
    return {
        'result': final,
        'sentence_steps': sentence_steps,
    }


# ─── Minimize (Bước 1d) ───────────────────────────────────────────────────────

def shorten_tlla_token(token: str) -> str | None:
    """
    Rút 1 ký tự từ cuối phần chữ của token TLLa.
    Trả về None nếu phần chữ chỉ còn 1 ký tự.
    """
    m = re.match(r'^([a-zA-Z.]+)(\d)$', token)
    if not m:
        return None
    letters, tone = m.group(1), m.group(2)
    core = letters.rstrip('.')
    if len(core) <= 1:
        return None
    return core[:-1] + tone


def parse_display_string(display: str) -> list:
    """
    Tách chuỗi TLLa (bước 1c) thành list segments:
    [{ 'type': 'tlla_block'|'non_tlla', 'content': str, 'tokens': list }]
    """
    segments = []
    for part in display.split(' '):
        if not part:
            continue
        parsed = parse_tlla_string(part)
        rebuilt = ''.join(parsed)
        if rebuilt == part and parsed:
            segments.append({'type': 'tlla_block', 'content': part, 'tokens': parsed})
        else:
            segments.append({'type': 'non_tlla', 'content': part})
    return segments


def segments_to_full_string(segments: list) -> str:
    return ' '.join(s['content'] for s in segments)


def translate_segments(segments: list) -> str:
    """Dịch toàn bộ segments sang tiếng Việt để kiểm tra."""
    parts = []
    for seg in segments:
        if seg['type'] == 'tlla_block':
            spaced = add_spaces_to_tlla(seg['content'])
            parts.append(predict_tlla(spaced))
        else:
            parts.append(seg['content'])
    return re.sub(r'\s+', ' ', ' '.join(parts)).strip()


def _translate_segment_only(seg: dict) -> str:
    """Dịch một segment TLLa block duy nhất, có cache."""
    if seg['type'] == 'tlla_block':
        return predict_tlla(add_spaces_to_tlla(seg['content']))
    return seg['content']


def minimize_tlla(initial_tlla: str, original_vi: str) -> dict:
    """
    Rút gọn tối đa chuỗi TLLa từ cuối câu về đầu.

    Tối ưu tốc độ:
    1. Cache kết quả dịch từng segment — các segment không thay đổi
       không bị dịch lại trong mỗi vòng lặp.
    2. Khi thử rút ngắn token, CHỈ dịch lại segment đang thay đổi,
       ghép với cached translations của các segment khác.
    3. Nhờ predict cache, cùng một chuỗi TLLa không bao giờ gọi AI 2 lần.
    """
    original_norm = normalize_compare(original_vi)
    steps = []

    segments = parse_display_string(initial_tlla)

    # Khởi tạo segment translation cache
    seg_cache = [_translate_segment_only(seg) for seg in segments]

    all_token_positions = []
    for si, seg in enumerate(segments):
        if seg['type'] == 'tlla_block':
            for ti in range(len(seg['tokens'])):
                all_token_positions.append((si, ti))

    for si, ti in reversed(all_token_positions):
        seg = segments[si]
        current_token = seg['tokens'][ti]

        steps.append({
            'action': 'start_token',
            'token': current_token,
            'full_string': segments_to_full_string(segments),
        })

        while True:
            shorter = shorten_tlla_token(current_token)
            if shorter is None:
                steps.append({'action': 'limit_reached', 'token': current_token})
                break

            # Tạo trial segment CHỈ cho segment đang thay đổi
            trial_tokens = seg['tokens'][:]
            trial_tokens[ti] = shorter
            trial_seg = {**seg, 'tokens': trial_tokens, 'content': ''.join(trial_tokens)}

            # Dịch trial segment (có cache) — KHÔNG dịch lại các segment khác
            trial_seg_vi = _translate_segment_only(trial_seg)

            # Ghép với cached translations của các segment khác
            trial_parts = [
                trial_seg_vi if i == si else seg_cache[i]
                for i in range(len(segments))
            ]
            trial_vi = re.sub(r'\s+', ' ', ' '.join(trial_parts)).strip()
            trial_norm = normalize_compare(trial_vi)
            ok = trial_norm == original_norm

            steps.append({
                'action': 'try_shorten',
                'from': current_token,
                'to': shorter,
                'result_vi': trial_vi,
                'ok': ok,
            })

            if ok:
                current_token = shorter
                seg['tokens'][ti] = shorter
                seg['content'] = ''.join(seg['tokens'])
                segments[si] = seg
                seg_cache[si] = trial_seg_vi  # cập nhật cache segment này
            else:
                steps.append({'action': 'revert', 'keep': current_token})
                break

    final_string = segments_to_full_string(segments)
    return {'result': final_string, 'steps': steps}


# ─── API endpoints ─────────────────────────────────────────────────────────────

@app.route('/api/translate', methods=['POST'])
def translate():
    data = request.get_json(force=True, silent=True) or {}
    raw = data.get('text', '')
    direction = data.get('direction', 'tlla->vi')

    if not isinstance(raw, str) or not raw.strip():
        return jsonify({'success': False, 'error': 'Vui lòng nhập văn bản.'}), 400

    raw = raw.strip()

    # ── TLLa → Tiếng Việt ──
    if direction == 'tlla->vi':
        result = translate_mixed_tlla_to_vi(raw)
        return jsonify({'success': True, 'text': result, 'guidance': GUIDANCE})

    # ── Tiếng Việt → TLLa ──
    if direction == 'vi->tlla':
        vi_result = translate_vi_to_tlla(raw)

        # Thu thập step1, step2, step3 từ câu đầu tiên để backward compat với frontend
        first_sent = vi_result['sentence_steps'][0] if vi_result['sentence_steps'] else {}
        step1_blocks = []
        step2_str = first_sent.get('result', '')
        step3_steps = []

        for sent_info in vi_result['sentence_steps']:
            for step in sent_info.get('steps', []):
                if step.get('step') == '1b3':
                    step1_blocks.extend(step.get('tokens', []))
                if step.get('step') == '1c':
                    step2_str = step.get('result', step2_str)
                if step.get('step') == '1d':
                    step3_steps.extend(step.get('minimize_steps', []))

        return jsonify({
            'success': True,
            'text': vi_result['result'],
            'step1': step1_blocks,
            'step2': step2_str,
            'step3_steps': step3_steps,
            'sentence_steps': vi_result['sentence_steps'],
            'guidance': GUIDANCE,
        })

    return jsonify({'success': False, 'error': 'Hướng dịch không hợp lệ.'}), 400


@app.route('/api/tlla_to_vi_simple', methods=['POST'])
def tlla_to_vi_simple():
    """Endpoint đơn giản chỉ dịch TLLa -> Việt, dùng cho kiểm tra realtime."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'success': False, 'error': 'Empty'}), 400
    result = translate_mixed_tlla_to_vi(text)
    return jsonify({'success': True, 'text': result})


# ─── TLC Compression endpoints ─────────────────────────────────────────────────

@app.route('/api/compress', methods=['POST'])
def compress_endpoint():
    """
    Pipeline đầy đủ: Tiếng Việt → TLLa rút gọn → TLC compact string.
    """
    data = request.get_json(force=True, silent=True) or {}
    raw = data.get('text', '').strip()
    if not raw:
        return jsonify({'success': False, 'error': 'Vui lòng nhập văn bản.'}), 400

    try:
        # Pipeline TLLa theo cơ chế mới (có tách câu, xử lý dấu câu)
        vi_result = translate_vi_to_tlla(raw)
        tlla_final = vi_result['result']

        # Thu thập metadata bước để trả về frontend
        step1_blocks = []
        step2_str = tlla_final
        step3_steps = []
        for sent_info in vi_result['sentence_steps']:
            for step in sent_info.get('steps', []):
                if step.get('step') == '1b3':
                    step1_blocks.extend(step.get('tokens', []))
                if step.get('step') == '1c':
                    step2_str = step.get('result', step2_str)
                if step.get('step') == '1d':
                    step3_steps.extend(step.get('minimize_steps', []))

        # Pipeline TLC compression
        compressor = get_compressor()
        split = compressor.compress_split(tlla_final)
        compact = split['combined']
        stats = compressor.stats(tlla_final, compact)

        vi_bytes = len(raw.encode('utf-8'))
        stats['original_vi_bytes'] = vi_bytes
        stats['vi_vs_compact_ratio'] = round(stats['compressed_bytes'] / vi_bytes, 4) if vi_bytes else 1
        stats['vi_vs_compact_saving_pct'] = round((1 - stats['vi_vs_compact_ratio']) * 100, 1)

        # Thống kê riêng 2 stream (chỉ có khi mode=TI)
        char_bytes_len = len(split['char_compact'].encode('ascii')) if split['char_compact'] else 0
        tone_bytes_len = len(split['tone_compact'].encode('ascii')) if split['tone_compact'] else 0

        return jsonify({
            'success': True,
            'tlla': tlla_final,
            'compact': compact,
            'char_stream': split['char_compact'],
            'tone_stream': split['tone_compact'],
            'compress_mode': split['mode'],
            'char_bytes': char_bytes_len,
            'tone_bytes': tone_bytes_len,
            'stats': stats,
            'step1': step1_blocks,
            'step2': step2_str,
            'step3_steps': step3_steps,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/decompress', methods=['POST'])
def decompress_endpoint():
    """
    Giải nén: compact string → TLLa → Tiếng Việt.
    """
    data = request.get_json(force=True, silent=True) or {}
    compact = data.get('compact', '').strip()
    if not compact:
        return jsonify({'success': False, 'error': 'Vui lòng nhập compact string.'}), 400

    try:
        compressor = get_compressor()
        tlla = compressor.decompress(compact)
        vi = translate_mixed_tlla_to_vi(tlla)
        return jsonify({'success': True, 'tlla': tlla, 'vi': vi})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─── .thualuu file endpoints ──────────────────────────────────────────────────

@app.route('/api/export_thualuu', methods=['POST'])
def export_thualuu():
    """
    Xuất TLC compact string ra file .thualuu (plain text, UTF-8).
    Frontend nhận blob và download về máy.

    Input:  { "compact": "..." }
    Output: file binary (text/plain) với tên gợi ý document.thualuu
    """
    data = request.get_json(force=True, silent=True) or {}
    compact = data.get('compact', '').strip()
    if not compact:
        return jsonify({'success': False, 'error': 'Không có dữ liệu để xuất.'}), 400

    # File .thualuu: plain text UTF-8, chỉ chứa compact string + newline cuối
    content_bytes = (compact + '\n').encode('utf-8')

    from flask import Response
    return Response(
        content_bytes,
        status=200,
        mimetype='text/plain; charset=utf-8',
        headers={
            'Content-Disposition': 'attachment; filename="document.thualuu"',
            'Content-Length': str(len(content_bytes)),
        }
    )


@app.route('/api/import_thualuu', methods=['POST'])
def import_thualuu():
    """
    Đọc file .thualuu được upload, trả về compact string bên trong.

    Input:  multipart/form-data với field "file"
    Output: { "success": true, "compact": "..." }
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Không tìm thấy file trong request.'}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'File rỗng hoặc không hợp lệ.'}), 400

    # Kiểm tra đuôi file (không bắt buộc nhưng nên báo lỗi rõ)
    if not file.filename.lower().endswith('.thualuu'):
        return jsonify({'success': False, 'error': 'Chỉ chấp nhận file .thualuu.'}), 400

    try:
        raw = file.read().decode('utf-8').strip()
        if not raw:
            return jsonify({'success': False, 'error': 'File trống.'}), 400
        # compact string là toàn bộ nội dung (strip whitespace)
        compact = raw
        return jsonify({'success': True, 'compact': compact})
    except UnicodeDecodeError:
        return jsonify({'success': False, 'error': 'File không phải UTF-8, có thể bị hỏng.'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)