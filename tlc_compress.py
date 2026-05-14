"""
TLC Compression Engine v2 — Adaptive Vietnamese TLLa Compressor
"""

import re
import zlib
import base64
import struct
import json
from collections import Counter
from typing import Tuple


class TLCCompressor:
    MAGIC_SHORT = b'\xA7\x01'
    MAGIC_LONG  = b'\xA7\x02'
    SHORT_THRESHOLD = 80

    # Các ký tự đặc biệt có thể xuất hiện trong văn bản TLLa (dấu câu, ký hiệu...)
    # và chuỗi "ngụy trang" tương ứng — dùng prefix \x00 để tránh trùng với nội dung thật
    _ESCAPE_TABLE = [
        ('%',  '\x00PC'),
        ('(',  '\x00LP'),
        (')',  '\x00RP'),
        (',',  '\x00CM'),
        (';',  '\x00SC'),
        (':',  '\x00CL'),
        ('!',  '\x00EX'),
        ('?',  '\x00QM'),
        ('[',  '\x00LB'),
        (']',  '\x00RB'),
        ('{',  '\x00LC'),
        ('}',  '\x00RC'),
        ('"',  '\x00DQ'),
        ("'",  '\x00SQ'),
        ('\\', '\x00BS'),
        ('#',  '\x00HS'),
        ('$',  '\x00DL'),
        ('&',  '\x00AM'),
        ('*',  '\x00ST'),
        ('+',  '\x00PL'),
        ('/',  '\x00SL'),
        ('<',  '\x00LT'),
        ('>',  '\x00GT'),
        ('=',  '\x00EQ'),
        ('@',  '\x00AT'),
        ('^',  '\x00CR'),
        ('_',  '\x00US'),
        ('`',  '\x00BT'),
        ('|',  '\x00PI'),
        ('~',  '\x00TL'),
    ]

    def _escape(self, text: str) -> str:
        """Ngụy trang các ký tự đặc biệt trước khi nén."""
        # Escape '\x00' thật trước (nếu có) để tránh conflict
        result = text.replace('\x00', '\x00NL')
        for char, placeholder in self._ESCAPE_TABLE:
            result = result.replace(char, placeholder)
        return result

    def _unescape(self, text: str) -> str:
        """Khôi phục các ký tự đặc biệt sau khi giải nén."""
        result = text
        for char, placeholder in reversed(self._ESCAPE_TABLE):
            result = result.replace(placeholder, char)
        # Unescape '\x00' thật cuối cùng
        result = result.replace('\x00NL', '\x00')
        return result

    def compress(self, tlla_text: str) -> str:
        if not tlla_text:
            return '~'
        # Ngụy trang ký tự đặc biệt trước khi đưa vào bộ nén
        safe_text = self._escape(tlla_text)
        text_bytes = safe_text.encode('utf-8')
        if len(safe_text) < self.SHORT_THRESHOLD:
            compressed = zlib.compress(text_bytes, level=9, wbits=-15)
            payload = self.MAGIC_SHORT + struct.pack('>H', len(text_bytes)) + compressed
        else:
            sym_encoded, decode_map = self._apply_symbol_dict(safe_text)
            dict_json = json.dumps(decode_map, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            text_enc = sym_encoded.encode('utf-8')
            inner = struct.pack('>H', len(dict_json)) + dict_json + text_enc
            compressed = zlib.compress(inner, level=9, wbits=-15)
            payload = self.MAGIC_LONG + compressed
        return base64.b85encode(payload).decode('ascii')

    def decompress(self, compact: str) -> str:
        if not compact or compact == '~':
            return ''
        try:
            payload = base64.b85decode(compact.encode('ascii'))
        except Exception as e:
            raise ValueError(f'Base85 decode error: {e}')
        magic = payload[:2]
        if magic == self.MAGIC_SHORT:
            text_bytes = zlib.decompress(payload[4:], wbits=-15)
            safe_text = text_bytes.decode('utf-8')
        elif magic == self.MAGIC_LONG:
            inner = zlib.decompress(payload[2:], wbits=-15)
            dict_len = struct.unpack('>H', inner[:2])[0]
            dict_json = inner[2:2 + dict_len]
            text_enc = inner[2 + dict_len:]
            decode_map = json.loads(dict_json.decode('utf-8'))
            sym_encoded = text_enc.decode('utf-8')
            safe_text = self._restore_symbol_dict(sym_encoded, decode_map)
        else:
            raise ValueError(f'Unknown magic: {magic!r}')
        # Khôi phục ký tự đặc biệt sau khi giải nén
        return self._unescape(safe_text)

    def _build_ngram_dict(self, tlla_text: str):
        tokens = re.findall(r'[a-zA-Z.]+\d', tlla_text)
        ngram_counts = Counter()
        for tok in tokens:
            m = re.match(r'^(.+?)(\d)$', tok)
            if not m:
                continue
            s = m.group(1)
            for n in (2, 3):
                for i in range(len(s) - n + 1):
                    ngram_counts[s[i:i+n]] += 1
        return ngram_counts

    def _apply_symbol_dict(self, tlla_text: str) -> Tuple[str, dict]:
        ngram_counts = self._build_ngram_dict(tlla_text)
        symbols = '!#$%&\'()*+,/:;<=>?@[\\]^_`{|}~'
        decode_map = {}
        encode_map = {}
        sym_idx = 0
        candidates = [(ng, cnt) for ng, cnt in ngram_counts.most_common(len(symbols))
                     if cnt >= 2 and len(ng) >= 2]
        for ngram, count in candidates:
            if sym_idx >= len(symbols):
                break
            saving = (len(ngram.encode('utf-8')) - 1) * count
            if saving <= 0:
                continue
            sym = symbols[sym_idx]
            encode_map[ngram] = sym
            decode_map[sym] = ngram
            sym_idx += 1
        result = tlla_text
        for ng, sym in sorted(encode_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(ng, sym)
        return result, decode_map

    def _restore_symbol_dict(self, encoded: str, decode_map: dict) -> str:
        result = encoded
        for sym, ng in decode_map.items():
            result = result.replace(sym, ng)
        return result

    def stats(self, original_tlla: str, compressed: str) -> dict:
        orig_b = len(original_tlla.encode('utf-8'))
        comp_b = len(compressed.encode('ascii'))
        ratio = comp_b / orig_b if orig_b > 0 else 1.0
        return {
            'original_chars': len(original_tlla),
            'original_bytes': orig_b,
            'compressed_chars': len(compressed),
            'compressed_bytes': comp_b,
            'ratio': round(ratio, 4),
            'saving_pct': round((1 - ratio) * 100, 1),
        }


_compressor = TLCCompressor()

def get_compressor() -> TLCCompressor:
    return _compressor


if __name__ == '__main__':
    c = TLCCompressor()
    tests = [
        ("Short: xin chao", "x0ch2b5"),
        ("Medium", "x0ch2b5m2l2tu2kh0ph2b5t1 " * 4),
        ("Long", "x0ch2b5m2l2tu2kh0ph2b5t1ng3tr2nh2cu3gi3 " * 8),
        ("Very long", "x0ch2b5m2l2tu2kh0ph2b5t1ng3tr2nh2cu3gi3th4vi3kh1 " * 15),
    ]
    print(f"{'Input':<20} {'Orig':>6} {'Comp':>6} {'Ratio':>7} {'Save':>7}  OK?")
    print("-" * 58)
    for desc, tlla in tests:
        compact = c.compress(tlla)
        restored = c.decompress(compact)
        s = c.stats(tlla, compact)
        ok = "✓" if restored == tlla else "✗"
        print(f"{desc:<20} {s['original_bytes']:>6}B {s['compressed_bytes']:>6}B {s['ratio']:>7.3f} {s['saving_pct']:>6.1f}%  {ok}")