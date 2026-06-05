"""
TLC Compression Engine v3 — Adaptive Vietnamese TLLa Compressor
Bổ sung cơ chế TI (Tone Interleaving):
  - Tách luồng ký tự và luồng tông thành 2 stream riêng
  - Mỗi ký tự trong luồng chữ có 1 giá trị 3-bit tương ứng:
      0-5  = tông thật (ký tự này là ký tự CUỐI của consonant cluster, kết thúc 1 token TLLa)
      6    = marker non-TLLa (ký tự này thuộc đoạn pass-through, không phải TLLa)
      7    = SKIP (ký tự giữa cluster, chưa kết thúc token)
  - Luồng chữ nén bằng zlib (text thuần, lặp lại nhiều)
  - Luồng tông pack 3-bit/entry rồi nén bằng zlib (SKIP lặp nhiều → nén tốt)
Magic byte mới: \xA7\x03 (MAGIC_TI)
"""

import re
import zlib
import base64
import struct
import json
from collections import Counter
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Regex nhận dạng 1 token TLLa: 1+ chữ cái ASCII + 1 chữ số (tông 0-5)
# ---------------------------------------------------------------------------
_TLLA_TOKEN_RE = re.compile(r'([a-zA-Z]+)([0-5])')


class TLCCompressor:
    MAGIC_SHORT = b'\xA7\x01'
    MAGIC_LONG  = b'\xA7\x02'
    MAGIC_TI    = b'\xA7\x03'   # Tone Interleaving
    SHORT_THRESHOLD = 80

    # Giá trị tông đặc biệt trong luồng tông
    _TONE_SKIP     = 7   # 111 — ký tự giữa cluster (chưa kết thúc token)
    _TONE_PASSTHRU = 6   # 110 — ký tự non-TLLa (pass-through)

    _ESCAPE_TABLE = [
        ('%',  '\x00PC'), ('(',  '\x00LP'), (')',  '\x00RP'),
        (',',  '\x00CM'), (';',  '\x00SC'), (':',  '\x00CL'),
        ('!',  '\x00EX'), ('?',  '\x00QM'), ('[',  '\x00LB'),
        (']',  '\x00RB'), ('{',  '\x00LC'), ('}',  '\x00RC'),
        ('"',  '\x00DQ'), ("'",  '\x00SQ'), ('\\', '\x00BS'),
        ('#',  '\x00HS'), ('$',  '\x00DL'), ('&',  '\x00AM'),
        ('*',  '\x00ST'), ('+',  '\x00PL'), ('/',  '\x00SL'),
        ('<',  '\x00LT'), ('>',  '\x00GT'), ('=',  '\x00EQ'),
        ('@',  '\x00AT'), ('^',  '\x00CR'), ('_',  '\x00US'),
        ('`',  '\x00BT'), ('|',  '\x00PI'), ('~',  '\x00TL'),
    ]

    # ------------------------------------------------------------------
    # Escape / unescape (giữ nguyên từ v2)
    # ------------------------------------------------------------------
    def _escape(self, text: str) -> str:
        result = text.replace('\x00', '\x00NL')
        for char, placeholder in self._ESCAPE_TABLE:
            result = result.replace(char, placeholder)
        return result

    def _unescape(self, text: str) -> str:
        result = text
        for char, placeholder in reversed(self._ESCAPE_TABLE):
            result = result.replace(placeholder, char)
        result = result.replace('\x00NL', '\x00')
        return result

    # ------------------------------------------------------------------
    # TI encode: tách chuỗi TLLa thành (char_stream, tone_stream)
    # ------------------------------------------------------------------
    def _ti_encode(self, tlla_text: str) -> Tuple[str, List[int]]:
        """
        Duyệt tlla_text, tách thành:
          char_stream : chuỗi tất cả ký tự (TLLa + non-TLLa), không có số tông
          tone_stream : list[int], mỗi phần tử tương ứng 1 ký tự trong char_stream
                        0-5 = tông thật (ký tự cuối cluster → flush token)
                        6   = PASSTHRU (ký tự non-TLLa)
                        7   = SKIP (ký tự giữa cluster)
        """
        char_stream: List[str] = []
        tone_stream: List[int] = []

        pos = 0
        text = tlla_text
        n = len(text)

        while pos < n:
            m = _TLLA_TOKEN_RE.match(text, pos)
            if m:
                letters = m.group(1)   # vd: "ch", "ng", "x"
                tone    = int(m.group(2))
                # Các ký tự trước ký tự cuối → SKIP
                for ch in letters[:-1]:
                    char_stream.append(ch)
                    tone_stream.append(self._TONE_SKIP)
                # Ký tự cuối cluster → mang tông thật
                char_stream.append(letters[-1])
                tone_stream.append(tone)
                pos = m.end()
            else:
                # Ký tự không phải TLLa (khoảng trắng, dấu câu, chữ Unicode...)
                ch = text[pos]
                char_stream.append(ch)
                tone_stream.append(self._TONE_PASSTHRU)
                pos += 1

        return ''.join(char_stream), tone_stream

    # ------------------------------------------------------------------
    # TI decode: ghép lại chuỗi TLLa từ (char_stream, tone_stream)
    # ------------------------------------------------------------------
    def _ti_decode(self, char_stream: str, tone_stream: List[int]) -> str:
        result: List[str] = []
        cluster: List[str] = []

        for ch, tone in zip(char_stream, tone_stream):
            if tone == self._TONE_PASSTHRU:
                # Flush cluster còn dở (không nên xảy ra nếu encode đúng)
                if cluster:
                    result.append(''.join(cluster))
                    cluster = []
                result.append(ch)
            elif tone == self._TONE_SKIP:
                cluster.append(ch)
            else:
                # tone 0-5: ký tự cuối cluster → flush token
                cluster.append(ch)
                result.append(''.join(cluster) + str(tone))
                cluster = []

        # Flush cluster còn dở (edge case)
        if cluster:
            result.append(''.join(cluster))

        return ''.join(result)

    # ------------------------------------------------------------------
    # Pack / unpack tone_stream (3 bit mỗi giá trị)
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_tones(tone_stream: List[int]) -> bytes:
        """Pack list[int 0-7] thành bytearray, 3 bit/giá trị, big-endian."""
        bits = 0
        bit_count = 0
        out = bytearray()
        for t in tone_stream:
            bits = (bits << 3) | (t & 0x07)
            bit_count += 3
            while bit_count >= 8:
                bit_count -= 8
                out.append((bits >> bit_count) & 0xFF)
        if bit_count > 0:
            out.append((bits << (8 - bit_count)) & 0xFF)
        return bytes(out)

    @staticmethod
    def _unpack_tones(data: bytes, count: int) -> List[int]:
        """Unpack bytearray về list[int], lấy đúng `count` giá trị."""
        tones: List[int] = []
        bits = 0
        bit_count = 0
        idx = 0
        while len(tones) < count:
            while bit_count < 3 and idx < len(data):
                bits = (bits << 8) | data[idx]
                bit_count += 8
                idx += 1
            if bit_count < 3:
                break
            bit_count -= 3
            tones.append((bits >> bit_count) & 0x07)
        return tones

    # ------------------------------------------------------------------
    # compress / decompress công khai
    # ------------------------------------------------------------------
    def compress(self, tlla_text: str) -> str:
        if not tlla_text:
            return '~'

        safe_text = self._escape(tlla_text)

        # Ngưỡng ngắn: dùng mode cũ MAGIC_SHORT (đơn giản, nhanh)
        if len(safe_text) < self.SHORT_THRESHOLD:
            text_bytes = safe_text.encode('utf-8')
            compressed = zlib.compress(text_bytes, level=9, wbits=-15)
            payload = self.MAGIC_SHORT + struct.pack('>H', len(text_bytes)) + compressed
            return base64.b85encode(payload).decode('ascii')

        # Mode TI: Tone Interleaving
        char_stream, tone_stream = self._ti_encode(safe_text)

        # Luồng chữ: UTF-8 → zlib
        char_bytes     = char_stream.encode('utf-8')
        char_compressed = zlib.compress(char_bytes, level=9, wbits=-15)

        # Luồng tông: pack 3-bit → zlib
        tone_packed     = self._pack_tones(tone_stream)
        tone_compressed = zlib.compress(tone_packed, level=9, wbits=-15)

        # Header: [magic 2B][tone_count 4B][char_clen 4B][tone_clen 4B]
        # Sau đó: [char_compressed][tone_compressed]
        tone_count = len(tone_stream)
        header = (
            self.MAGIC_TI
            + struct.pack('>I', tone_count)
            + struct.pack('>I', len(char_compressed))
            + struct.pack('>I', len(tone_compressed))
        )
        payload = header + char_compressed + tone_compressed
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
            text_enc  = inner[2 + dict_len:]
            decode_map = json.loads(dict_json.decode('utf-8'))
            sym_encoded = text_enc.decode('utf-8')
            safe_text = self._restore_symbol_dict(sym_encoded, decode_map)

        elif magic == self.MAGIC_TI:
            tone_count  = struct.unpack('>I', payload[2:6])[0]
            char_clen   = struct.unpack('>I', payload[6:10])[0]
            # tone_clen = struct.unpack('>I', payload[10:14])[0]  # không cần dùng
            char_compressed = payload[14:14 + char_clen]
            tone_compressed = payload[14 + char_clen:]

            char_bytes   = zlib.decompress(char_compressed, wbits=-15)
            char_stream  = char_bytes.decode('utf-8')

            tone_packed  = zlib.decompress(tone_compressed, wbits=-15)
            tone_stream  = self._unpack_tones(tone_packed, tone_count)

            safe_text = self._ti_decode(char_stream, tone_stream)

        else:
            raise ValueError(f'Unknown magic: {magic!r}')

        return self._unescape(safe_text)

    # ------------------------------------------------------------------
    # Helpers giữ nguyên từ v2 (MAGIC_LONG vẫn dùng được)
    # ------------------------------------------------------------------
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

    def compress_split(self, tlla_text: str) -> dict:
        """
        Giống compress() nhưng trả về 2 stream riêng biệt (Base85 ASCII):
          char_compact : luồng ký tự đã nén
          tone_compact : luồng tông đã pack + nén
          combined     : compact string đầy đủ (dùng để decompress / export)
          mode         : 'SHORT' | 'TI'
        Với SHORT (văn bản ngắn), char_compact = combined, tone_compact = None.
        """
        if not tlla_text:
            return {'char_compact': '~', 'tone_compact': None, 'combined': '~', 'mode': 'SHORT'}

        safe_text = self._escape(tlla_text)

        if len(safe_text) < self.SHORT_THRESHOLD:
            combined = self.compress(tlla_text)
            return {'char_compact': combined, 'tone_compact': None, 'combined': combined, 'mode': 'SHORT'}

        # TI mode: tách thành 2 stream
        char_stream, tone_stream = self._ti_encode(safe_text)

        char_bytes      = char_stream.encode('utf-8')
        char_compressed = zlib.compress(char_bytes, level=9, wbits=-15)

        tone_packed     = self._pack_tones(tone_stream)
        tone_compressed = zlib.compress(tone_packed, level=9, wbits=-15)

        char_compact = base64.b85encode(char_compressed).decode('ascii')
        tone_compact = base64.b85encode(tone_compressed).decode('ascii')

        # combined = full payload để decompress
        tone_count = len(tone_stream)
        header = (
            self.MAGIC_TI
            + struct.pack('>I', tone_count)
            + struct.pack('>I', len(char_compressed))
            + struct.pack('>I', len(tone_compressed))
        )
        combined_payload = header + char_compressed + tone_compressed
        combined = base64.b85encode(combined_payload).decode('ascii')

        return {
            'char_compact': char_compact,
            'tone_compact': tone_compact,
            'combined':     combined,
            'mode':         'TI',
        }

    def stats(self, original_tlla: str, compressed: str) -> dict:
        orig_b = len(original_tlla.encode('utf-8'))
        comp_b = len(compressed.encode('ascii'))
        ratio  = comp_b / orig_b if orig_b > 0 else 1.0
        return {
            'original_chars':   len(original_tlla),
            'original_bytes':   orig_b,
            'compressed_chars': len(compressed),
            'compressed_bytes': comp_b,
            'ratio':      round(ratio, 4),
            'saving_pct': round((1 - ratio) * 100, 1),
        }


_compressor = TLCCompressor()

def get_compressor() -> TLCCompressor:
    return _compressor


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
if __name__ == '__main__':
    c = TLCCompressor()

    # Kiểm tra TI encode/decode trực tiếp
    print("=== TI encode/decode unit test ===")
    samples = [
        "x0ch2b5",
        "x0ch2b5m2l2 Tuấn. x0ch2",
        "ng3tr2nh2th4kh1",
    ]
    for s in samples:
        safe = c._escape(s)
        cs, ts = c._ti_encode(safe)
        restored_safe = c._ti_decode(cs, ts)
        restored = c._unescape(restored_safe)
        ok = "✓" if restored == s else f"✗ got: {restored!r}"
        print(f"  {s!r:40s} → {ok}")

    print()
    print(f"{'Input':<22} {'Orig':>6} {'Comp':>6} {'Ratio':>7} {'Save':>7}  {'Mode':<6}  OK?")
    print("-" * 66)

    tests = [
        ("Short",     "x0ch2b5"),
        ("Mixed",     "x0ch2b5m2l2 Tuấn. x0ch2"),
        ("Medium",    "x0ch2b5m2l2tu2kh0ph2b5t1 " * 4),
        ("Long",      "x0ch2b5m2l2tu2kh0ph2b5t1ng3tr2nh2cu3gi3 " * 8),
        ("Very long", "x0ch2b5m2l2tu2kh0ph2b5t1ng3tr2nh2cu3gi3th4vi3kh1 " * 15),
        ("Mixed long","x0ch2b5 Xin chào bạn mình là Tuấn. x0ch2b5m2l2 " * 10),
    ]

    for desc, tlla in tests:
        compact  = c.compress(tlla)
        restored = c.decompress(compact)
        s        = c.stats(tlla, compact)
        ok       = "✓" if restored == tlla else "✗"
        # Đọc magic để biết mode
        import base64 as _b64
        raw   = _b64.b85decode(compact.encode('ascii'))
        magic = raw[:2]
        mode  = {b'\xA7\x01': 'SHORT', b'\xA7\x02': 'LONG', b'\xA7\x03': 'TI'}.get(magic, '???')
        print(f"{desc:<22} {s['original_bytes']:>6}B {s['compressed_bytes']:>6}B "
              f"{s['ratio']:>7.3f} {s['saving_pct']:>6.1f}%  {mode:<6}  {ok}")