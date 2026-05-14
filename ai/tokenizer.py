import json
from models import Triplet, Word

from utils.decorators import singleton
from utils.path import resource_path

@singleton
class Tokenizer:
    def __init__(
        self, 
        enum_path=resource_path("checkpoints/enum.json"), 
        renum_path=resource_path("checkpoints/renum.json"), 
        renum_crt_path=resource_path("checkpoints/renum_crt.json"),
        verbose=1
    ):  
        self.location = "<ai.tokenizer.Tokenizer>"
        self.PADDING_TOKEN_INDEX = 0
        
        with open(enum_path, 'r', encoding='utf-8') as f:
            self.enum: dict[str, int] = json.load(f)
        with open(renum_path, 'r', encoding='utf-8') as f:
            self.renum: dict[int, str] = json.load(f)
        with open(renum_crt_path, 'r', encoding='utf-8') as f:
            self.renum_crt: list[tuple[str, str, int]] = json.load(f)
        self.renum_triplet = [None] + [Triplet(consonant=c, rhyme=r, tone=t) for c, r, t in self.renum_crt[1:]]
        
        if verbose:
            print(f"{self.location} Đã tải: {len(self.renum_triplet)} token")

                            
    def tokenize(self, words: list[str]) -> list[int]:
        """Chuyển đổi danh sách từ thành danh sách chỉ số token"""
        return [self.enum[word] for word in words if word in self.enum]
    def detokenize(self, tensor: list[int]) -> list[Word]:
        """Chuyển đổi danh sách chỉ số token thành danh sách từ"""
        return [self.renum[id] for id in tensor]
    def analyze(self, tensor: list[int]) -> list[tuple[str, str, int]]:
        """Phân tích danh sách chỉ số token thành danh sách bộ ba (phụ âm, vần, tông)"""
        return [self.renum_crt[id] for id in tensor]
    def triplets(self, tensor: list[int]) -> list[Triplet]:
        """Chuyển đổi danh sách chỉ số token thành danh sách Triplet"""
        return [self.renum_triplet[id] for id in tensor]

tokenizer = Tokenizer()
