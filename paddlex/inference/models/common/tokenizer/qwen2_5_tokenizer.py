# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List

from .qwen2_tokenizer import Qwen2Tokenizer
from .tokenizer_utils_base import AddedToken, TextInput


class MIXQwen2_5_Tokenizer(Qwen2Tokenizer):
    def __init__(self, *args, **kwargs):
        super(MIXQwen2_5_Tokenizer, self).__init__(*args, **kwargs)

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """

        split_special_tokens = kwargs.pop(
            "split_special_tokens", self.split_special_tokens
        )

        all_special_tokens_extended = dict(
            (str(t), t)
            for t in self.all_special_tokens_extended
            if isinstance(t, AddedToken)
        )

        # Add special tokens
        for t in self.added_tokens_decoder:
            token = self.added_tokens_decoder[t]
            if isinstance(token, AddedToken) and token.special:
                all_special_tokens_extended[str(token)] = token
                if str(token) not in self.all_special_tokens:
                    self.all_special_tokens.append(str(token))
                if str(token) not in self.unique_no_split_tokens:
                    self.unique_no_split_tokens.append(str(token))

        self._create_trie(self.unique_no_split_tokens)

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok)
                for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(
                pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text
            )

        if split_special_tokens:
            no_split_token = []
            tokens = [text]
        else:
            no_split_token = set(
                self.unique_no_split_tokens
            )  # don't split on any of the added tokens
            tokens = self.tokens_trie.split(text)

        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = all_special_tokens_extended.get(token, None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here

        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))

        return tokenized_text
