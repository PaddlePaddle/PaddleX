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


def _is_sentence_dot(text, i):
    """
    Check if the given character is a sentence ending punctuation.
    """
    # if the character is not a period, return False
    if text[i] != ".":
        return False
    # previous character
    prev = text[i - 1] if i > 0 else ""
    # next character
    next = text[i + 1] if i + 1 < len(text) else ""
    # previous is digit or letter, then not sentence ending punctuation
    if prev.isdigit() or prev.isalpha():
        return False
    # next is digit or letter, then not sentence ending punctuation
    if next.isdigit() or next.isalpha():
        return False
    # next is a punctuation, then sentence ending punctuation
    if next in ("", " ", "\t", "\n", '"', "'", "”", "’", ")", "】", "」", "》"):
        return True
    return False


def _find_split_pos(text, chunk_size):
    """
    Find the position to split the text into two chunks.

    Args:
        text (str): The original text to be split.
        chunk_size (int): The maximum size of each chunk.

    Returns:
        int: The index where the text should be split.
    """
    center = len(text) // 2
    split_chars = ["\n", "。", ";", "；", "!", "！", "?", "？"]

    # Search forward
    for i in range(center, len(text)):
        if text[i] in split_chars:
            # Check for whitespace around the split character
            j = i + 1
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if j < len(text) and len(text[:j]) <= chunk_size:
                return i, j
        elif text[i] == "." and _is_sentence_dot(text, i):
            j = i + 1
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if j < len(text) and len(text[:j]) <= chunk_size:
                return i, j

    # Search backward
    for i in range(center, 0, -1):
        if text[i] in split_chars:
            j = i + 1
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if len(text[:j]) <= chunk_size:
                return i, j
        elif text[i] == "." and _is_sentence_dot(text, i):
            j = i + 1
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if len(text[:j]) <= chunk_size:
                return i, j

    # If no suitable position is found, split directly
    return min(chunk_size, len(text)), min(chunk_size, len(text))


def split_text_recursive(text, chunk_size, translate_func):
    """
    Split the text recursively and translate each chunk.

    Args:
        text (str): The original text to be split.
        chunk_size (int): The maximum size of each chunk.
        translate_func (callable): A function that translates a single chunk of text.
        results (list): A list to store the translated chunks.

    Returns:
        None
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return translate_func(text)
    else:
        split_pos, end_whitespace = _find_split_pos(text, chunk_size)
        left = text[:split_pos]
        right = text[end_whitespace:]
        whitespace = text[split_pos:end_whitespace]

        if left:
            left_text = split_text_recursive(left, chunk_size, translate_func)
        if right:
            right_text = split_text_recursive(right, chunk_size, translate_func)

        return left_text + whitespace + right_text


def translate_code_block(code_block, chunk_size, translate_func, results):
    """
    Translate a code block and append the result to the results list.

    Args:
        code_block (str): The code block to be translated.
        chunk_size (int): The maximum size of each chunk.
        translate_func (callable): A function that translates a single chunk of text.
        results (list): A list to store the translated chunks.

    Returns:
        None
    """
    lines = code_block.strip().split("\n")
    if lines[0].startswith("```") or lines[0].startswith("~~~"):
        header = lines[0]
        footer = (
            lines[-1]
            if (lines[-1].startswith("```") or lines[-1].startswith("~~~"))
            else ""
        )
        code_content = "\n".join(lines[1:-1]) if footer else "\n".join(lines[1:])
    else:
        header = ""
        footer = ""
        code_content = code_block

    translated_code_lines = split_text_recursive(
        code_content, chunk_size, translate_func
    )

    # drop ``` or ~~~
    filtered_code_lines = [
        line
        for line in translated_code_lines.split("\n")
        if not (line.strip().startswith("```") or line.strip().startswith("~~~"))
    ]
    translated_code = "\n".join(filtered_code_lines)

    result = f"{header}\n{translated_code}\n{footer}" if header else translated_code
    results.append(result)


def translate_html_block(html_block, chunk_size, translate_func, results):
    """
    Translate a HTML block and append the result to the results list.

    Args:
        html_block (str): The HTML block to be translated.
        chunk_size (int): The maximum size of each chunk.
        translate_func (callable): A function that translates a single chunk of text.
        results (list): A list to store the translated chunks.

    Returns:
        None
    """
    from bs4 import BeautifulSoup
    import copy

    # If the HTML is short and simple, translate directly
    if (
        html_block.count("<") < 5
        and html_block.count(">") < 5
        and html_block.count("<") == html_block.count(">")
        and len(html_block) < chunk_size
    ):
        translated = translate_func(html_block)
        results.append(translated)
        return

    soup = BeautifulSoup(html_block, "html.parser")

    td_seen = set()
    td_batch_nodes = []
    td_batch_texts = []

    # Find all <td> and <th> elements and collect their inner HTML for batch translation

    for node in soup.find_all(string=True, recursive=True):
        parent_td = node.find_parent(["td", "th"])
        if parent_td and id(parent_td) not in td_seen:
            td_text = parent_td.decode_contents().strip()
            if td_text:
                td_batch_nodes.append(parent_td)
                td_batch_texts.append(td_text)
            td_seen.add(id(parent_td))
            
    # Process <td>/<th> nodes in batches
    batch_size = chunk_size
    i = 0
    while i < len(td_batch_nodes):
        # A batch of nodes and the assembled content to be translated
        batch_nodes = []
        batch_texts = []
        current_length = 0
        while i < len(td_batch_nodes) and current_length + len(td_batch_texts[i]) <= batch_size:
            batch_nodes.append(td_batch_nodes[i])
            batch_texts.append(td_batch_texts[i])
            current_length += len(td_batch_texts[i])
            i += 1
        
        # Translate the batch and reinsert translated content
        placeholder = "__TD__"
        batch_text = placeholder.join(batch_texts)
        translated_batch = translate_func(batch_text)
        translated_lines = translated_batch.split(placeholder)

        for td_node, line in zip(batch_nodes, translated_lines):
            td_node.clear()
            frag = BeautifulSoup(line, "html.parser")
            for child in frag.contents:
                td_node.append(copy.deepcopy(child))


    text_nodes = []
    for node in soup.find_all(string=True, recursive=True):
        if not node.find_parent(["td", "th"]) and node.strip():
            text_nodes.append(node)

    idx = 0
    total = len(text_nodes)
    while idx < total:
        batch_nodes = []
        li_texts = []
        current_length = len("<ol></ol>")
        while idx < total:
            node_text = text_nodes[idx].strip()
            if len(node_text) > chunk_size:
                translated_text = split_text_recursive(node_text, chunk_size, translate_func)
                text_nodes[idx].replace_with(translated_text)
                idx += 1
                continue
            li_str = f"<li>{node_text}</li>"
            if current_length + len(li_str) > chunk_size:
                break
            batch_nodes.append(text_nodes[idx])
            li_texts.append(li_str)
            current_length += len(li_str)
            idx += 1
        if not batch_nodes:
            # if all individual nodes are longer than chunk_size, translate it alone
            node_text = text_nodes[idx - 1].strip()
            li_str = f"<li>{node_text}</li>"
            batch_nodes = [text_nodes[idx - 1]]
            li_texts = [li_str]

        if batch_nodes:
            batch_text = "<ol>" + "".join(li_texts) + "</ol>"
            translated = translate_func(batch_text)
            trans_soup = BeautifulSoup(translated, "html.parser")
            translated_lis = trans_soup.find_all("li")
            for orig_node, li_tag in zip(batch_nodes, translated_lis):
                orig_node.replace_with(li_tag.decode_contents())

    results.append(str(soup))


def split_original_texts(text):
    """
    Split the original text into chunks.
    """
    from bs4 import BeautifulSoup

    # find all html blocks and replace them with placeholders
    soup = BeautifulSoup(text, "html.parser")
    html_blocks = []
    html_placeholders = []
    placeholder_fmt = "<<HTML_BLOCK_{}>>"
    text_after_placeholder = ""

    index = 0
    for elem in soup.contents:
        if hasattr(elem, "name") and elem.name is not None:
            html_str = str(elem)
            placeholder = placeholder_fmt.format(index)
            html_blocks.append(html_str)
            html_placeholders.append(placeholder)
            text_after_placeholder += placeholder
            index += 1
        else:
            text_after_placeholder += str(elem)

    # split text into paragraphs
    splited_block = []
    splited_block = split_and_append_text(splited_block, text_after_placeholder)

    # replace placeholders with html blocks
    current_index = 0
    for idx, block in enumerate(splited_block):
        _, content = block
        while (
            current_index < len(html_placeholders)
            and html_placeholders[current_index] in content
        ):
            content = content.replace(
                html_placeholders[current_index], html_blocks[current_index]
            )
            current_index += 1
            splited_block[idx] = ("html", content)

    return splited_block


def split_and_append_text(result, text_content):
    """
    Split the text and append the result to the result list.

    Args:
        result (list): The current result list.
        text_content (str): The text content to be processed.

    Returns:
        list: The updated result list after processing the text content.
    """
    if text_content.strip():
        # match all code block interval
        code_pattern = re.compile(r"(```.*?\n.*?```|~~~.*?\n.*?~~~)", re.DOTALL)
        last_pos = 0
        for m in code_pattern.finditer(text_content):
            # process text before code block
            if m.start() > last_pos:
                non_code = text_content[last_pos : m.start()]
                paragraphs = re.split(r"\n{2,}", non_code)
                for p in paragraphs:
                    if p.strip():
                        result.append(("text", p.strip()))
            # process code block
            result.append(("code", m.group()))
            last_pos = m.end()
        # process remaining text
        if last_pos < len(text_content):
            non_code = text_content[last_pos:]
            paragraphs = re.split(r"\n{2,}", non_code)
            for p in paragraphs:
                if p.strip():
                    result.append(("text", p.strip()))
    return result
