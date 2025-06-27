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
    # Search forward
    for i in range(center, len(text)):
        if text[i] in ["\n", ".", "。", ";", "；", "!", "！", "?", "？"]:
            if i + 1 < len(text) and len(text[: i + 1]) <= chunk_size:
                return i + 1
    # Search backward
    for i in range(center, 0, -1):
        if text[i] in ["\n", ".", "。", ";", "；", "!", "！", "?", "？"]:
            if len(text[: i + 1]) <= chunk_size:
                return i + 1
    # If no suitable position is found, split directly
    return min(chunk_size, len(text))


def split_text_recursive(text, chunk_size, translate_func, results):
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
        results.append(translate_func(text))
    else:
        split_pos = _find_split_pos(text, chunk_size)
        left = text[:split_pos].strip()
        right = text[split_pos:].strip()
        if left:
            split_text_recursive(left, chunk_size, translate_func, results)
        if right:
            split_text_recursive(right, chunk_size, translate_func, results)


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

    translated_code_lines = []
    split_text_recursive(
        code_content, chunk_size, translate_func, translated_code_lines
    )

    # drop ``` or ~~~
    filtered_code_lines = [
        line
        for line in translated_code_lines
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

    soup = BeautifulSoup(html_block, "html.parser")

    # collect text nodes
    text_nodes = []
    for node in soup.find_all(string=True, recursive=True):
        text = node.strip()
        if text:
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
                # if node_text is too long, split it
                translated_lines = []
                split_text_recursive(
                    node_text, chunk_size, translate_func, translated_lines
                )
                # concatenate translated lines with \n
                text_nodes[idx].replace_with("\n".join(translated_lines))
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

    Args:
        text (str): The original text to be split.

    Returns:
        list: A list of strings representing the chunks of the original text.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "html.parser")
    result = []
    last_position = 0
    contents = soup.contents
    i = 0
    while i < len(contents):
        element = contents[i]
        str_element = str(element)
        if len(str_element) == 0:
            i += 1
            continue

        # find element in original text
        start = text.find(str_element, last_position)
        if start != -1:
            end = start + len(str_element)
            element_str = text[start:end]
        else:
            # if element is not a tag, try to find it in original text
            if hasattr(element, "name") and element.name is not None:
                tag = element.name
                pat = r"<{tag}.*?>.*?</{tag}>".format(tag=tag)
                re_html = re.compile(pat, re.DOTALL)
                match = re_html.search(text, last_position)
                if match:
                    start = match.start()
                    end = match.end()
                    element_str = text[start:end]
                else:
                    element_str = str_element
                    start = -1
                    end = -1
            else:
                element_str = str_element
                start = -1
                end = -1

        true_start = True
        if start > 0 and text[start - 1] != "\n":
            true_start = False

        # process previous text
        if start != -1 and last_position < start:
            text_content = text[last_position:start]
            result = split_and_append_text(result, text_content)

        if hasattr(element, "name") and element.name is not None:
            if (
                end < len(text)
                and end >= 0
                and (text[end] not in ["\n", " "] or element_str.endswith("\n"))
            ):
                next_block_pos = text.find("\n\n", end)
                if next_block_pos == -1:
                    mix_region_end = len(text)
                else:
                    mix_region_end = next_block_pos + 2

                j = i + 1
                while j < len(contents):
                    next_element_str = str(contents[j])
                    next_start = text.find(next_element_str, end)
                    if next_start == -1 or next_start >= mix_region_end:
                        break
                    j += 1
                if true_start:
                    # merge text and html
                    result.append(
                        ("text_with_html", text[start:mix_region_end].rstrip("\n"))
                    )
                else:
                    _, last_content = result[-1]
                    result.pop()
                    result.append(
                        (
                            "text_with_html",
                            last_content + text[start:mix_region_end].rstrip("\n"),
                        )
                    )
                last_position = mix_region_end
                i = j
            else:
                # pure HTML block
                if true_start:
                    result.append(("html", element_str))
                else:
                    _, last_content = result[-1]
                    result.pop()
                    result.append(("html", last_content + element_str))
                last_position = end
                i += 1
        else:
            # normal text
            result = split_and_append_text(result, element_str)
            last_position = end if end != -1 else last_position + len(element_str)
            i += 1

    # process remaining text
    if last_position < len(text):
        text_content = text[last_position:]
        result = split_and_append_text(result, text_content)

    return result


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
