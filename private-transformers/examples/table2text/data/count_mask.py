#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 GPT2 分词器统计：
1) 文本总 token 数
2) 带尖括号（< 或 >）的 token 数
3) 占比（百分比）

用法示例：
    python count_brackets_gpt2.py your_file.txt
"""

import sys
from transformers import GPT2Tokenizer

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <file.txt>")
        sys.exit(1)

    filename = sys.argv[1]

    # 读入整个文件
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # 初始化 GPT2 分词器
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 对文本进行 GPT2 tokenize（得到子词序列）
    tokens = tokenizer.tokenize(text)
    total_tokens = len(tokens)

    # 统计带有尖括号的 token
    bracket_count = sum(1 for t in tokens if "<" in t or ">" in t)

    # 计算占比
    ratio = (bracket_count / total_tokens * 100) if total_tokens > 0 else 0.0

    print(f"Bracket Tokens: {bracket_count}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Percentage: {ratio:.2f}%")

if __name__ == "__main__":
    main()
