#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
We have two text files, line by line:
  1) marked.txt (with lines that contain <PROPN>, <OBJ>, etc.)
  2) original.txt (the original lines)

We want to produce a new file (output.txt) where each line i from original.txt
is aligned with the same line i from marked.txt. In that line, every
placeholder (<PROPN>, etc.) is replaced by the corresponding original word,
wrapped with angle brackets. The rest is left as the original word.

We keep line breaks the same as original.txt (if line counts match).
If line count mismatch, we process up to the min. The remainder is optionally appended.

Example:
 marked line i:    "= <PROPN> <PROPN> <PROPN> ="
 original line i:  "= Valkyria Chronicles III ="
 => output line i: "= <Valkyria> <Chronicles> <III> ="

Caveats:
  - If lines mismatch in number of words, we do naive partial alignment.
  - If file line counts differ, we only process up to min. The remainder from original can be appended.
"""

import sys

PLACEHOLDERS = {
    "<PROPN>", "<OBJ>", "<SUBJ>", "<PRON>", "<CARDINAL>", "<DATE>",
    "<EVENT>", "<ORG>", "<NORP>", "<ORDINAL>", "<MASK>"
}

def read_file_as_lines(path):
    """ 读取文件所有行（保留换行结构），返回一个 list[str]. """
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f]
    return lines

def main():
    if len(sys.argv) < 4:
        print(f"Usage: python {sys.argv[0]} marked.txt original.txt output.txt")
        sys.exit(1)

    file_marked   = sys.argv[1]
    file_original = sys.argv[2]
    file_out      = sys.argv[3]

    # 1) 逐行读 marked / original
    marked_lines   = read_file_as_lines(file_marked)
    original_lines = read_file_as_lines(file_original)

    # 2) 对齐行数
    len_m = len(marked_lines)
    len_o = len(original_lines)
    min_len = min(len_m, len_o)
    if len_m != len_o:
        print(f"[WARN] line count mismatch: marked={len_m}, original={len_o}. "
              f"Will only process up to line {min_len}.\n"
              "The remainder from original can be appended unmodified, or from marked.")

    out_lines = []

    # 3) 行对行处理
    for i in range(min_len):
        # 分别以空格拆分
        marked_words   = marked_lines[i].split()
        original_words = original_lines[i].split()

        # 如果 word 数不一样，也只对齐到 min
        min_word_len = min(len(marked_words), len(original_words))

        merged_words = []
        for w in range(min_word_len):
            mw = marked_words[w]
            ow = original_words[w]
            if mw in PLACEHOLDERS:
                merged_words.append(f"<{ow}>")
            else:
                merged_words.append(ow)

        # 如果 original 这一行有更多单词，就直接拼接
        if len(original_words) > min_word_len:
            merged_words.extend(original_words[min_word_len:])
        elif len(marked_words) > min_word_len:
            # 如果你想把多余的 marked placeholder扔掉也行；这里简单忽略
            pass

        # line_out => 用空格拼回
        line_out = " ".join(merged_words)
        out_lines.append(line_out)

    # 4) 如果 original_lines 比 marked_lines 多，就附加剩余行(可选)
    if len_o > min_len:
        out_lines.extend(original_lines[min_len:])
    # 或者如果你想附加 marked_lines[min_len:] 也可以，但通常不需要

    # 5) 写回 output
    with open(file_out, 'w', encoding='utf-8') as f_out:
        for line in out_lines:
            f_out.write(line + "\n")

    print(f"[INFO] Done. Wrote {len(out_lines)} lines to {file_out}")

if __name__ == "__main__":
    main()
