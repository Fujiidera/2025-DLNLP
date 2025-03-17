import os
import math
import jieba
import re
import matplotlib.pyplot as plt
from collections import Counter

def calc_entropy(tokens):
    """Calculate the entropy of a given list of tokens."""
    total = len(tokens)
    freq = Counter(tokens)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
    return entropy

root_dir = "/Users/limenghuan/Downloads/wiki_zh"

all_files = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.startswith("wiki_"):
            file_path = os.path.join(root, file)
            all_files.append(file_path)

total_word_entropy = 0
total_sentence_entropy = 0
file_count = 0
all_characters = []
all_words = []

for file_path in all_files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue
    words = list(jieba.cut(text))
    if not words:
        continue
    word_entropy = calc_entropy(words)
    sentences = re.split(r'[。！？]', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
    if not sentences:
        continue
    sentence_entropy = calc_entropy(sentences)

    all_characters.extend([ch for ch in text if ch.strip()])  # All non-whitespace characters
    all_words.extend(words)
    total_word_entropy += word_entropy
    total_sentence_entropy += sentence_entropy
    file_count += 1

    print(f"{file_path}: Word entropy = {word_entropy:.4f}, Sentence entropy = {sentence_entropy:.4f}")

if file_count > 0:
    avg_word_entropy = total_word_entropy / file_count
    avg_sentence_entropy = total_sentence_entropy / file_count
    print("\nAverage entropy across all files:")
    print(f"Average word entropy = {avg_word_entropy:.4f}")
    print(f"Average sentence entropy = {avg_sentence_entropy:.4f}")
else:
    print("No valid files found.")

char_freq = Counter(all_characters)
word_freq = Counter(all_words)
sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
ranks_chars = range(1, len(sorted_chars) + 1)
freq_values_chars = [item[1] for item in sorted_chars]
ranks_words = range(1, len(sorted_words) + 1)
freq_values_words = [item[1] for item in sorted_words]

plt.figure(figsize=(12, 7))
plt.plot(ranks_chars, freq_values_chars, marker='o', linestyle='-', label='Character Frequency')
plt.plot(ranks_words, freq_values_words, marker='x', linestyle='-', label='Word Frequency')
plt.xscale('log')
plt.yscale('log')
plt.title("Long-Tail Distribution of Character and Word Frequency in Chinese Texts")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.show()
