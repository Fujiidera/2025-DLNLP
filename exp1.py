import nltk
import math
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import gutenberg
from collections import Counter

file_ids = gutenberg.fileids()
letter_entropies = []
word_entropies = []
all_letters = []
all_words = []

for file_id in file_ids:
    text = gutenberg.raw(file_id)
    letters = [ch.lower() for ch in text if ch.isalpha()]
    total_letters = len(letters)
    freq_letters = Counter(letters)
    if total_letters > 0:
        entropy_letters = -sum((count / total_letters) * math.log2(count / total_letters)
                               for count in freq_letters.values())
    else:
        entropy_letters = 0.0
    words = [word.lower() for word in gutenberg.words(file_id) if word.isalpha()]
    total_words = len(words)
    freq_words = Counter(words)
    if total_words > 0:
        entropy_words = -sum((count / total_words) * math.log2(count / total_words)
                             for count in freq_words.values())
    else:
        entropy_words = 0.0
    letter_entropies.append(entropy_letters)
    word_entropies.append(entropy_words)
    all_letters.extend(letters)
    all_words.extend(words)
    print(f"{file_id}：letter entropy = {entropy_letters:.4f}, word entropy = {entropy_words:.4f}")

avg_letter_entropy = sum(letter_entropies) / len(letter_entropies)
avg_word_entropy = sum(word_entropies) / len(word_entropies)
print("\nAverage entropy across all texts：")
print(f"Average letter entropy = {avg_letter_entropy:.4f}")
print(f"Average word entropy = {avg_word_entropy:.4f}")

freq_letters = Counter(all_letters)
freq_words = Counter(all_words)
sorted_letters = sorted(freq_letters.items(), key=lambda x: x[1], reverse=True)
sorted_words = sorted(freq_words.items(), key=lambda x: x[1], reverse=True)
ranks_letters = range(1, len(sorted_letters) + 1)
freq_values_letters = [item[1] for item in sorted_letters]
ranks_words = range(1, len(sorted_words) + 1)
freq_values_words = [item[1] for item in sorted_words]

plt.figure(figsize=(12, 7))
plt.plot(ranks_letters, freq_values_letters, marker='o', linestyle='-', label='Letter Frequency')
plt.plot(ranks_words, freq_values_words, marker='x', linestyle='-', label='Word Frequency')
plt.xscale('log')
plt.yscale('log')
plt.title("Long-Tail Distribution of Letter and Word Frequency in English Texts")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.show()
