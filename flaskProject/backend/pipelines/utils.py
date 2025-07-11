import re

def remove_word_repetitions(text: str) -> str:
    words = text.split()
    cleaned = [words[0]] if words else []
    for i in range(1, len(words)):
        if words[i] != words[i - 1]:
            cleaned.append(words[i])
    return " ".join(cleaned)

def structure_paragraphs(chunks: list[str], max_sentences_per_paragraph: int = 4) -> str:
    full_text = " ".join(chunks)
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
    paragraphs = [
        " ".join(sentences[i:i + max_sentences_per_paragraph])
        for i in range(0, len(sentences), max_sentences_per_paragraph)
    ]
    return "\n\n".join(paragraphs)

def clean_and_format_summary(text):
    text = re.sub(r"<s>|</s>", "", text)
    text = re.sub(r"\b\w\b", "", text)
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    formatted = "\n\n".join(["".join(s).capitalize() for s in sentences if len(s) > 20])
    return formatted
