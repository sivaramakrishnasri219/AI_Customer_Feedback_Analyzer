"""
clean_text.py
----------------
Utility functions for cleaning raw customer review text.
"""

import re


class TextCleaner:
    """
    Clean raw text before tokenization.
    """

    def clean(self, text: str) -> str:
        """
        Clean input text.

        Steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Remove numbers
        4. Remove extra spaces

        Args:
            text (str): Raw input text

        Returns:
            str: Cleaned text
        """

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text


if __name__ == "__main__":

    cleaner = TextCleaner()

    sample = "I LOVE this Phone!!! Camera is Amazing 123."

    cleaned = cleaner.clean(sample)

    print("Original : ", sample)
    print("Cleaned  : ", cleaned)