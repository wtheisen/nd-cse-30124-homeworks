# Shared cipher evidence

`cipher_note.txt` is the forensics lab's checked transcription of the note, still encoded with ROT13; it is never a training sample. HW08–10 generate their 2,000 training phrases in the notebook (NLTK `words`, seed 1), so the phrases are the same in all three. There are no validation or test splits: the note is the only text the networks read that they never trained on. This is a position-aligned ROT13 exercise, not a general cipher solver. Digits are preserved by ROT13. Physical finale details remain subject to instructor confirmation.
