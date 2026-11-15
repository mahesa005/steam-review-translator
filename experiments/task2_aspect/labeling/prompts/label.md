# Task: label sentences

You will receive one game name and a batch of complete Steam reviews for that game. Every review
is followed by its sentences, each with an id such as `[1850475-3]`.

Label **every listed sentence** following the guideline above. Read the whole review first so
that references and sarcasm are understood, then label each sentence on its own content.

Return one entry per sentence id, in the order given, with no ids missing or added. Use the exact
aspect and polarity names from the guideline. Use an empty list when the sentence has no aspect
and no overall verdict.
