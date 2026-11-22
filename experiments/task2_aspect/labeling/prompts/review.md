# Task: review another annotator's labels

You will receive one game name, a batch of complete Steam reviews, and the labels another
annotator gave to every sentence. Check each sentence's labels against the guideline above.

For each sentence id, return:
- `ok` when the labels are correct, or acceptable under the guideline.
- `flag` when a label is wrong. Give the issue type, a short reason that cites the guideline
  rule or the sentence wording, and the labels you think are correct.

Issue types:
- `wrong_aspect`: a label uses the wrong aspect.
- `missing_aspect`: an aspect the sentence discusses has no label.
- `extra_aspect`: a label for an aspect the sentence does not discuss.
- `wrong_polarity`: the aspect is right, the polarity is not.
- `guideline`: the labels break a specific guideline rule (name it).

Flag real errors only. Do not flag a defensible judgment call just because you would have chosen
differently, and do not flag wording or style. Missing a real error is worse than not flagging a
correct label.

Finally give one verdict for the whole batch: `pass` if no sentence needs a change, otherwise
`needs_revision`, with a one-sentence note on any pattern you saw.
