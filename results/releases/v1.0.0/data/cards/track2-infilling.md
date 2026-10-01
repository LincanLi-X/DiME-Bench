# Track 2: Text Infilling

Logical datasets: WikiText-103, CNN/DailyMail, and arXiv abstracts.

Each normalized sample contains non-empty prefix, hidden middle, and suffix
segments together with whitespace-token lengths, domain, span-length bin, and
boundary-density label. The gold middle is never inserted into a model prompt.
The committed examples are synthetic fixtures, not evaluation samples.
