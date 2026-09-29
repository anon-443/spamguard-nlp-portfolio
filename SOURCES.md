# Sources and data provenance

## UCI SMS Spam Collection

- Official dataset record: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Legacy record URL: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- Direct UCI download: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
- UCI-reported size: 5,574 labeled English SMS messages; text file `SMSSpamCollection` (about 466.7 KB; compressed archive about 198.6 KB).
- Schema: one message per line, label (`ham` or `spam`) followed by raw text. The file is not chronologically sorted.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0). The project does not commit or redistribute the raw SMS corpus.
- Dataset citation: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
- Introductory paper listed by UCI: Almeida, T., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. ACM Symposium on Document Engineering. https://doi.org/10.1145/2034691.2034742

The training script will record the observed label counts and split sizes at runtime. Do not assume class counts from secondary mirrors; rely on the downloaded UCI file and report the exact counts measured by the run.


Local acquisition check (2026-09-29): the official archive downloaded successfully (203,415 bytes); the extracted TSV parsed as 5,574 messages: 4,827 ham and 747 spam. The project training script re-counts the file on every run rather than hard-coding these numbers.
