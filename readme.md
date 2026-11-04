# AI Resume Screening

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange)

A Python tool that ranks PDF resumes against a job description using **TF-IDF vectorization** and **cosine similarity**.

> **Note:** This is a keyword-based ranking tool, not a trained machine-learning model. Scores reflect weighted vocabulary overlap with the job description, not candidate quality. It is a learning project and should not be used for real hiring decisions.

## Features

- Extracts text from multiple PDF resumes automatically (every `.pdf` in the chosen folder)
- Cleans text while preserving technical terms such as `c++` and `c#`
- Uses unigrams and bigrams so phrases like "machine learning" are captured
- Fits one shared TF-IDF vocabulary across the job description and all resumes, so scores are comparable
- Command-line options for the resume folder and job description file
- Prints a ranked list from best to worst match

## How It Works

```
 PDF resumes ──► text extraction ──► cleaning ─┐
                                               ├──► TF-IDF vectors ──► cosine similarity ──► ranked output
 job description (.txt) ───────────► cleaning ─┘
```

1. **Extract**: `pdfplumber` reads text from each page; pages are joined with newlines.
2. **Clean**: lowercase the text and replace punctuation with spaces (keeping `+` and `#`).
3. **Vectorize**: one `TfidfVectorizer` (English stop words removed, `ngram_range=(1, 2)`) is fit on the job description plus all resumes.
4. **Score**: cosine similarity between the job description vector and each resume vector.
5. **Rank**: sort resumes by score, descending.

**TF-IDF** weights a term by how often it appears in a document (TF) and how rare it is across all documents (IDF), so common words like "experience" count less and specific skills count more.

**Cosine similarity** measures the angle between two vectors: `cos(θ) = (A · B) / (‖A‖ ‖B‖)`. It ranges from 0 (no shared weighted terms) to 1 (identical direction).

## Tech Stack

| Purpose | Library |
|---|---|
| PDF text extraction | `pdfplumber` |
| TF-IDF and similarity | `scikit-learn` |
| Text cleaning, CLI | `re`, `argparse`, `glob` (standard library) |

## Project Structure

```
resume-screening/
├── resume_screening.py          # main script
├── requirements.txt
├── sample_job_description.txt   # sample job description
├── resumes_sample/              # 3 synthetic resumes for testing
│   ├── sample_resume_1.pdf
│   ├── sample_resume_2.pdf
│   └── sample_resume_3.pdf
├── .gitignore
└── README.md
```

Your own `resumes/` folder and `job_description.txt` are listed in `.gitignore`, so real candidate data is never committed.

## Installation

```bash
git clone https://github.com/Ardra-06/resume-screening.git
cd AI-Resume-Screening

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

**Try it with the included samples:**

```bash
python resume_screening.py --resumes resumes_sample --jd sample_job_description.txt
```

**Use your own data:**

1. Put the job description in `job_description.txt`.
2. Put PDF resumes in a `resumes/` folder.
3. Run with defaults:

```bash
python resume_screening.py
```

| Option | Default | Description |
|---|---|---|
| `--resumes` | `resumes` | Folder containing PDF resumes |
| `--jd` | `job_description.txt` | Job description text file |

## Sample Output

Running on the included synthetic resumes (a strong ML profile, a Java backend profile, and a graphic design profile):

```
Resume ranking (TF-IDF cosine similarity):

resumes_sample/sample_resume_1.pdf  ->  0.401
resumes_sample/sample_resume_2.pdf  ->  0.059
resumes_sample/sample_resume_3.pdf  ->  0.017
```

The ML-focused resume ranks first, the partially relevant one second, and the unrelated one last, as expected. A higher score means more overlap with the job description's weighted terms. Scores are only meaningful relative to each other for the same job description. A score of 0.40 does not mean "40% qualified."

## Design Decisions

- **Single shared vectorizer:** fitting TF-IDF separately per resume gives each its own vocabulary and makes scores incomparable.
- **Punctuation replaced with spaces, not deleted:** avoids merging words (`node.js` becomes `node js`) while keeping `c++` and `c#` intact.
- **Raw cosine score instead of a "match %":** a percentage implies an absolute meaning the score doesn't have.
- **Bigrams:** capture multi-word skills such as "machine learning" and "data structures".

## Limitations

- **No semantic understanding:** "ML" and "machine learning" are treated as different terms; synonyms are not matched.
- **Keyword overlap, not skill:** keyword repetition can inflate a resume's score.
- **Scanned PDFs:** image-only PDFs produce no text and score 0 (no OCR).
- **Layout-sensitive extraction:** multi-column or heavily designed resumes may extract in the wrong order.
- **No bias or fairness evaluation:** scoring on vocabulary can disadvantage candidates who describe the same skills differently.
- **Not formally evaluated:** the sample run above uses three hand-written resumes, so it is a sanity check, not an accuracy measurement. No precision or ranking-quality metric is reported.

## Privacy and Ethics

- No real candidate data is included in this repository; only synthetic samples are provided.
- Never commit real resumes. They contain personal information (names, phone numbers, emails, addresses).
- Automated screening can encode bias. Any real-world use should include human review and a documented fairness check.

## Future Work

- [ ] Sentence embeddings (`sentence-transformers`) to match synonyms
- [ ] Skill extraction and section-aware parsing (skills, education, experience)
- [ ] `.docx` resume support
- [ ] OCR fallback for scanned PDFs
- [ ] Labeled set of resumes to report precision@k or rank correlation
- [ ] Streamlit web interface
- [ ] CSV export of results

## Author

GitHub: [@Ardra-06](https://github.com/Ardra-06)