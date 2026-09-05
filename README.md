# IMEN266 Operations Research II — Course Materials (Fall 2026)

Course materials for POSTECH IMEN266 (Ross, *Introduction to Probability
Models*, Ch. 1–9). Each chapter ships in four layers:

| layer | where | how to use it |
|---|---|---|
| Slides | `chXX/slides/*.pdf` | the lecture deck; problem slides carry Colab badges and gray Companion pointers, and each concept / class-problem slide is followed by the worked derivation developed in class (gray numbers = results you verify in the notebook) |
| Class-problem notebooks | `chXX/notebooks/` | open in Colab. In class we work on the `_INCLASS` version: fill the `TODO`s and let the `check(...)` cells confirm your answer. The plain version is the completed reference for later study |
| Concept & Proof Companion | `chXX/companion/` | `*_gapfill.pdf` is the one to work through — it replaces note-taking in this course; `*_full.pdf` is the completed reference |
| Homework | `chXX/homework/` | `hwN.pdf` (Parts A–D) plus the `HWN.ipynb` starter for the simulation part. Not collected or graded — Part A solutions are posted about 10 days after release; some Part A problems may reappear on exams |

## Quick start

- **Zero installation.** Every notebook runs on Google Colab as-is; the course
  package `imen266` is installed automatically from this repository.
- **Read the AI policy first.** `docs/ai_usage_guide.pdf` defines what AI use
  is expected in this course, what must be declared (prompt logs), and what is
  out of bounds. A printed copy is in the week-1 materials.
- The 5-minute checkpoint quizzes are run in class (paper/PLMS) and are not
  distributed here.

## Layout

```
ch01-03/    Basic probability      (released)
ch04/       Markov chains          (released)
imen266/    course Python package  (auto-installed by the notebooks)
docs/       AI usage guide
```

Further chapters (CTMCs, renewal, queueing, reliability) are added here as the
semester progresses. This repository always holds the **current** version of
every file — when a fix is announced, re-download from here.

## Notes

- All PDFs are the official compiled versions. The LaTeX sources are maintained
  in a separate private repository and are not distributed.
- Found a typo or a bug in a notebook? Post it on PLMS or email the TAs
  (Junhui Park, junhui.park@lstlab.org; Jeesoo Baik, jeesoo.baik@lstlab.org).
