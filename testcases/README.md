# Eval Testcases

Each YAML file in this directory is one eval topic. A topic contains multiple
single-turn questions and answer descriptions for `/api/eval`.

Suggested schema:

```yaml
topic: short-topic-name
description: What this file is trying to check.
defaults:
  sandbox: workspace-write
  autonomous: true
cases:
  - id: stable_case_id
    question: "User-facing prompt to send to /api/eval."
    requirements:
      gpu: H200
    answer_description: >
      What a good answer should contain. This is not an exact expected string;
      it is a rubric for later scoring.
```

Keep each `question` self-contained. Use `requirements` for hardware or
environment assumptions needed to run the case. Keep `answer_description`
focused on observable behavior: commands suggested or run, important facts,
required caveats, artifacts, and cases where the agent should refuse or ask for
authorization.
