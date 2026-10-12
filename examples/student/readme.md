# Student School Model

## Summary
This model is based on School Enrollment Model ODD Protocol Catalina Canals, Enrique Canessa, Spiro Maroulis, Alejandra Mizala & Sergio Chaigneau July 8, 2024

This model simulates a school enrollment system using an Agent-Based Model (ABM), where students and schools interact within a competitive educational environment. The model captures how individual student decisions and institutional constraints lead to emergent enrollment patterns across different types of schools.

Students progress through grades over time, updating their academic achievement and deciding whether to remain in school, switch schools, or drop out. Schools compete for students by offering different levels of quality, selectivity, and tuition.

The model is inspired by real-world school choice systems, where enrollment distribution emerges from decentralized decision-making rather than centralized assignment.

## Technical Details

### Student Agent
The primary decision-making entity representing an individual learner.
- achievement → academic performance
- grade → current grade level (1–12)
- passed → pass/fail status
- state → enrolled / searching / dropout / graduated
- budget → maximum affordable tuition
- ses → socioeconomic status
- current_school → assigned school
- choice_set → list of considered schools
- social_network → connected peers

#### Behavior:
- Updates achievement based on past performance and school quality
- Competes with peers for promotion (ranking-based pass/fail)
- May drop out based on academic performance and grade
- Selects schools using probabilistic decision-making

### School Agent
Represents an educational institution with strategic behavior.

#### Internal State:
- capacity → maximum students per grade
- students → enrolled students
- tuition → cost of attendance
- sector → public / private / voucher
- value_added → contribution to student achievement
- mean_achievement → average performance of enrolled students
- selective → whether admission is merit-based

#### Behavior:
- Updates tuition dynamically
- Accepts students subject to capacity constraints
- Selects students either randomly or based on achievement

### Achievement Update
``` bash
achievement = (
            alpha_ach
            + beta_ach * achievement
            + school_effect
            + noise
        )
```
where:
``` bash
noise = random.normalvariate(0, sigma_ach)
```

## How to Run

If you have cloned the repo into your local machine, run ``pip install -e .`` from the project root. Then obtain an API key for an LLM provider of your choice and follow the steps below to configure this model. This model makes a large number of calls per minute, so a paid API key with higher rate limits is recommended.
1) Create a `.env` file in the project root.
2) In `.env`, set the API key variable that matches the provider prefix in `llm_model`. For example: ``OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx`` for `openai/...`, or ``GEMINI_API_KEY=your-gemini-api-key-here`` for `gemini/...`. The app uses `load_dotenv()` to load this automatically. If you use `ollama/...`, no API key is required, but you may need to configure `api_base` instead.
3) Update the ``llm_model`` attribute in `app.py` to a model you have access to. Use the format ``{provider}/{model_name}``, for example ``openai/gpt-4o-mini``.

Once you have configured `.env` and `llm_model`, run the following command from this directory:
``` solara run app.py
```

## Files

* ``model.py``: Core model code.
* ``agent.py``: Agent classes.
* ``app.py``: Sets up the interactive visualization.
* ``tools.py``: Tools for the llm-agents to use.

## More About
This model is based on (School Enrollment Model ODD Protocol Catalina Canals, Enrique Canessa, Spiro Maroulis, Alejandra Mizala & Sergio Chaigneau July 8, 2024)[https://www.comses.net/codebases/528f9fea-a3e2-4ad7-a8bc-9d9c77ac03ea/releases/1.0.0/]






