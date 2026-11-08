# Negotiation Model

**Disclaimer**: This is a toy model designed for illustrative purposes and is not based on a real research paper.

## Summary

This model is a simple negotiation model where two types of agents (buyer and seller) negotiate over a product. The objective is to test out which seller is able to capture most sales and therefore determine what influences customers the most: The Sales pitch, or the value for money.

**Seller A** : Has all the necessary skills for negotiation. It's comfortable with persuation, and is pretty committed to the job. It's pitching for Brand A and is aiming to sell running shoes and track suits. Although the seller has a more superior skill-set, Brand A products are slightly more over-prized for the same quality : $40 for Shoes and $50 for the Track Suit

**Seller B** : Is a bit more apathetic and is not particularly good at persuation or negotiation. It sells the same products as seller A, but is prized slightly cheaper. : $35 for Shoes and $47 for Track Suit

**Buyers** : there are mainly 2 types of buyers, buyers with a budget of $50 and buyers who have a budget of $100.

## Agent Decision Logic

Both buyers and sellers are implemented as LLM-powered agents. Their actions (such as moving, speaking, or buying) are selected by an action workflow that receives the agent’s internal state, local observations, and a set of available actions.
Sellers do not move; they use the `speak_to` action to pitch products to buyers in their cell or neighboring cells, attempting to persuade buyers until a sale is made or the buyer refuses.
Buyers can move using the `teleport_to_location` action if not engaged with a seller, gather information from sellers using `speak_to`, and make purchases using the `buy_product` action. Their decision is influenced by their budget and the information received from sellers.

## Agent Attributes

**Sellers**: Each seller has a set of internal attributes (e.g., persuasive, hardworking, lazy, unmotivated) and a sales counter.
**Buyers**: Each buyer has a budget ($50 or $100) and a list of purchased products.

## Negotiation Protocol

The negotiation is conducted through action-based interactions, where sellers initiate conversations and buyers respond, gather information, and decide on purchases.
The action workflow selects state-changing behavior based on prompts and observations, simulating realistic negotiation dynamics.

## Data Collection

The model tracks the number of sales for each seller using a data collector, allowing for analysis of which seller is more successful.

## How to Run

If you have cloned the repo into your local machine, run ``pip install -e .`` from the project root. Then obtain an API key for an LLM provider of your choice and follow the steps below to configure this model.
1) Create a `.env` file in the project root.
2) In `.env`, set the API key variable that matches the provider prefix in `llm_model`. For example: ``OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx`` for `openai/...`, ``GEMINI_API_KEY=your-gemini-api-key-here`` for `gemini/...`, or ``ANTHROPIC_API_KEY=your-anthropic-api-key-here`` for `anthropic/...`. The app uses `load_dotenv()` to load this automatically. If you use `ollama/...`, no API key is required, but you may need to configure `api_base` instead.
3) Update the ``llm_model`` attribute in `app.py` to a model you have access to. Use the format ``{provider}/{model_name}``, for example ``openai/gpt-4o-mini``.

Once you have configured `.env` and `llm_model`, run the following command from this directory:

```
    $ solara run app.py
```

## Files

* ``model.py``: Core model code.
* ``agent.py``: Agent classes.
* ``app.py``: Sets up the interactive visualization.
* ``actions.py``: Custom actions for the llm-agents to use.
