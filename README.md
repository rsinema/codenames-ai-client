# Codenames AI Agent

This was a project for my Computational Creativity class. The goal was to create a agent that would act as the 'Spymaster' on your team, and would creatively generate clues based on the current gameboard.

We had two tournamnets in our class to see whose agent was the most creative, and we approximated 'creativity' with wins (i.e. the most creative agent will make better clues, and thus their team will win more). My agent got me to 3rd place in our class, and it seemed to be somewhat creative in generating clues.

## My Agent

My agent uses a openai API call, I initally used [`llama.cpp`](https://github.com/ggml-org/llama.cpp) to have a server running Llama3.1-8b locally to create the clues. One prompt has the LLM create a clue based on the game state (e.g. our team's words, the other team's, assassin word) which usually output a clue intended for 2-3 words. This clue was then critiqued, which uses a different prompt to see if the clue was effective and wasn't dangerous to the team (it wasn't related to the assassin word). This either passed or failed the clue, if it failed the critique output was used to generate a new clue, which was then critiqued again. Eventually a fall back prompt was created that was made to focus on one word, because I didn't want this iterating too long while we had our tournament in class.

I attempted to model how I play the game Codenames into a agentic flow. Looking at the board and generating potential clues, checking if those clues could lead to the assassin word or the other teams word, repeat until a clue is sufficiently effective and not dangerous.

The code for my agent can be found in [`/agents/multi_llm_agent.py`](https://github.com/rsinema/codenames-ai-client/blob/main/agents/multi_llm_agent.py)

## Augmentation Attempts

### Prompt Engineering

My inital prompts for the LLM weren't as structured as the prompts that are currently used in agent. The prompt structuring greatly improved the structure of the output from the LLM and also helped the clue quality improve as well.

### WordNet

I attempted to augment the information given to the LLM by using `nltk`'s wordnet to semantically group the words prior to feeding them into the model. This might have marginally improved the clues, but overall prompt engineering and model size were the biggest factors in generating effective clues.
