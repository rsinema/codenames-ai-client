# Codenames AI Client

A Python client for developing AI agents that play Codenames by providing clues based on the current game state.

## Quick Start
1. Clone this repository
```bash
git clone git@github.com:rmorain/codenames-ai-client.git
cd codenames-ai-client
```
2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  
```
If you prefer, you can set up a Conda environment. Python 3.11 is recommended.

3. Install the package in editable mode:
```bash
pip install -e .
```

## Using Word2Vec Agent
1. Download `GoogleNews-vectors-negative300.bin`
```bash
curl -L -o ~/Downloads/googlenewsvectorsnegative300.zip\
  https://www.kaggle.com/api/v1/datasets/download/leadbest/googlenewsvectorsnegative300
unzip ~/Downloads/googlenewsvectorsnegative300.zip
```
2. Move `GoogleNews-vectors-negative300.bin` to `/client` folder

You can delete `GoogleNews-vectors-negative300.bin.gz`

3. Run client with Word2Vec AI spymaster
From the project root directory
```bash
python client.py <GAME_ID> <TEAM>
```
TEAM must be either 'red' or 'blue'\
Example: `python client.py ABCD red`

# Creating Codenames AI Agents

This guide will help you create your own AI agent for playing Codenames.

## Getting Started

1. Create a new file in the `agents` directory (e.g., `my_agent.py`)
2. Import the required base classes:
```python
from base.assoc import Assoc
from base.constants import Team
from base.spymaster import BaseSpymaster
```

## Implementing Your Agent

You need to implement two classes:

### 1. Word Association Class

Inherit from `Assoc` and implement these methods:

```python
class MyAssoc(Assoc):
    def __init__(self):
        super().__init__()
        # Initialize your model/embeddings/data here
    
    def getAssocs(self, pos, neg, topn):
        """
        Find words associated with positive words but not negative words.
        
        Args:
            pos: List of words to associate with
            neg: List of words to avoid
            topn: Number of associations to return
            
        Returns:
            List of (word, score) tuples
        """
        pass

    def preprocess(self, w):
        """
        Preprocess words before looking up associations.
        
        Args:
            w: Input word
            
        Returns:
            Processed word
        """
        pass
```

### 2. Spymaster Class

Inherit from `BaseSpymaster` and implement the clue generation:

```python
class MySpymaster(BaseSpymaster):
    def __init__(self, assoc):
        super().__init__(assoc)
    
    def makeClue(self, board, team: Team):
        """
        Generate a clue for your team.
        
        Args:
            board: Dictionary with keys:
                'R': List of red team's words
                'U': List of blue team's words
                'N': List of neutral words
                'A': List of assassin words
                'team': Team.RED or Team.BLUE
            
        Returns:
            tuple: ((clue_word, number_of_words), debug_info)
        """
        pass
```

## Using Your Agent

Update `client.py` to use your agent:

```python
from agents.my_agent import MyAssoc, MySpymaster

def getAI():
    return MySpymaster(MyAssoc())
```

## Example Approaches

1. **Word Embeddings**: Use models like Word2Vec, GloVe, or BERT to find semantically similar words
2. **Language Models**: Use GPT or other LLMs to generate associations
3. **Knowledge Graphs**: Use ConceptNet or WordNet to find related concepts
4. **Custom Datasets**: Create your own word association database

## Tips

1. Use the `utils.helpers.isValid()` function to check if your clue is valid
2. Test your agent with different board configurations
3. Consider both positive and negative associations
4. Remember that clues must be:
   - Single English words
   - Not derivatives of board words
   - Not proper nouns
   - Not acronyms

See `agents/word2vec.py` for a complete example implementation using Word2Vec embeddings.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

