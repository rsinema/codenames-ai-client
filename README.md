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

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Set up Word2Vec AI
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

## Creating Your AI Agent
1. Create a new Python file for your AI agent
2. Import the base AI class:

```python
from codenames_client.ai import BaseAI

class MyCodenamesAI(BaseAI):
    def generate_clue(self, game_state):
        # Your AI logic here
        return clue, number
```

## Game State Format
The game state provided to your AI will be a dictionary containing:
```python
{
    "board": [
        {"word": "WORD1", "type": "RED"},
        {"word": "WORD2", "type": "BLUE"},
        # ... more words
    ],
    "team": "RED",  # Your team color
    "remaining_words": {
        "RED": 8,   # Number of red words left
        "BLUE": 7   # Number of blue words left
    }
}
```

### Word Types
- "RED": Red team's words 
- "BLUE": Blue team's words 
- "NEUTRAL": Neutral words 
- "ASSASSIN": Game-ending assassin word

## Example Implementation
```python
class SimpleAI(BaseAI):
    def generate_clue(self, game_state):
        # Find all friendly words
        friendly_words = [
            word["word"] for word in game_state["board"] 
            if word["type"] == game_state["team"]
        ]
        
        # Simple example: return first word with count 1
        return friendly_words[0].lower(), 1
```

## Testing Your AI
Run your AI against the game server:
```bash
python your_ai_file.py --host localhost --port 8765
```

## Development Guidelines
1. Your AI must implement the `generate_clue` method
2. Return format: `(clue: str, count: int)`
3. Clues must be single words (no spaces)
4. Numbers and proper nouns are not allowed

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

