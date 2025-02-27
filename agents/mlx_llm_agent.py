from base.assoc import Assoc
from base.constants import Team
from base.spymaster import BaseSpymaster
from utils.helpers import isValid

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from openai import OpenAI
from random import shuffle


class MyAssoc(Assoc):
    def __init__(self):
        super().__init__()
        # Initialize your model/embedding here

    def getAssocs(self, pos, neg, topn):
        # Implement your word association logic
        pass

    def preprocess(self, w):
        # Implement any word preprocessing
        pass

from mlx_lm import load, generate

class MlxLLMSpymaster(BaseSpymaster):
    def __init__(self, assoc, model_name='Qwen/Qwen2-7B-Instruct-MLX'):
        super().__init__(assoc)
        self.device = self._get_device()
        self.model_name = model_name
        print(f"Using model: {self.model_name} on device: {self.device}")

        # Load model and tokenizer
        model, tokenizer = load('Qwen/Qwen2-7B-Instruct-MLX', tokenizer_config={"eos_token": "<|im_end|>"})
        
        self.model = model
        self.tokenizer = tokenizer
        print("model loaded")

        self.my_team_words = None
        self.assassin_word = None

        self.subset_size = 4
        self.subset = None


    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
        
    def _get_word_subset(self):
        """
        Select a subset of words from the team's words
        """
        # Make sure subset size isn't larger than available words
        actual_size = min(self.subset_size, len(self.my_team_words))
        
        subset = self.my_team_words[:actual_size]
        self.subset = subset
        
    def _get_init_prompt(self, board, team: Team):
        my_team_words = board['R'] if team == Team.RED else board['U']
        other_team_words = board['U'] if team == Team.RED else board['R']
        neutral_words = board['N']
        assassin_word = board['A']

        self.assassin_word = assassin_word

        words_to_avoid = other_team_words + neutral_words + assassin_word

        self.my_team_words = my_team_words
        # self._get_word_subset()
        # shuffle(self.subset)


        prompt = f'''
        You are playing Codenames as the Spymaster. Your goal is to help your team win by giving effective clues.

        RULES:
        - Provide exactly ONE word and ONE number
        - The number indicates how many team words relate to your clue
        - FORBIDDEN: Using any word (or variation) that appears on the board
        - FORBIDDEN: Proper nouns, hyphenated words, or compound words
        - FORBIDDEN: Using words that relate to the opposing team's words

        YOUR TEAM'S WORDS: {self.my_team_words}
        WORDS TO AVOID: {words_to_avoid}

        SCORING PRIORITIES:
        1. Link as many team words as possible with a single clue, without connecting to opponent words. (2-3 is a good target)
        2. Avoid any connection to words in WORDS TO AVOID
        3. Make connections that will be clear to human players

        REQUIRED OUTPUT FORMAT:
        Clue: <word> <number>
        '''

        return prompt
    
    def _get_critique_prompt(self, board, team: Team, clue):
        my_team_words = board['R'] if team == Team.RED else board['U']
        other_team_words = board['U'] if team == Team.RED else board['R']
        neutral_words = board['N']
        assassin_word = board['A']

        words_to_avoid = other_team_words + neutral_words + assassin_word

        # prompt = f'''
        # You are a master codebreaker. You need to critique the clue given by the spymaster to help your team guess the words associated with your team.\n
        # Here is the clue given by the spymaster: {clue}\n
        # Think critically and provide feedback on how the clue is associated with the words on the board.\n
        # You are only allowed to critique the clue given by the spymaster.\n
        # Here is the logic you should follow:\n
        # If you think the clue will possibly lead to the assassin word "{assassin_word}", you should say so.\n
        # If you think the clue is associated with any of the words for the other team here [{other_team_words}], say so.\n
        # If you think the clue is good, say PASS.\n
        # If you think the clue is bad, say FAIL.\n
        # Here are examples of how you should critique the clue:\n\n
        # '''


        prompt = f'''
        You are an expert Codenames judge evaluating the safety and effectiveness of Spymaster clues.
        The clue is a single word and a number of words that it relates to.

        CLUE TO EVALUATE: {clue}

        CRITICAL EVALUATION CRITERIA:
        1. ASSASSIN RISK: Check if clue could lead to assassin word "{assassin_word}"
        2. OPPONENT WORDS: Check for connections to opponent words: {other_team_words + neutral_words}
        3. CLARITY: Evaluate if the clue has clear, logical connections to my team's words: {self.my_team_words}
        4. EFFICIENCY: Assess if the number given matches reasonable word associations to my team's words

        REQUIRED OUTPUT FORMAT:
        1. Start with either "PASS" or "FAIL"
        2. Provide specific reasoning for your decision
        3. List any dangerous associations found
        4. Suggest improvements if applicable

        Example outputs:

        "FAIL
        - Dangerous association with assassin word 'POISON'
        - Could also connect to opponent's word 'MEDICINE'
        - Recommend choosing a different semantic field"

        "PASS
        - Clear connection to target words
        - No risky associations detected
        - Good numerical assessment"

        IMPORTANT: Always prioritize safety (avoiding assassin/opponent words) over clue effectiveness.
        '''

        return prompt
    
    def _get_second_prompt(self, board, team: Team, clue):
        my_team_words = board['R'] if team == Team.RED else board['U']
        other_team_words = board['U'] if team == Team.RED else board['R']
        neutral_words = board['N']
        assassin_word = board['A']

        words_to_avoid = other_team_words + neutral_words + assassin_word

        shuffle(my_team_words)

        prompt = f'''
        You are playing Codenames as the Spymaster. Your goal is to help your team win by giving effective clues.
        You have already given a clue: {clue}
        But a expert codebreaker has critiqued your clue and found it to be dangerous. You need to provide a new clue.\n

        RULES:
        - Provide exactly ONE word and ONE number
        - The number indicates how many team words relate to your clue
        - FORBIDDEN: Using any word (or variation) that appears on the board
        - FORBIDDEN: Proper nouns, hyphenated words, or compound words
        - FORBIDDEN: Using words that relate to the opposing team's words

        YOUR TEAM'S WORDS: {self.my_team_words}
        WORDS TO AVOID: {words_to_avoid}

        SCORING PRIORITIES:
        1. Link as many team words as possible with a single clue
        2. Avoid any connection to words in WORDS TO AVOID
        3. Make connections that will be clear to human players

        REQUIRED OUTPUT FORMAT:
        Clue: <word> <number>
        '''
        return prompt
    
    def _get_not_valid_prompt(self, board, team: Team, clue):
        my_team_words = board['R'] if team == Team.RED else board['U']
        other_team_words = board['U'] if team == Team.RED else board['R']
        neutral_words = board['N']
        assassin_word = board['A']

        words_to_avoid = other_team_words + neutral_words + assassin_word

        shuffle(my_team_words)

        prompt = f'''
        You are playing Codenames as the Spymaster. Your goal is to help your team win by giving effective clues.
        You have already given a clue: {clue}
        But the clue is not valid. You need to provide a new clue.\n

        RULES:
        - Provide exactly ONE word and ONE number
        - The number indicates how many team words relate to your clue
        - FORBIDDEN: Using any word (or variation) that appears on the board
        - FORBIDDEN: Proper nouns, hyphenated words, or compound words
        - FORBIDDEN: Using words that relate to the opposing team's words

        YOUR TEAM'S WORDS: {self.my_team_words}
        WORDS TO AVOID: {words_to_avoid}

        SCORING PRIORITIES:
        1. Link as many team words as possible with a single clue
        2. Avoid any connection to words in WORDS TO AVOID
        3. Make connections that will be clear to human players

        REQUIRED OUTPUT FORMAT:
        Clue: <word> <number>
        '''
        return prompt

    def _generate(self, prompt, max_tokens=50, temperature=0.8, stop=None):
        # mlx-lm generation
        messages = [
            {"role": "system", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            temperature=temperature,
        )
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=text, 
            verbose=True
        )
        return response

    def _parse_clue(self, clue_string):
        # Parse the clue string
        resp_arr = clue_string.split('\n')
        for line in resp_arr:
            if "Clue" in line:
                clue_parts = line.split()
                clue_word = clue_parts[1]
                number_of_words = clue_parts[2]
                break
        if not clue_word:
            raise ValueError("Clue word not found in response")
        if not number_of_words:
            raise ValueError("Number of words not found in response")
        
        return (clue_word, number_of_words)
    
    def _parse_critique(self, critique_string):
        # Parse the critique string
        resp_arr = critique_string.split('\n')
        for line in resp_arr:
            if "FAIL" in line:
                print(line)
                return False
        
        return True

    def makeClue(self, board, team: Team):
        """
        Generate a clue for your team.

        Args:
            board: Dictionary with keys 'R' (red), 'U' (blue), 'N' (neutral), 'A'
                (assassin)
            team: Team.RED or Team.BLUE indicating your team

        Returns:
            tuple: ((clue_word, number_of_words), debug_info)
        """
        print("Generating clue")
        clue_string = self._generate(self._get_init_prompt(board, team), temperature=0.8, max_tokens=25)
        clue_word, number_of_words = self._parse_clue(clue_string)
        clue = (clue_word, number_of_words)
        print(f"Clue generated")

        print("Generating critique")
        critique_string = self._generate(self._get_critique_prompt(board, team, clue), temperature=0.5, max_tokens=75)
        critique = self._parse_critique(critique_string)
        if not critique:
            print("Generating second clue")
            clue_string = self._generate(self._get_second_prompt(board, team, clue))
            clue_word, number_of_words = self._parse_clue(clue_string)
            clue = (clue_word, number_of_words)

        while not isValid(clue_word, set(board['R'] + board['U'] + board['N'] + board['A'])):
            print(f"Clue is not valid. Generating new clue")
            clue_string = self._generate(self._get_not_valid_prompt(board, team, clue), temperature=1.1)
            clue_word, number_of_words = self._parse_clue(clue_string)
            clue = (clue_word, number_of_words)

        return (clue, [f"Subset: {self.subset}", f"Assassin word: {self.assassin_word}"])
    

# # instantiate the spymaster for testing
# assoc = MyAssoc()
# spymaster = MlxLLMSpymaster(assoc)
# spymaster._generate(spymaster._get_init_prompt({'R': ['apple', 'banana'], 'U': ['carrot', 'dog'], 'N': ['elephant', 'frog'], 'A': ['gun']}, Team.RED))
# print(spymaster.makeClue({'R': ['apple', 'banana'], 'U': ['carrot', 'dog'], 'N': ['elephant', 'frog'], 'A': ['gun']}, Team.RED))