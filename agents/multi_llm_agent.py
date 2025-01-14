from base.assoc import Assoc
from base.constants import Team
from base.spymaster import BaseSpymaster
from utils.helpers import isValid

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from openai import OpenAI
from random import shuffle

from sklearn.metrics.pairwise import cosine_similarity
import spacy
import en_core_web_lg
import numpy as np


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

class MultiLLMSpymaster(BaseSpymaster):
    def __init__(self, assoc, base_url="http://localhost:8181/v1"):
        super().__init__(assoc)
        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy",  # API key is required but not used by llama.cpp
        )

        self.my_team_words = None
        self.words_to_avoid = None
        self.assassin_word = None

        self.subset_size = 6
        self.subset = None
        self.nlp = spacy.load('en_core_web_lg') # this is used for the word embeddings


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
    
    def _semantic_grouping(self):
        """Group words based on semantic similarity"""
        # Get word vectors
        docs = [self.nlp(word) for word in self.my_team_words]
        vectors = np.array([doc.vector for doc in docs])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(vectors)
        
        # Group similar words together
        grouped_indices = []
        remaining = set(range(len(self.my_team_words)))
        
        while remaining:
            current = remaining.pop()
            group = [current]
            
            # Find similar words
            for idx in list(remaining):
                if similarity_matrix[current][idx] > 0.3:  # Adjustable threshold
                    group.append(idx)
                    remaining.remove(idx)
            
            grouped_indices.extend(group)
        
        self.my_team_words = [self.my_team_words[i] for i in grouped_indices]
        
    def _get_init_prompt(self, board, team: Team):
        my_team_words = board['R'] if team == Team.RED else board['U']
        other_team_words = board['U'] if team == Team.RED else board['R']
        neutral_words = board['N']
        assassin_word = board['A']

        self.assassin_word = assassin_word

        words_to_avoid = other_team_words + neutral_words + assassin_word
        self.words_to_avoid = words_to_avoid

        self.my_team_words = my_team_words
        self._semantic_grouping()

        prompt = f'''
        You are playing Codenames as the Spymaster. Your goal is to help your team win by giving effective clues that connect multiple words.

        YOUR TEAM'S WORDS: {self.my_team_words}
        WORDS TO AVOID: {words_to_avoid}

        THINKING PROCESS (REQUIRED):
        1. First, examine ALL your team's words and identify potential thematic groups or relationships. Be creative!
        2. Look for words that share:
            - Category relationships (e.g., both are food items)
            - Function relationships (e.g., both are used for cleaning)
            - Conceptual relationships (e.g., both relate to speed)
        3. Prioritize groups of 2-3 words over single-word connections
        4. Double-check that your clue doesn't relate to any words in WORDS TO AVOID

        SCORING PRIORITIES:
        1. MAXIMIZE CONNECTIONS: Your primary goal is finding clues that link 2-4 team words
        2. CLARITY: Connections should be obvious to human players
        3. SAFETY: Avoid ANY possible connection to words in WORDS TO AVOID

        RULES:
        - Provide exactly ONE word and ONE number
        - The number indicates how many team words relate to your clue
        - FORBIDDEN: Using any word (or variation) that appears on the board
        - FORBIDDEN: Proper nouns, hyphenated words, or compound words
        - FORBIDDEN: Using words that relate to the opposing team's words

        REQUIRED OUTPUT FORMAT:
        Clue: <word> <number>
        Then, provide a brief explanation of your reasoning for the new clue.
        '''


        return prompt
    
    def _get_critique_prompt(self, board, team: Team, clue):
        my_team_words = board['R'] if team == Team.RED else board['U']
        other_team_words = board['U'] if team == Team.RED else board['R']
        neutral_words = board['N']
        assassin_word = board['A']

        words_to_avoid = other_team_words + neutral_words + assassin_word

        prompt = f'''
        You are an expert Codenames judge evaluating the safety and effectiveness of Spymaster clues.
        The clue is a single word and a number of words that it relates to.

        CLUE TO EVALUATE: {clue}

        CRITICAL EVALUATION CRITERIA:
        1. ASSASSIN RISK: Check if clue could lead to assassin word "{assassin_word}"
        2. OPPONENT WORDS: Check for connections to opponent words: {other_team_words}
        3. CLARITY: Evaluate if the clue has clear, logical connections to my team's words: {self.my_team_words}
        4. EFFICIENCY: Assess if the number given matches reasonable word associations to my team's words

        REQUIRED OUTPUT FORMAT:
        1. Start with either "PASS" or "FAIL"
        2. Provide specific reasoning for your decision
        3. List any dangerous associations found
        4. Suggest improvements if applicable

        Example outputs:

        FAIL
        - Dangerous association with assassin word '{assassin_word}'
        - Could also connect to opponent's word '{other_team_words[0]}'
        - Recommend choosing a different semantic field

        PASS
        - Clear connection to target words
        - No risky associations detected
        - Good numerical assessment

        IMPORTANT: Always prioritize safety (avoiding assassin/opponent words) over clue effectiveness, but do not make connections to the assassin word that are far stretched.
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
        You are playing Codenames as the Spymaster. Your goal is to help your team win by giving effective clues that connect multiple words.
        You have already given a clue: {clue}
        But a expert codebreaker has critiqued your clue and found it to be dangerous. You need to provide a new clue.\n

        YOUR TEAM'S WORDS: {self.my_team_words}
        WORDS TO AVOID: {words_to_avoid}

        THINKING PROCESS (REQUIRED):
        1. First, examine ALL your team's words and identify potential thematic groups or relationships
        2. Look for words that share:
            - Category relationships (e.g., both are food items)
            - Function relationships (e.g., both are used for cleaning)
            - Conceptual relationships (e.g., both relate to speed)
        3. Prioritize groups of 2-3 words over single-word connections
        4. Double-check that your clue doesn't relate to any words in WORDS TO AVOID

        SCORING PRIORITIES:
        1. MAXIMIZE CONNECTIONS: Your primary goal is finding clues that link 2-4 team words
        2. CLARITY: Connections should be obvious to human players
        3. SAFETY: Avoid ANY possible connection to words in WORDS TO AVOID

        RULES:
        - Provide exactly ONE word and ONE number
        - The number indicates how many team words relate to your clue
        - FORBIDDEN: Using any word (or variation) that appears on the board
        - FORBIDDEN: Proper nouns, hyphenated words, or compound words
        - FORBIDDEN: Using words that relate to the opposing team's words

        REQUIRED OUTPUT FORMAT:
        Clue: <word> <number>
        Then, provide a brief explanation of your reasoning for the new clue.
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
        You are playing Codenames as the Spymaster. Your goal is to help your team win by giving effective clues that connect multiple words.
        You have already given a clue: {clue}
        But the clue is not valid. You need to provide a new clue.\n

        YOUR TEAM'S WORDS: {self.my_team_words}
        WORDS TO AVOID: {words_to_avoid}

        THINKING PROCESS (REQUIRED):
        1. First, examine ALL your team's words and identify potential thematic groups or relationships
        2. Look for words that share:
            - Category relationships (e.g., both are food items)
            - Function relationships (e.g., both are used for cleaning)
            - Conceptual relationships (e.g., both relate to speed)
        3. Prioritize groups of 2-3 words over single-word connections
        4. Double-check that your clue doesn't relate to any words in WORDS TO AVOID

        SCORING PRIORITIES:
        1. MAXIMIZE CONNECTIONS: Your primary goal is finding clues that link 2-4 team words
        2. CLARITY: Connections should be obvious to human players
        3. SAFETY: Avoid ANY possible connection to words in WORDS TO AVOID

        RULES:
        - Provide exactly ONE word and ONE number
        - The number indicates how many team words relate to your clue
        - FORBIDDEN: Using any word (or variation) that appears on the board
        - FORBIDDEN: Proper nouns, hyphenated words, or compound words
        - FORBIDDEN: Using words that relate to the opposing team's words

        REQUIRED OUTPUT FORMAT:
        Clue: <word> <number>
        Then, provide a brief explanation of your reasoning for the new clue.
        '''
        return prompt
    
    def _get_final_prompt(self, board, team: Team, clue):

        prompt = f'''
        You are playing Codenames as the Spymaster. Your goal is to give a simple, clear clue for at least one of your team's words.

        RULES:
        - Provide exactly ONE word and ONE number
        - FORBIDDEN: Using any word (or variation) that appears on the board
        - FORBIDDEN: Proper nouns, hyphenated words, or compound words

        YOUR TEAM'S WORDS: {self.my_team_words[0]}
        WORDS TO AVOID: {self.words_to_avoid}

        INSTRUCTIONS:
        1. Pick the EASIEST word from YOUR TEAM'S WORDS to give a clue for
        2. Choose a simple, obvious association for that word
        3. Verify your clue doesn't connect to any WORDS TO AVOID
        4. Use "1" as your number since you're targeting one word

        REQUIRED OUTPUT FORMAT:
        Clue: <word> 1
        '''

    def _generate(self, prompt, max_tokens=50, temperature=0.8, stop=None):
        response = self.client.chat.completions.create(
            model="local_model",
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop
        )

        return response.choices[0].message.content

    def _parse_clue(self, clue_string):
        """Parse clue with enhanced error handling"""
        try:
            # Handle empty or None responses
            if not clue_string or clue_string.strip() == "":
                raise ValueError("Empty response received")

            # Look for "Clue:" in any case
            resp_arr = clue_string.split('\n')
            clue_word = None
            number_of_words = None

            for line in resp_arr:
                if "clue" in line.lower():
                    # Remove any punctuation and split
                    parts = line.replace(":", " ").split()
                    # Find index of "clue" and get following elements
                    try:
                        clue_index = [i for i, part in enumerate(parts) if "clue" in part.lower()][0]
                        if len(parts) > clue_index + 2:
                            clue_word = parts[clue_index + 1]
                            number_of_words = parts[clue_index + 2]
                    except IndexError:
                        continue

            if not clue_word or not number_of_words:
                raise ValueError(f"Invalid clue format: {clue_string}")

            # Validate number_of_words is numeric
            if not number_of_words.isdigit():
                raise ValueError(f"Invalid number format: {number_of_words}")

            # Basic word validation
            if not clue_word.isalpha():
                raise ValueError(f"Invalid word format: {clue_word}")

            return (clue_word.lower(), number_of_words)

        except Exception as e:
            print(f"Error parsing clue: {str(e)}")
            return None, None
    
    def _parse_critique(self, critique_string):
        """Parse critique with enhanced error handling"""
        try:
            if not critique_string or critique_string.strip() == "":
                print("Empty critique received")
                return False

            # Split into lines and look for PASS/FAIL
            resp_arr = critique_string.split('\n')
            found_decision = False

            for line in resp_arr:
                line = line.strip().upper()
                if "FAIL" in line:
                    return False
                if "PASS" in line:
                    found_decision = True
                    break

            if not found_decision:
                print("No clear PASS/FAIL found in critique")
                return False

            return True

        except Exception as e:
            print(f"Error parsing critique: {str(e)}")
            return False

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
        MAX_ATTEMPTS = 5
        attempts = 0
        
        while attempts < MAX_ATTEMPTS:
            try:
                print(f"Generating clue (attempt {attempts + 1}/{MAX_ATTEMPTS})")
                
                # Generate initial clue
                clue_string = self._generate(
                    self._get_init_prompt(board, team), 
                    temperature=0.5 + (attempts * 0.2),  # Increase temperature with each attempt
                    max_tokens=50
                )
                
                # Parse clue
                clue_word, number_of_words = self._parse_clue(clue_string)
                print(clue_word, number_of_words)
                if not clue_word or not number_of_words:
                    attempts += 1
                    continue

                clue = (clue_word, number_of_words)
                
                # Generate and check critique
                print("Generating critique")
                critique_string = self._generate(
                    self._get_critique_prompt(board, team, clue),
                    temperature=0.5,
                    max_tokens=75
                )

                print(critique_string)
                
                if not self._parse_critique(critique_string):
                    print("Failed critique, generating new clue")
                    clue_string = self._generate(
                        self._get_second_prompt(board, team, clue),
                        temperature=0.7 + (attempts * 0.1)
                    )
                    clue_word, number_of_words = self._parse_clue(clue_string)
                    if not clue_word or not number_of_words:
                        attempts += 1
                        continue
                    clue = (clue_word, number_of_words)
                
                # Validate clue
                if not isValid(clue_word, set(board['R'] + board['U'] + board['N'] + board['A'])):
                    print(f"Clue is not valid, generating new clue")
                    clue_string = self._generate(
                        self._get_not_valid_prompt(board, team, clue),
                        temperature=0.9
                    )
                    clue_word, number_of_words = self._parse_clue(clue_string)
                    if not clue_word or not number_of_words:
                        attempts += 1
                        continue
                    clue = (clue_word, number_of_words)
                    
                    # Final validation check
                    if not isValid(clue_word, set(board['R'] + board['U'] + board['N'] + board['A'])):
                        attempts += 1
                        continue
                
                # If we got here, we have a valid clue
                return (clue, [f"Words: {self.my_team_words}", f"Assassin word: {self.assassin_word}"])
                
            except Exception as e:
                print(f"Error during clue generation: {str(e)}")
                attempts += 1
                
        # If we've exhausted all attempts, return a safe fallback
        print("Failed to generate valid clue after maximum attempts")

        while not isValid(clue_word, set(board['R'] + board['U'] + board['N'] + board['A'])):
            clue_string = self._generate(
                self._get_final_prompt(board, team, clue),
                temperature=0.9
            )
            clue_word, number_of_words = self._parse_clue(clue_string)
            if not clue_word or not number_of_words:
                continue
            clue = (clue_word, number_of_words)

        return (clue, ["Failed to generate valid clue after maximum attempts, returning fallback"])
    