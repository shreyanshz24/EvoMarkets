from collections import namedtuple
from abc import ABC, abstractmethod
import random

# Define Action namedtuple
Action = namedtuple("Action", ["type", "price_level", "size"])

COOPERATE = "COOPERATE"
DEFECT = "DEFECT"

payoffs = {
    COOPERATE: {COOPERATE: (3, 3), DEFECT: (0, 5)},
    DEFECT:    {COOPERATE: (5, 0), DEFECT: (1, 1)}
}


# ----------------------- Agent Base Class -----------------------
class Agent(ABC):
    def __init__(self, agent_id, latency):
        self.agent_id = agent_id
        self.latency = latency
        self.portfolio = {}

    def _perceive(self, true_move: str) -> str:
        if random.random() < getattr(self, 'p_noise', 0):
            if true_move == COOPERATE:
                return DEFECT
            else:
                return COOPERATE
        return true_move

    @abstractmethod
    def play(self, lob_state: dict = None, opponent_last_action: Action = None) -> Action:
        pass


# ----------------------- Game Class -----------------------
class Game:
    def __init__(self, agent_a: Agent, agent_b: Agent, rounds: int):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.rounds = rounds
        self.history = []

    def run(self):
        last_a = None
        last_b = None

        for _ in range(self.rounds):
            move_a = self.agent_a.play(lob_state={}, opponent_last_action=last_b)
            move_b = self.agent_b.play(lob_state={}, opponent_last_action=last_a)
            payoff_a, payoff_b = payoffs[COOPERATE][COOPERATE]  # placeholder payoff logic
            self.history.append((move_a, move_b, payoff_a, payoff_b))
            last_a, last_b = move_a, move_b

        print("Game finished. Details Below.")
        for record in self.history:
            print(record)


# ----------------------- Agent Strategies -----------------------
class AlwaysCooperate(Agent):
    def __init__(self, agent_id, latency):
        super().__init__(agent_id, latency)

    def play(self, lob_state: dict = None, opponent_last_action: Action = None) -> Action:
        return Action(type='LIMIT', price_level=0, size=1)


class AlwaysDefect(Agent):
    def __init__(self, agent_id, latency):
        super().__init__(agent_id, latency)

    def play(self, lob_state: dict = None, opponent_last_action: Action = None) -> Action:
        return Action(type='MARKET', price_level=0, size=100)


class TitForTat(Agent):
    def __init__(self, agent_id, latency):
        super().__init__(agent_id, latency)

    def play(self, lob_state: dict = None, opponent_last_action: Action = None) -> Action:
        if opponent_last_action is None:
            return Action(type='LIMIT', price_level=0, size=1)
        else:
            if opponent_last_action.type == 'LIMIT':
                return Action(type='LIMIT', price_level=0, size=1)
            else:
                return Action(type='MARKET', price_level=0, size=1)


class RandomAgent(Agent):
    def __init__(self, agent_id: str, latency: float = 0.0):
        super().__init__(agent_id, latency)

    def play(self, lob_state: dict = None, opponent_last_action: Action = None) -> Action:
        chosen_type = random.choice(['LIMIT', 'MARKET'])
        chosen_price = random.choice([0, 1, 2])
        chosen_size = random.choice([1, 5, 10, 100])
        return Action(type=chosen_type, price_level=chosen_price, size=chosen_size)


class GrudgerAgent(Agent):
    def __init__(self, agent_id: str, latency: float = 0.0):
        super().__init__(agent_id, latency)
        self.has_been_betrayed = False

    def play(self, lob_state: dict = None, opponent_last_action: Action = None) -> Action:
        if opponent_last_action is None:
            return Action(type='LIMIT', price_level=0, size=1)
        if self.has_been_betrayed:
            return Action(type='MARKET', price_level=0, size=1)
        if opponent_last_action.type == 'MARKET':
            self.has_been_betrayed = True
            return Action(type='MARKET', price_level=0, size=1)
        else:
            return Action(type='LIMIT', price_level=0, size=1)


class SneakyAgent(Agent):
    def __init__(self, agent_id: str, latency: float = 0.0, sneak_prob: float = 0.1):
        super().__init__(agent_id, latency)
        self.sneak_prob = sneak_prob

    def play(self, lob_state: dict = None, opponent_last_action: Action = None) -> Action:
        planned_move = None
        if opponent_last_action is None:
            planned_move = Action(type='LIMIT', price_level=0, size=1)
        elif opponent_last_action.type == 'LIMIT':
            planned_move = Action(type='LIMIT', price_level=0, size=1)
        else:
            planned_move = Action(type='MARKET', price_level=0, size=1)

        if random.random() < self.sneak_prob and planned_move.type == 'LIMIT':
            planned_move = Action(type='MARKET', price_level=0, size=10)

        return planned_move


# ----------------------- Population Class -----------------------
class Population:
    def __init__(self, agents: list):
        self.agents = {}
        self.proportions = {}
        num_agents = len(agents)
        initial_share = 1 / num_agents

        for agent in agents:
            name = agent.__class__.__name__
            self.agents[name] = agent
            self.proportions[name] = initial_share

    def _calculate_fitness(self, rounds_per_match: int = 100) -> dict:
        num_strategies = len(self.agents)
        total_scores = {name: 0 for name in self.agents.keys()}

        for name_a, agent_a in self.agents.items():
            for name_b, agent_b in self.agents.items():
                game = Game(agent_a, agent_b, rounds=rounds_per_match)
                game.run()
                score_a = sum([r[2] for r in game.history])
                total_scores[name_a] += score_a

        total_rounds_played = num_strategies * rounds_per_match
        fitness = {name: score / total_rounds_played for name, score in total_scores.items()}

        return fitness

    def evolve(self):
        fitness = self._calculate_fitness()
        avg_pop_fitness = sum(self.proportions[name] * fitness[name] for name in self.proportions)
        new_proportions = {}

        for name, old_proportion in self.proportions.items():
            strategy_fitness = fitness[name]
            if avg_pop_fitness == 0:
                new_proportions[name] = old_proportion
            else:
                new_proportions[name] = old_proportion * (strategy_fitness / avg_pop_fitness)

        self.proportions = new_proportions


# ----------------------- Limit Order Book -----------------------
class LimitOrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []
        self.orders = {}
        self.next_order_id = 0

    def add_limit_order(self, order: dict):
        new_order_id = self.next_order_id
        order['order_id'] = new_order_id
        self.next_order_id += 1
        self.orders[new_order_id] = order
        if order['side'] == 'buy':
            self.bids.append(order)
            self.bids.sort(key=lambda x: x['price'], reverse=True)
        else:
            self.asks.append(order)
            self.asks.sort(key=lambda x: x['price'])


# ----------------------- Bayesian Agent -----------------------

class BayesianAgent(Agent):
    def __init__(self, agent_id: str, latency: float, sus_agents: list):
        super().__init__(agent_id, latency)
        self.models = sus_agents
        self.beliefs = {}
        shareofagents = 1 / len(sus_agents)
        for agent in sus_agents:
            name = agent.__class__.__name__
            self.beliefs[name] = shareofagents
        self.alternate_flag = False 

    def play(self, lob_state: dict, opponent_last_action: Action) -> Action:
        most_likely_opponent = max(self.beliefs, key=self.beliefs.get)
        if most_likely_opponent == 'AlwaysDefect':
            return Action(type='MARKET', price_level=0, size=1)
            
        elif most_likely_opponent == 'AlwaysCooperate':
            return Action(type='MARKET', price_level=0, size=10)
            
        elif most_likely_opponent == 'TitForTat':
            return Action(type='LIMIT', price_level=0, size=1)
            
        elif most_likely_opponent == 'TitForTwoTats':
            if self.alternate_flag:
                self.alternate_flag = False
                return Action(type='LIMIT', price_level=0, size=1)
            else:
                self.alternate_flag = True
                return Action(type='MARKET', price_level=0, size=1) 
        else:
            if opponent_last_action is None:
                return Action(type='LIMIT', price_level=0, size=1)
            elif opponent_last_action.type == 'LIMIT':
                return Action(type='LIMIT', price_level=0, size=1)
            else:
                return Action(type='MARKET', price_level=0, size=1)

    
    def update_beliefs(self, observed_action: Action):
        pass
