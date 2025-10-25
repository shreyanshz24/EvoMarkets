from abc import ABC, abstractmethod
import random
COOPERATE = "COOPERATE"
DEFECT = "DEFECT"
payoffs = {
    COOPERATE: {COOPERATE: (3, 3), DEFECT: (0, 5)},
    DEFECT:    {COOPERATE: (5, 0), DEFECT: (1, 1)}
}
class Agent(ABC):
    def __init__(self, p_noise: float = 0.05):
      self.p_noise = p_noise

    def _perceive(self, true_move: str) -> str:
      if random.random() < self.p_noise:
        if true_move == COOPERATE:
          return DEFECT
        else:
          return COOPERATE
      return true_move
    @abstractmethod
    def play(self, opponent_history: list) -> str:
      pass
class Game:
    def __init__(self, agent_a: Agent, agent_b: Agent, rounds: int):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.rounds = rounds
        self.history = []

    def run(self):
        for _ in range(self.rounds):
            history_a = [move[0] for move in self.history]
            history_b = [move[1] for move in self.history]
            move_a = self.agent_a.play(opponent_history=history_b)
            move_b = self.agent_b.play(opponent_history=history_a)
            payoff_a, payoff_b = payoffs[move_a][move_b]
            self.history.append((move_a, move_b, payoff_a, payoff_b))
        print("Game finished. Details Below.")

class AlwaysCooperate(Agent):
  def play(self, opponent_history: list) -> str:
    return COOPERATE

class AlwaysDefect(Agent):
  def play(self, opponent_history: list) -> str:
    return DEFECT

class NoisyTitForTat(Agent):
    def __init__(self, p_noise: float = 0.05):
        super().__init__(p_noise=p_noise)

    def play(self, opponent_history: list) -> str:
        if not opponent_history:
            return COOPERATE
        else:
            true_last_move = opponent_history[-1]
            return self._perceive(true_last_move)

def play(self, opponent_history: list) -> str:
        if len(opponent_history) < 2:
            return COOPERATE
        else:
            perceived_last_move = self._perceive(opponent_history[-1])
            perceived_second_to_last_move = self._perceive(opponent_history[-2])
            if perceived_last_move == DEFECT and perceived_second_to_last_move == DEFECT:
                return DEFECT
            else:
                return COOPERATE
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

    def __init__(self, sus_agents):
      self.models = sus_agents
      self.beliefs = {}
      shareofagents =  1 / len(sus_agents)
      for agent in sus_agents:
        name = agent.__class__.__name__
        self.beliefs[name] = shareofagents
