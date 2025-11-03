import random
import numpy as np
import math
from collections import namedtuple, defaultdict
from abc import ABC, abstractmethod
from scipy.stats import norm
import time

# --- 1. GLOBAL DEFINITIONS ---
Action = namedtuple('Action', ['type', 'price_level', 'size'])

# --- 2. AGENT BASE CLASS ---
class Agent(ABC):
    def __init__(self, agent_id: str, latency: float = 0.0):
        self.agent_id = agent_id
        self.latency = latency
        self.portfolio = {'cash': 100000.0, 'shares': 0}

    @abstractmethod
    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        pass

    @abstractmethod
    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float:
        pass

    @abstractmethod
    def learn(self, reward: float, new_lob_state: dict):
        pass

# --- 3. THE "AGENT ZOO" (STRATEGIES) ---
class AlwaysCooperate(Agent):
    """A simple agent that always places a passive LIMIT order."""
    def __init__(self, agent_id, latency=0.0):
        super().__init__(agent_id, latency)
        self.prob_limit = 0.99
        self.price_mu = 0.0
        self.price_sigma = 1.0
        self.size_mu = 1.0
        self.size_sigma = 0.5

    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        return Action(type='LIMIT', price_level=0, size=1)

    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float:
        if observed_action.type == 'LIMIT': prob_type = self.prob_limit
        else: prob_type = 1.0 - self.prob_limit

        prob_price = 1.0
        if observed_action.type == 'LIMIT':
            prob_price = norm.pdf(
                observed_action.price_level,
                loc=self.price_mu,
                scale=self.price_sigma
            )

        prob_size = norm.pdf(
            observed_action.size,
            loc=self.size_mu,
            scale=self.size_sigma
        )
        return prob_type * prob_price * prob_size

    def learn(self, reward: float, new_lob_state: dict):
        pass


class AlwaysDefect(Agent):
    """A simple agent that always places an aggressive MARKET order."""
    def __init__(self, agent_id, latency=0.0):
        super().__init__(agent_id, latency)
        self.prob_limit = 0.01; self.price_mu = 0.0; self.price_sigma = 1.0;
        self.size_mu = 100.0; self.size_sigma = 5.0

    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        # Big market order to ensure execution
        return Action(type='MARKET', price_level=0, size=100)

    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float:
        if observed_action.type == 'LIMIT': prob_type = self.prob_limit
        else: prob_type = 1.0 - self.prob_limit
        prob_price = 1.0
        if observed_action.type == 'LIMIT':
            prob_price = norm.pdf(
                observed_action.price_level,
                loc=self.price_mu,
                scale=self.price_sigma
            )
        prob_size = norm.pdf(
            observed_action.size,
            loc=self.size_mu,
            scale=self.size_sigma
        )
        return prob_type * prob_price * prob_size

    def learn(self, reward: float, new_lob_state: dict):
        pass


class NoisyTitForTat(Agent):
    """A Tit-for-Tat agent with a conditional, noisy personality."""
    def __init__(self, agent_id: str, latency: float = 0.0):
        super().__init__(agent_id, latency)
        self.coop_profile = {
            'prob_limit': 0.99, 'price_mu': 0.0, 'price_sigma': 1.0,
            'size_mu': 1.0, 'size_sigma': 0.5
        }
        self.defect_profile = {
            'prob_limit': 0.01, 'price_mu': 0.0, 'price_sigma': 1.0,
            'size_mu': 1.0, 'size_sigma': 0.5
        }

    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        ideal_coop_action = Action(
            type='LIMIT',
            price_level=self.coop_profile['price_mu'],
            size=int(self.coop_profile['size_mu'])
        )
        ideal_defect_action = Action(
            type='MARKET',
            price_level=self.defect_profile['price_mu'],
            size=int(self.defect_profile['size_mu'])
        )

        if opponent_last_action is None:
            return ideal_coop_action
        elif opponent_last_action.type == 'LIMIT':
            return ideal_coop_action
        else:
            return ideal_defect_action

    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float:
        profile_to_use = self.coop_profile
        if opponent_last_action and opponent_last_action.type == 'MARKET':
            profile_to_use = self.defect_profile

        if observed_action.type == 'LIMIT': prob_type = profile_to_use['prob_limit']
        else: prob_type = 1.0 - profile_to_use['prob_limit']

        prob_price = 1.0
        if observed_action.type == 'LIMIT':
            prob_price = norm.pdf(
                observed_action.price_level,
                loc=profile_to_use['price_mu'],
                scale=profile_to_use['price_sigma']
            )

        prob_size = norm.pdf(
            observed_action.size,
            loc=profile_to_use['size_mu'],
            scale=profile_to_use['size_sigma']
        )
        return prob_type * prob_price * prob_size

    def learn(self, reward: float, new_lob_state: dict):
        pass


class GrudgerAgent(Agent):
    """Cooperates until the first defection, then defects forever."""
    def __init__(self, agent_id: str, latency: float = 0.0):
        super().__init__(agent_id, latency)
        self.has_been_betrayed = False

    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        if opponent_last_action is None: return Action(type='LIMIT', price_level=0, size=1)
        if self.has_been_betrayed: return Action(type='MARKET', price_level=0, size=1)
        if opponent_last_action.type == 'MARKET':
            self.has_been_betrayed = True
            return Action(type='MARKET', price_level=0, size=1)
        else: return Action(type='LIMIT', price_level=0, size=1)

    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float:
        ideal_action = self.play({}, opponent_last_action)
        return 1.0 if (observed_action.type == ideal_action.type and observed_action.price_level == ideal_action.price_level and observed_action.size == ideal_action.size) else 0.0

    def learn(self, reward: float, new_lob_state: dict):
        pass


class SneakyAgent(Agent):
    """Plays Tit-for-Tat but has a small random chance of "sneaky" defecting."""
    def __init__(self, agent_id: str, latency: float = 0.0, sneak_prob: float = 0.1):
        super().__init__(agent_id, latency)
        self.sneak_prob = sneak_prob

    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        planned_move = None
        if opponent_last_action is None: planned_move = Action(type='LIMIT', price_level=0, size=1)
        elif opponent_last_action.type == 'LIMIT': planned_move = Action(type='LIMIT', price_level=0, size=1)
        else: planned_move = Action(type='MARKET', price_level=0, size=1)

        if random.random() < self.sneak_prob and planned_move.type == 'LIMIT':
            planned_move = Action(type='MARKET', price_level=0, size=10)

        return planned_move

    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float:
        if (observed_action.type == 'MARKET' and
            observed_action.size == 10 and
            (opponent_last_action is None or opponent_last_action.type == 'LIMIT')):
            return self.sneak_prob
        return 0.5

    def learn(self, reward: float, new_lob_state: dict):
        pass


class RandomAgent(Agent):
    """Makes random moves. A baseline control group."""
    def __init__(self, agent_id: str, latency: float = 0.0):
        super().__init__(agent_id, latency)
        self.price_choices = [0, 1, 2]
        self.size_choices = [1, 5, 10, 100]
        self.type_choices = ['LIMIT', 'MARKET']

    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        chosen_type = random.choice(self.type_choices)
        chosen_price = random.choice(self.price_choices)
        chosen_size = random.choice(self.size_choices)
        return Action(type=chosen_type, price_level=chosen_price, size=chosen_size)

    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float:
        prob_type = 1 / len(self.type_choices)
        prob_price = 1 / len(self.price_choices)
        prob_size = 1 / len(self.size_choices)

        if (observed_action.type in self.type_choices and
            observed_action.price_level in self.price_choices and
            observed_action.size in self.size_choices):
            return prob_type * prob_price * prob_size
        else:
            return 0.0

    def learn(self, reward: float, new_lob_state: dict):
        pass

class LobAwareTFT(NoisyTitForTat):
    """An agent that overrides TFT logic if it detects a market imbalance."""
    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        state = self._get_current_state(lob_state)
        if state[1] == 'Ask_Heavy':
            return Action(type='MARKET', price_level=0, size=5)
        elif state[1] == 'Bid_Heavy':
            return Action(type='MARKET', price_level=0, size=5)
        else:
            return super().play(lob_state, opponent_last_action)
    def _get_current_state(self, lob_state: dict) -> tuple:
        return lob_state.get('discretized_state', ('Medium', 'Balanced'))

# --- 4. ADVANCED LEARNING AGENTS ---
class QLearningAgent(Agent):
    def __init__(self, agent_id: str, latency: float = 0.0, epsilon: float = 0.1, alpha: float = 0.1, gamma: float = 0.9):
        super().__init__(agent_id, latency)
        self.action_space = [
            Action(type='LIMIT', price_level=0, size=1),  # 0
            Action(type='LIMIT', price_level=2, size=1),  # 1
            Action(type='MARKET', price_level=0, size=1), # 2
            Action(type='MARKET', price_level=0, size=10),# 3
            Action(type='HOLD', price_level=0, size=0)   # 4
        ]
        self.num_actions = len(self.action_space)
        self.q_table = defaultdict(lambda: [0.0] * self.num_actions)
        self.epsilon = epsilon; self.alpha = alpha; self.gamma = gamma
        self.last_state = None; self.last_action_index = None

    def _get_current_state(self, lob_state: dict) -> tuple:
        return lob_state.get('discretized_state', ('Medium', 'Balanced'))

    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        current_state = self._get_current_state(lob_state)
        action_index = 0

        if random.random() < self.epsilon:
            action_index = random.randint(0, self.num_actions - 1)
        else:
            state_q_values = self.q_table[current_state]
            action_index = int(np.argmax(state_q_values))

        chosen_action = self.action_space[action_index]
        self.last_state = current_state
        self.last_action_index = action_index
        return chosen_action

    def learn(self, reward: float, new_lob_state: dict):
        if self.last_state is None or self.last_action_index is None:
            return
        new_state = self._get_current_state(new_lob_state)
        max_q_future = max(self.q_table[new_state]) if self.q_table[new_state] else 0.0
        old_q_value = self.q_table[self.last_state][self.last_action_index]
        learned_value = reward + self.gamma * max_q_future
        update = self.alpha * (learned_value - old_q_value)
        self.q_table[self.last_state][self.last_action_index] = old_q_value + update

    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float: return 0.0


class BayesianAgent(Agent):
    def __init__(self, agent_id: str, latency: float, sus_agents: list):
        super().__init__(agent_id, latency)
        self.models = sus_agents
        self.beliefs = {}
        shareofagents = 1 / (len(sus_agents) + 1e-9)
        for agent in sus_agents:
            name = agent.__class__.__name__
            self.beliefs[name] = shareofagents
        self.alternate_flag = False

    def play(self, lob_state: dict, opponent_last_action: Action = None) -> Action:
        most_likely_opponent = max(self.beliefs, key=self.beliefs.get) if self.beliefs else None

        if most_likely_opponent == 'AlwaysDefect': return Action(type='MARKET', price_level=0, size=1)
        elif most_likely_opponent == 'AlwaysCooperate': return Action(type='MARKET', price_level=0, size=10)
        elif most_likely_opponent == 'NoisyTitForTat': return Action(type='LIMIT', price_level=0, size=1)
        else:
            if opponent_last_action is None: return Action(type='LIMIT', price_level=0, size=1)
            elif opponent_last_action.type == 'LIMIT': return Action(type='LIMIT', price_level=0, size=1)
            else: return Action(type='MARKET', price_level=0, size=1)

    def update_beliefs(self, opponent_last_action: Action, observed_action: Action):
        new_beliefs = {}
        total_probability = 0.0
        for model in self.models:
            name = model.__class__.__name__
            prior = self.beliefs.get(name, 1e-9)
            likelihood = model.get_action_probability(opponent_last_action, observed_action)
            unnormalized_posterior = likelihood * prior
            new_beliefs[name] = unnormalized_posterior
            total_probability += unnormalized_posterior

        if total_probability > 0:
            for name in new_beliefs:
                new_beliefs[name] /= total_probability
        else:
            new_beliefs = dict(self.beliefs)

        self.beliefs = new_beliefs

    def learn(self, reward: float, new_lob_state: dict): pass
    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float: return 0.0


class ThompsonAgent(Agent):
    def __init__(self, agent_id: str, latency: float, sub_agents: list):
        super().__init__(agent_id, latency)
        if not sub_agents: raise ValueError("ThompsonAgent must be given a non-empty list of sub-agents.")
        self.sub_agents = sub_agents
        self.sum_of_rewards = {}
        self.n_plays = {}
        for agent in self.sub_agents:
            name = agent.__class__.__name__
            self.sum_of_rewards[name] = 1.0
            self.n_plays[name] = 1
        self.last_played_agent_name = None

    def play(self, lob_state: dict, opponent_last_action: Action) -> Action:
        best_agent = None; best_sample = -float('inf')
        for agent in self.sub_agents:
            name = agent.__class__.__name__
            mu = self.sum_of_rewards[name] / self.n_plays[name]
            sigma = 1.0 / (self.n_plays[name] ** 0.5)
            sample = np.random.normal(loc=mu, scale=sigma)
            if sample > best_sample:
                best_sample, best_agent = sample, agent
        self.last_played_agent_name = best_agent.__class__.__name__
        return best_agent.play(lob_state, opponent_last_action)

    def learn(self, reward: float, new_lob_state: dict):
        if self.last_played_agent_name:
            self.sum_of_rewards[self.last_played_agent_name] += reward
            self.n_plays[self.last_played_agent_name] += 1

    def get_action_probability(self, opponent_last_action: Action, observed_action: Action) -> float: return 0.0

# -----------------------------------------------------------------
# --- 5. THE "DIRECTOR" (SIMULATION ENGINE)
# -----------------------------------------------------------------

try:
    from fast_lob import LimitOrderBook, Trade
    print("Successfully imported C++ LOB engine (fast_lob).")
except ImportError:
    print("FATAL ERROR: C++ (fast_lob) module not found.")
    print("Please build the C++ module by running: pip install .")
    print("--- Using slow Python LOB fallback. ---")
    from collections import deque, namedtuple as _namedtuple
    Trade = _namedtuple("Trade", ["buyer_id", "seller_id", "price", "quantity"])
    class LimitOrderBook:
        def __init__(self):
            # asks: price -> deque of orders (lowest price first)
            # bids: price -> deque of orders (highest price first)
            self.asks = {}
            self.bids = {}
            self.orders = {}
            self.next_order_id = 0

        def get_best_bid(self) -> float:
            if not self.bids: return 0.0
            return max(self.bids.keys())

        def get_best_ask(self) -> float:
            if not self.asks: return 1000000000
            return min(self.asks.keys())

        def add_limit_order(self, is_buy_side, price, agent_id, quantity):
            order = {'side': 'buy' if is_buy_side else 'sell', 'price': price, 'agent_id': agent_id, 'quantity': quantity}
            new_order_id = self.next_order_id; order['order_id'] = new_order_id; self.next_order_id += 1; self.orders[new_order_id] = order
            if is_buy_side:
                self.bids.setdefault(price, []).append(order)
            else:
                self.asks.setdefault(price, []).append(order)

        def process_market_order(self, is_buy_side, quantity, agent_id):
            trades_made = []; shares_needed = quantity
            if is_buy_side:
                ask_prices = sorted(self.asks.keys())
                for price in ask_prices:
                    queue = self.asks.get(price, [])
                    while queue and shares_needed > 0:
                        best_order = queue[0]
                        trade_quantity = min(shares_needed, best_order['quantity'])
                        trades_made.append(Trade(buyer_id=agent_id, seller_id=best_order['agent_id'], price=price, quantity=trade_quantity))
                        shares_needed -= trade_quantity
                        best_order['quantity'] -= trade_quantity
                        if best_order['quantity'] == 0:
                            queue.pop(0)
                    if not queue:
                        del self.asks[price]
                    if shares_needed == 0:
                        break
            else:
                bid_prices = sorted(self.bids.keys(), reverse=True)
                for price in bid_prices:
                    queue = self.bids.get(price, [])
                    while queue and shares_needed > 0:
                        best_order = queue[0]
                        trade_quantity = min(shares_needed, best_order['quantity'])
                        trades_made.append(Trade(buyer_id=best_order['agent_id'], seller_id=agent_id, price=price, quantity=trade_quantity))
                        shares_needed -= trade_quantity
                        best_order['quantity'] -= trade_quantity
                        if best_order['quantity'] == 0:
                            queue.pop(0)
                    if not queue:
                        del self.bids[price]
                    if shares_needed == 0:
                        break
            return trades_made

        def cancel_order(self, order_id):
            pass

# MarketSimulator
class MarketSimulator:
    def __init__(self, agents: list, book: LimitOrderBook, tick_size: int = 1, start_price: int = 10000):
        self.agents = agents
        self.book = book
        self.tick_size = tick_size
        self.start_price = start_price
        self.current_time = 0
        self.agent_map = {agent.agent_id: agent for agent in self.agents}
        self.pnl_history = {agent.agent_id: [] for agent in self.agents}
        self._seeded = False

    def _get_lob_state(self) -> dict:
        best_bid = self.book.get_best_bid()
        best_ask = self.book.get_best_ask()

        if best_bid == 0: best_bid = self.start_price - self.tick_size
        if best_ask == 1000000000: best_ask = self.start_price + self.tick_size

        spread = best_ask - best_bid

        spread_bucket = 'Medium'
        if spread <= self.tick_size: spread_bucket = 'Tight'
        elif spread > self.tick_size * 5: spread_bucket = 'Wide'

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'discretized_state': (spread_bucket, 'Balanced')
        }

    def _seed_the_book(self):
        if self._seeded:
            return
        seed_price_level = 3
        seed_quantity = 500

        lob_state = self._get_lob_state()
        best_bid = lob_state['best_bid']
        best_ask = lob_state['best_ask']

        buy_price = best_bid - (seed_price_level * self.tick_size)
        self.book.add_limit_order(
            is_buy_side=True,
            price=int(buy_price),
            agent_id="MARKET_MAKER_BID",
            quantity=seed_quantity
        )

        sell_price = best_ask + (seed_price_level * self.tick_size)
        self.book.add_limit_order(
            is_buy_side=False,
            price=int(sell_price),
            agent_id="MARKET_MAKER_ASK",
            quantity=seed_quantity
        )
        mid = self.start_price
        self.book.add_limit_order(is_buy_side=True, price=int(mid - self.tick_size), agent_id="INIT_BID", quantity=50)
        self.book.add_limit_order(is_buy_side=False, price=int(mid + self.tick_size), agent_id="INIT_ASK", quantity=50)

        self._seeded = True

    def _translate_action_to_order(self, agent: Agent, action_obj: Action) -> tuple:
        chosen_side = 'buy' if (hash(agent.agent_id) + random.randint(0, 1)) % 2 == 0 else 'sell'
        agent_id = agent.agent_id
        quantity = max(1, int(action_obj.size))

        if action_obj.type == 'HOLD':
            return (None, 0, agent_id, 0)

        if action_obj.type == 'MARKET':
            return (chosen_side == 'buy', 0, agent_id, quantity)

        elif action_obj.type == 'LIMIT':
            lob_state = self._get_lob_state()
            price = 0

            if chosen_side == 'buy':
                price = int(lob_state['best_bid'] - (action_obj.price_level * self.tick_size))
            else:
                price = int(lob_state['best_ask'] + (action_obj.price_level * self.tick_size))
            price = max(1, price)

            return (chosen_side == 'buy', price, agent_id, quantity)

        return (None, 0, agent_id, 0)
    
    def _process_action(self, agent: Agent, action_obj: Action, new_lob_state: dict) -> tuple[float, list]:
        """
        Processes one agent's action, updates portfolios,
        and returns its P&L (reward) based on the BID-ASK SPREAD.
        """
        
        (is_buy_side, price, agent_id, quantity) = self._translate_action_to_order(agent, action_obj)
        
        trades_made = []
        active_reward = 0.0

        lob_state = self._get_lob_state()
        mid_price = (lob_state['best_bid'] + lob_state['best_ask']) / 2.0

        if action_obj.type == 'HOLD':
            pass
            
        elif action_obj.type == 'LIMIT':
            self.book.add_limit_order(is_buy_side, price, agent_id, quantity)
            active_reward = 0.0 
            
        elif action_obj.type == 'MARKET':
            trades_made = self.book.process_market_order(is_buy_side, quantity, agent_id)
            for trade in trades_made:
                if is_buy_side:
                    active_reward += (mid_price - trade.price) * trade.quantity
                else:
                    active_reward += (trade.price - mid_price) * trade.quantity
        
        for trade in trades_made:
            buyer = self.agent_map.get(trade.buyer_id)
            seller = self.agent_map.get(trade.seller_id)
            
            trade_value = trade.price * trade.quantity
            if buyer:
                buyer.portfolio['cash'] -= trade_value
                buyer.portfolio['shares'] += trade.quantity
            if seller:
                seller.portfolio['cash'] += trade_value
                seller.portfolio['shares'] -= trade.quantity

            if agent.agent_id == trade.buyer_id and seller and seller.agent_id != "MARKET_MAKER_ASK":
                passive_reward = (trade.price - mid_price) * trade.quantity
                if hasattr(seller, 'learn'):
                    seller.learn(passive_reward, new_lob_state) 
                self.pnl_history[seller.agent_id][-1] += passive_reward
            elif agent.agent_id == trade.seller_id and buyer and buyer.agent_id != "MARKET_MAKER_BID":
                passive_reward = (mid_price - trade.price) * trade.quantity
                if hasattr(buyer, 'learn'):
                    buyer.learn(passive_reward, new_lob_state)
                self.pnl_history[buyer.agent_id][-1] += passive_reward
        self.pnl_history[agent.agent_id][-1] += active_reward
        return active_reward, trades_made

    def run_simulation(self, num_rounds: int):
        if len(self.agents) != 2:
            print("Error: This simulator is designed for 1-vs-1.")
            return

        agent_a, agent_b = self.agents[0], self.agents[1]
        last_action_a, last_action_b = None, None
        cancel_fee_per_order = 0.1 

        print(f"Running simulation: {agent_a.agent_id} (Lat: {agent_a.latency}) vs. {agent_b.agent_id} (Lat: {agent_b.latency})")

        for i in range(num_rounds):
            self.current_time += 1
            self.pnl_history[agent_a.agent_id].append(0.0)
            self.pnl_history[agent_b.agent_id].append(0.0)
            self._seed_the_book()
            lob_state = self._get_lob_state()
            first_mover, second_mover = (agent_a, agent_b) if agent_a.latency < agent_b.latency else (agent_b, agent_a)
            last_action_first = last_action_a if first_mover == agent_a else last_action_b
            last_action_second = last_action_b if first_mover == agent_a else last_action_a
            new_lob_state_1 = self._get_lob_state()
            action1 = first_mover.play(lob_state, last_action_second)
            reward1, trades1 = self._process_action(first_mover, action1, new_lob_state_1)
            new_lob_state_2 = self._get_lob_state()
            action2 = second_mover.play(new_lob_state_2, last_action_first)
            reward2, trades2 = self._process_action(second_mover, action2, new_lob_state_2)
            new_lob_state = self._get_lob_state()
            
            reward_a = reward1 if first_mover == agent_a else reward2
            reward_b = reward2 if first_mover == agent_a else reward1
            action_a = action1 if first_mover == agent_a else action2
            action_b = action2 if first_mover == agent_a else action1

            agent_a.learn(reward_a, new_lob_state)
            agent_b.learn(reward_b, new_lob_state)
            
            if hasattr(agent_a, 'update_beliefs'):
                agent_a.update_beliefs(opponent_last_action=last_action_b, observed_action=action_b)
            if hasattr(agent_b, 'update_beliefs'):
                agent_b.update_beliefs(opponent_last_action=last_action_a, observed_action=action_a)

            last_action_a, last_action_b = action_a, action_b         
            fee_A = self.book.clear_agent_orders(agent_a.agent_id, cancel_fee_per_order)
            fee_B = self.book.clear_agent_orders(agent_b.agent_id, cancel_fee_per_order)
            self.pnl_history[agent_a.agent_id][-1] -= fee_A
            self.pnl_history[agent_b.agent_id][-1] -= fee_B
            
            agent_a.portfolio['cash'] -= fee_A
            agent_b.portfolio['cash'] -= fee_B
            
            if i % (num_rounds // 10 or 1) == 0:
                print(f"  Round {i}/{num_rounds}...")
        
        print("Simulation complete.")

# -----------------------------------------------------------------
# --- 6. RISK & ANALYSIS FUNCTIONS
# -----------------------------------------------------------------

def calculate_var(pnl_list: list, percentile: int = 5) -> float:
    if not pnl_list: return 0.0
    return np.percentile(np.array(pnl_list), percentile)

def calculate_cvar(pnl_list: list, var_boundary: float) -> float:
    if not pnl_list: return 0.0
    pnl_array = np.array(pnl_list)
    tail_losses = pnl_array[pnl_array < var_boundary]
    return float(tail_losses.mean()) if tail_losses.size > 0 else float(var_boundary)

# -----------------------------------------------------------------
# --- 7. MAIN SCRIPT (TO RUN THE EXPERIMENT)
# -----------------------------------------------------------------

if __name__ == "__main__":
    suspect_zoo = [
        NoisyTitForTat(agent_id="suspect_tft", latency=10),
        AlwaysDefect(agent_id="suspect_ad", latency=10),
        AlwaysCooperate(agent_id="suspect_ac", latency=10)
    ]
    thompson_portfolio = [
        NoisyTitForTat(agent_id="thompson_tft", latency=10),
        GrudgerAgent(agent_id="thompson_grudge", latency=10),
        AlwaysDefect(agent_id="thompson_ad", latency=10)
    ]
    agent_A = GrudgerAgent(agent_id="Grudger", latency=10.0)
    agent_B = ThompsonAgent(agent_id="Thompson", latency=5.0, sub_agents=thompson_portfolio)
    try:
        lob = LimitOrderBook()
    except Exception as e:
        print(f"Fatal Error: Could not initialize C++ LOB: {e}")
        print("Please ensure 'fast_lob' is built (pip install .) and imported.")
        exit()
    simulator = MarketSimulator(
        agents=[agent_A, agent_B],
        book=lob,
        tick_size=1,        # $0.01
        start_price=10000   # $100.00
    )
    start_time = time.time()
    simulator.run_simulation(num_rounds=2000)
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds.")
    print("\n--- Simulation Results ---")

    pnl_A = simulator.pnl_history[agent_A.agent_id]
    pnl_B = simulator.pnl_history[agent_B.agent_id]
    pnl_A_dollars = [p / 100.0 for p in pnl_A]
    pnl_B_dollars = [p / 100.0 for p in pnl_B]

    final_pnl_A = sum(pnl_A_dollars)
    final_pnl_B = sum(pnl_B_dollars)

    var_95_A = calculate_var(pnl_A_dollars, 5)
    cvar_95_A = calculate_cvar(pnl_A_dollars, var_95_A)

    var_95_B = calculate_var(pnl_B_dollars, 5)
    cvar_95_B = calculate_cvar(pnl_B_dollars, var_95_B)

    print(f"\n--- {agent_A.agent_id} (Latency: {agent_A.latency}ms) ---")
    print(f"  Total P&L: ${final_pnl_A:.2f}")
    print(f"  Avg. P&L / round: ${np.mean(pnl_A_dollars):.4f}")
    print(f"  VaR (95%): ${var_95_A:.4f} (95% of rounds, loss was no worse than this)")
    print(f"  CVaR (95%): ${cvar_95_A:.4f} (In the worst 5% of rounds, avg. loss was this)")

    print(f"\n--- {agent_B.agent_id} (Latency: {agent_B.latency}ms) ---")
    print(f"  Total P&L: ${final_pnl_B:.2f}")
    print(f"  Avg. P&L / round: ${np.mean(pnl_B_dollars):.4f}")
    print(f"  VaR (95%): ${var_95_B:.4f}")
    print(f"  CVaR (95%): ${cvar_95_B:.4f}")