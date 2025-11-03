import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import itertools
from collections import namedtuple, defaultdict
from abc import ABC, abstractmethod
from scipy.stats import norm
from main import (
    Agent,
    MarketSimulator,
    Action,
    NoisyTitForTat,
    GrudgerAgent,
    AlwaysDefect,
    AlwaysCooperate,
    BayesianAgent,
    QLearningAgent,
    ThompsonAgent,
    RandomAgent,
    SneakyAgent,
    LobAwareTFT
)
try:
    from fast_lob import LimitOrderBook
    print("Successfully imported C++ LOB engine (fast_lob).")
except ImportError:
    print("FATAL ERROR: C++ (fast_lob) module not found.")
    print("Please build the C++ module by running: pip install .")
    exit()


# -----------------------------------------------------------------
# --- 3. THE EVOLUTIONARY SIMULATOR CLASS
# -----------------------------------------------------------------

class EvolutionarySimulator:
    
    def __init__(self, agent_classes: list, initial_lob_class, tick_size: int, start_price: int):
        
        self.agent_classes = agent_classes
        self.lob_class = initial_lob_class
        self.tick_size = tick_size
        self.start_price = start_price
        
        self.population_proportions = {}
        num_agents = len(agent_classes)
        initial_share = 1.0 / (num_agents + 1e-9)
        
        for agent_class in agent_classes:
            name = agent_class.__name__
            self.population_proportions[name] = initial_share
            
        self.population_history = [self.population_proportions.copy()]
        
        self.agent_names = [cls.__name__ for cls in agent_classes]
        self.agent_class_map = {cls.__name__: cls for cls in agent_classes}
        base_agents = [cls for cls in agent_classes if cls not in [BayesianAgent, QLearningAgent, ThompsonAgent, LobAwareTFT]]
        
        self.suspect_zoo = [cls(agent_id=f"suspect_{cls.__name__}", latency=10) 
                            for cls in base_agents]
        
        self.thompson_portfolio = [cls(agent_id=f"thompson_{cls.__name__}", latency=10) for cls in base_agents]

    def _run_generation(self, rounds_per_matchup: int) -> dict:
        
        fitness_scores = {name: 0.0 for name in self.agent_names}
        all_matchups = list(itertools.combinations(self.agent_names, 2))
        
        for (name_A, name_B) in all_matchups:
            
            (pnl_A, _), (pnl_B, _) = self._run_single_matchup(name_A, name_B, rounds_per_matchup)
            
            proportion_B = self.population_proportions[name_B]
            fitness_scores[name_A] += pnl_A * proportion_B
            
            proportion_A = self.population_proportions[name_A]
            fitness_scores[name_B] += pnl_B * proportion_A
            
        return fitness_scores

    
    def _run_single_matchup(self, name_A: str, name_B: str, num_rounds: int) -> tuple:
        
        class_A = self.agent_class_map[name_A]
        class_B = self.agent_class_map[name_B]

        agent_A, agent_B = None, None

        if class_A == BayesianAgent:
            agent_A = BayesianAgent(agent_id="Agent_A", latency=7.0, sus_agents=self.suspect_zoo)
        elif class_A == ThompsonAgent:
            agent_A = ThompsonAgent(agent_id="Agent_A", latency=5.0, sub_agents=self.thompson_portfolio)
        elif class_A == QLearningAgent:
            agent_A = QLearningAgent(agent_id="Agent_A", latency=15.0)
        else: 
            agent_A = class_A(agent_id="Agent_A", latency=10.0)

        if class_B == BayesianAgent:
            agent_B = BayesianAgent(agent_id="Agent_B", latency=7.0, sus_agents=self.suspect_zoo)
        elif class_B == ThompsonAgent:
            agent_B = ThompsonAgent(agent_id="Agent_B", latency=5.0, sub_agents=self.thompson_portfolio)
        elif class_B == QLearningAgent:
            agent_B = QLearningAgent(agent_id="Agent_B", latency=15.0)
        else: 
            agent_B = class_B(agent_id="Agent_B", latency=10.0)
        
        lob = self.lob_class()
        simulator = MarketSimulator(
            agents=[agent_A, agent_B],
            book=lob,
            tick_size=self.tick_size,
            start_price=self.start_price
        )
        
        simulator.run_simulation(num_rounds=num_rounds)
        pnl_A = sum(simulator.pnl_history["Agent_A"]) / 100.0
        pnl_B = sum(simulator.pnl_history["Agent_B"]) / 100.0
        
        return (pnl_A, 0.0), (pnl_B, 0.0)


    def run(self, num_generations: int, rounds_per_matchup: int, mutation_rate: float = 0.01):
        print(f"--- Starting Evolutionary Simulation ---")
        print(f"Generations: {num_generations}, Rounds: {rounds_per_matchup}, Mutation: {mutation_rate*100}%")
        
        for gen in range(num_generations):
            fitness_map = self._run_generation(rounds_per_matchup)
            min_fitness = min(fitness_map.values())
            shifted_fitness_map = {name: (fitness - min_fitness + 0.1) for name, fitness in fitness_map.items()}
            avg_pop_fitness = sum(
                shifted_fitness_map[name] * self.population_proportions[name]
                for name in self.agent_names
            )
            new_proportions = {}
            for name in self.agent_names:
                shifted_fitness = shifted_fitness_map.get(name, 0.1)
                old_proportion = self.population_proportions[name]
                
                if avg_pop_fitness == 0:
                    new_proportions[name] = old_proportion
                else:
                    new_proportions[name] = old_proportion * (shifted_fitness / avg_pop_fitness)
            mutated_proportions = {}
            num_agents = len(self.agent_names)
            mutation_share_per_agent = mutation_rate / num_agents

            for name in self.agent_names:
                taxed_proportion = new_proportions.get(name, 0.0) * (1 - mutation_rate)
                mutated_proportions[name] = taxed_proportion + mutation_share_per_agent
            total = sum(mutated_proportions.values())
            for name in mutated_proportions:
                mutated_proportions[name] /= total
            self.population_proportions = mutated_proportions
            self.population_history.append(mutated_proportions.copy())
            
            if (gen + 1) % (num_generations // 10 or 1) == 0:
                print(f"Generation {gen+1}/{num_generations} complete.")
        
        print("Evolutionary simulation complete.")


# -----------------------------------------------------------------
# --- 4. MAIN SCRIPT TO RUN THE EVOLUTION
# -----------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    all_agent_classes = [
        AlwaysCooperate,
        AlwaysDefect,
        NoisyTitForTat,
        GrudgerAgent,
        RandomAgent,
        SneakyAgent,
        QLearningAgent,
        BayesianAgent,
        ThompsonAgent,
        LobAwareTFT
    ]
    evo_sim = EvolutionarySimulator(
        agent_classes=all_agent_classes,
        initial_lob_class=LimitOrderBook,
        tick_size=1,
        start_price=10000
    )
    NUM_GENERATIONS = 100
    ROUNDS_PER_MATCHUP = 50
    evo_sim.run(
        num_generations=NUM_GENERATIONS, 
        rounds_per_matchup=ROUNDS_PER_MATCHUP,
        mutation_rate=0.01 # 1% mutation
    )
    total_time = time.time() - start_time
    print(f"\n--- EVOLUTION COMPLETE (Total time: {total_time:.2f}s) ---")
    history_df = pd.DataFrame(evo_sim.population_history)
    print("Generating population plot...")
    plt.figure(figsize=(14, 8))
    plt.stackplot(
        history_df.index,
        [history_df[col] for col in history_df.columns], 
        labels=history_df.columns
    )
    plt.title("Evolution of Agent Population Over Time", fontsize=18)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Population Proportion (%)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("evolution_plot.png")
    print("Final evolution plot saved as 'evolution_plot.png'.")
    print("\n--- Final Population ---")
    final_pop = evo_sim.population_history[-1]
    sorted_pop = sorted(final_pop.items(), key=lambda item: item[1], reverse=True)
    for agent_name, proportion in sorted_pop:
        print(f"  {agent_name}: {proportion * 100:.2f}%")