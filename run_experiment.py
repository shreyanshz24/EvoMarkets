import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from main import (
    MarketSimulator,
    NoisyTitForTat,
    GrudgerAgent,
    AlwaysDefect,
    RandomAgent,
    SneakyAgent,
    AlwaysCooperate,
    BayesianAgent,
    QLearningAgent,
    ThompsonAgent,
    calculate_var,
    calculate_cvar
)

try:
    from fast_lob import LimitOrderBook
    print("Successfully imported C++ LOB engine (fast_lob).")
except ImportError:
    print("C++ LOB not found. Using slow Python LOB fallback.")
    from main import LimitOrderBook

# --- 2. Seting up the Experiment Parameters ---
NUM_ROUNDS_PER_MATCH = 1000
NUM_RUNS_PER_MATCHUP = 20
TICK_SIZE = 1
START_PRICE = 10000

def run_matchup(agent_a_class, agent_b_class):
    """
    Runs a full N-run simulation for a single pair of agents.
    Returns the average P&L and CVaR for both agents from this matchup.
    """

    matchup_results_A = {'final_pnl': [], 'cvar': []}
    matchup_results_B = {'final_pnl': [], 'cvar': []}
    
    for _ in range(NUM_RUNS_PER_MATCHUP):
        suspect_zoo = [
            NoisyTitForTat(agent_id="suspect_tft", latency=10),
            AlwaysDefect(agent_id="suspect_ad", latency=10),
        ]
        thompson_portfolio = [
        NoisyTitForTat(agent_id="thompson_tft", latency=10),
        GrudgerAgent(agent_id="thompson_grudge", latency=10),
        AlwaysCooperate(agent_id="thompson_ac", latency=10) # <-- The new "good employee"
    ]

        if agent_a_class == BayesianAgent:
            agent_A = BayesianAgent(agent_id="Agent_A", latency=7.0, sus_agents=suspect_zoo)
        elif agent_a_class == ThompsonAgent:
            agent_A = ThompsonAgent(agent_id="Agent_A", latency=5.0, sub_agents=thompson_portfolio)
        elif agent_a_class == QLearningAgent:
            agent_A = QLearningAgent(agent_id="Agent_A", latency=15.0)
        else:
            agent_A = agent_a_class(agent_id="Agent_A", latency=10.0)

        if agent_b_class == BayesianAgent:
            agent_B = BayesianAgent(agent_id="Agent_B", latency=7.0, sus_agents=suspect_zoo)
        elif agent_b_class == ThompsonAgent:
            agent_B = ThompsonAgent(agent_id="Agent_B", latency=5.0, sub_agents=thompson_portfolio)
        elif agent_b_class == QLearningAgent:
            agent_B = QLearningAgent(agent_id="Agent_B", latency=15.0)
        else:
            agent_B = agent_b_class(agent_id="Agent_B", latency=10.0)

        lob = LimitOrderBook()
        simulator = MarketSimulator(
            agents=[agent_A, agent_B],
            book=lob,
            tick_size=TICK_SIZE,
            start_price=START_PRICE
        )


        simulator.run_simulation(num_rounds=NUM_ROUNDS_PER_MATCH)
        
        pnl_A = simulator.pnl_history[agent_A.agent_id]
        pnl_B = simulator.pnl_history[agent_B.agent_id]
        
        pnl_A_dollars = [p / 100.0 for p in pnl_A]
        pnl_B_dollars = [p / 100.0 for p in pnl_B]
        
        matchup_results_A['final_pnl'].append(sum(pnl_A_dollars))
        matchup_results_B['final_pnl'].append(sum(pnl_B_dollars))
        
        var_A = calculate_var(pnl_A_dollars, 5)
        matchup_results_A['cvar'].append(calculate_cvar(pnl_A_dollars, var_A))
        
        var_B = calculate_var(pnl_B_dollars, 5)
        matchup_results_B['cvar'].append(calculate_cvar(pnl_B_dollars, var_B))

    # --- Return the AVERAGE results for this whole matchup ---
    avg_pnl_A = np.mean(matchup_results_A['final_pnl'])
    avg_cvar_A = np.mean(matchup_results_A['cvar'])
    avg_pnl_B = np.mean(matchup_results_B['final_pnl'])
    avg_cvar_B = np.mean(matchup_results_B['cvar'])
    
    return (avg_pnl_A, avg_cvar_A), (avg_pnl_B, avg_cvar_B)


# --- 4. The Main Tournament Loop ---
if __name__ == "__main__":
    
    start_time = time.time()
    
    agent_classes = [
        AlwaysCooperate,
        AlwaysDefect,
        NoisyTitForTat,
        GrudgerAgent,
        RandomAgent,
        SneakyAgent,
        QLearningAgent,
        BayesianAgent,
        ThompsonAgent
    ]

    all_matchups = list(itertools.combinations(agent_classes, 2))
    tournament_results = {agent_class.__name__: {'pnl': [], 'cvar': []} for agent_class in agent_classes}

    print(f"--- Starting Round-Robin Tournament ---")
    print(f"Total Agents: {len(agent_classes)}")
    print(f"Total Matchups: {len(all_matchups)}")
    print(f"Runs per Matchup: {NUM_RUNS_PER_MATCHUP}")
    print(f"Rounds per Game: {NUM_ROUNDS_PER_MATCH}\n")

    for i, (class_A, class_B) in enumerate(all_matchups):
        name_A = class_A.__name__
        name_B = class_B.__name__
        
        print(f"--- Running Matchup {i+1}/{len(all_matchups)}: {name_A} vs. {name_B} ---")
        
        (pnl_A, cvar_A), (pnl_B, cvar_B) = run_matchup(class_A, class_B)
        tournament_results[name_A]['pnl'].append(pnl_A)
        tournament_results[name_A]['cvar'].append(cvar_A)
        tournament_results[name_B]['pnl'].append(pnl_B)
        tournament_results[name_B]['cvar'].append(cvar_B)
        
        print(f"  {name_A}: P&L ${pnl_A:.2f}, CVaR ${cvar_A:.2f}")
        print(f"  {name_B}: P&L ${pnl_B:.2f}, CVaR ${cvar_B:.2f}")

    total_time = time.time() - start_time
    print(f"\n--- TOURNAMENT COMPLETE (Total time: {total_time:.2f}s) ---")
    
    # --- 5. Final Analysis & Visualization ---
    
    plt.figure(figsize=(14, 10))
    
    for agent_name, results in tournament_results.items():
        if not results['pnl']:
            continue
            
        avg_pnl = np.mean(results['pnl'])
        avg_cvar = np.mean(results['cvar'])
        
        print(f"\n--- {agent_name} (Overall) ---")
        print(f"  Average Final P&L: ${avg_pnl:.2f}")
        print(f"  Average CVaR (Tail Risk): ${avg_cvar:.2f}")
        
        plt.scatter([avg_cvar], [avg_pnl], s=300, label=f"{agent_name}\n(Avg. CVaR: ${avg_cvar:.2f})", alpha=0.7)
    
    plt.title("Tournament Results: Profit vs. Tail Risk (All Agents)", fontsize=18)
    plt.xlabel("Average Tail Risk (95% CVaR) - (More negative is worse)", fontsize=14)
    plt.ylabel("Average Final P&L (Profit)", fontsize=14)
    
    plt.axvline(x=0, color='grey', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("tournament_profit_vs_risk_plot.png")
    
    print("\nFull tournament plot saved as 'tournament_profit_vs_risk_plot.png'.")