from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """
    def minimax(self, state, agent_index, max_depth, num_agents):
        
        if state.is_win() or state.is_lose() or max_depth==0:
            return self.evaluation_function(state)

        next_agent = (agent_index+1)%num_agents
        
        if next_agent == 0:
            max_depth -= 1

        if agent_index == 0:
            value = float('-inf')

            for action in state.get_legal_actions(agent_index):
                successor = state.generate_successor(agent_index, action)
                
                value = max(value, self.minimax(successor, next_agent, max_depth, num_agents))
            
            return value

        else:
            value = float('inf')

            for action in state.get_legal_actions(agent_index):
                successor = state.generate_successor(agent_index, action)
                
                value = min(value, self.minimax(successor, next_agent, max_depth, num_agents))
            
            return value


    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        num_agents = state.get_num_agents()

        best_action = None
        best_value = float('-inf')

        for action in state.get_legal_actions(0):
            succesor = state.generate_successor(0, action)

            value = self.minimax(succesor, 1, self.depth, num_agents)

            if value>best_value:
                best_value=value
                best_action=action

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """
    def alphabeta(self, state, agent_index, max_depth, num_agents, alpha, beta):
          # Igual que minimax pero con poda alfa-beta
        if state.is_win() or state.is_lose() or max_depth == 0:
            return self.evaluation_function(state)
 
        next_agent = (agent_index + 1) % num_agents
 
        if next_agent == 0:
            max_depth -= 1
 
        if agent_index == 0:
            value = float('-inf')
 
            for action in state.get_legal_actions(agent_index):
                successor = state.generate_successor(agent_index, action)
                value = max(value, self.alphabeta(successor, next_agent, max_depth, num_agents, alpha, beta))
                alpha = max(alpha, value)
                if value > beta:
                    break
 
            return value
 
        else:
            value = float('inf')
 
            for action in state.get_legal_actions(agent_index):
                successor = state.generate_successor(agent_index, action)
                value = min(value, self.alphabeta(successor, next_agent, max_depth, num_agents, alpha, beta))
                beta = min(beta, value)
                if value < alpha:
                    break
 
            return value
    
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        num_agents = state.get_num_agents()
 
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
 
        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            value = self.alphabeta(successor, 1, self.depth, num_agents, alpha, beta)
 
            if value > best_value:
                best_value = value
                best_action = action
 
            alpha = max(alpha, best_value)
 
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def expectimax(self, state, agent_index, max_depth, num_agents):
        if state.is_win() or state.is_lose() or max_depth == 0:
            return self.evaluation_function(state)

        legal_actions = state.get_legal_actions(agent_index)
        if not legal_actions:
            return self.evaluation_function(state)

        next_agent = (agent_index + 1) % num_agents
        next_depth = max_depth - 1 if next_agent == 0 else max_depth

        if agent_index == 0:
            value = float('-inf')
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                value = max(
                    value,
                    self.expectimax(successor, next_agent, next_depth, num_agents),
                )
            return value

        child_values = []
        for action in legal_actions:
            successor = state.generate_successor(agent_index, action)
            child_values.append(
                self.expectimax(successor, next_agent, next_depth, num_agents)
            )

        min_value = min(child_values)
        mean_value = sum(child_values) / len(child_values)
        return (1 - self.prob) * min_value + self.prob * mean_value

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """
        num_agents = state.get_num_agents()
        legal_actions = state.get_legal_actions(0)

        if not legal_actions:
            return None

        best_action = None
        best_value = float('-inf')

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            value = self.expectimax(successor, 1, self.depth, num_agents)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action
