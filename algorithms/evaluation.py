from __future__ import annotations

from typing import TYPE_CHECKING

from algorithms.utils import dijkstra, bfs_distance


if TYPE_CHECKING:
    from world.game_state import GameState

pos_visited = set()

def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    if state.is_win():
        return 1000
    
    if state.is_lose():
        return -1000
    
    pos_drone = state.get_drone_position()
    pos_hunters = state.get_hunter_positions()
    deliveries = state.get_pending_deliveries()

    layout = state.get_layout()
    score = state.get_score()

    value = 0

    #(a) BFS distance from drone to nearest delivery point (closer is better).
    # Uses actual path distance so walls and terrain are respected.'''
    if len(deliveries):
        
        dist_delivery = float('inf')
        for d in deliveries:
            if dijkstra(layout, pos_drone, d)[0]<dist_delivery:
                dist_delivery = dijkstra(layout, pos_drone, d)[0]
        
        value -=5 *dist_delivery 

    #(b) BFS distance from each hunter to the drone, traversing only normal terrain ('.' / ' ').  
    # Hunters blocked by mountains, fog, or storms are treated as unreachable (distance = inf) and pose no threat.]
    min_hunter = float('inf')
    for hunter in pos_hunters:
        dist_hunter = bfs_distance(layout, hunter, pos_drone, True)

        if dist_hunter == 0:
            return -1000
        if dist_hunter < min_hunter:
            min_hunter = dist_hunter
        if dist_hunter != float('inf'):
            value +=3 *dist_hunter
    
    #(c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
    if min_hunter != float('inf'):
        value +=5 *min_hunter

    #(d) Number of pending deliveries (fewer is better).
    value -=25 *len(deliveries)

    #(e) Current score (higher is better).
    value +=8 *score

    #(f) Delivery urgency: reward the drone for being close to a delivery it can reach strictly before 
    # any hunter, so it commits to nearby pickups rather than oscillating in place out of excessive hunter fear.     
    for delivery in deliveries:
        dist_drone = dijkstra(layout, pos_drone, delivery)[0]

        dist_hunters = []
        for hunter in pos_hunters:
            d_hunter = bfs_distance(layout, hunter, delivery, True)
            dist_hunters.append(d_hunter)
        
        if dist_drone < min(dist_hunters):
            value +=20
  
    #(g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.
    if pos_drone in pos_visited:
        value -=5

    pos_visited.add(pos_drone)


    value -=1
    #para asegurar value en rango (-1000, 1000)
    if value < -1000:
        value = -1000
    if value > 1000:
        value = 1000

    return value
