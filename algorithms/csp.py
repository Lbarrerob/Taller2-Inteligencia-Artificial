from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        # Si la asignación está completa, retornar la solución
        if csp.is_complete(assignment):
            return assignment
        
        # Seleccionar una variable no asignada
        unassigned = csp.get_unassigned_variables(assignment)
        if not unassigned:
            return None
        var = unassigned[0]
        
        # Probar cada valor en el dominio de la variable
        for value in csp.domains[var]:
            # Verificar si la asignación es consistente
            if csp.is_consistent(var, value, assignment):
                # Asignar el valor
                csp.assign(var, value, assignment)
                
                # Recursión
                result = backtrack(assignment)
                if result is not None:
                    return result
                
                # Backtrack: deshacer la asignación
                csp.unassign(var, assignment)
        
        return None
    
    return backtrack({})


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    def forward_check(var: str, value: str, assignment: dict[str, str]) -> dict[str, list[str]] | None:
        """
        Realiza forward checking después de asignar var=value.
        Retorna un diccionario con los valores eliminados de cada dominio,
        o None si algún dominio queda vacío.
        """
        removed: dict[str, list[str]] = {}
        
        # Para cada vecino no asignado
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue
            
            removed[neighbor] = []
            # Verificar cada valor en el dominio del vecino
            for val in list(csp.domains[neighbor]):
                # Si el valor no es consistente con la asignación actual
                if not csp.is_consistent(neighbor, val, assignment):
                    csp.domains[neighbor].remove(val)
                    removed[neighbor].append(val)
            
            # Si el dominio queda vacío, falla
            if not csp.domains[neighbor]:
                return None
        
        return removed
    
    def restore_domains(removed: dict[str, list[str]]) -> None:
        """Restaura los valores eliminados a los dominios."""
        for var, values in removed.items():
            csp.domains[var].extend(values)
    
    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        # Si la asignación está completa, retornar la solución
        if csp.is_complete(assignment):
            return assignment
        
        # Seleccionar una variable no asignada
        unassigned = csp.get_unassigned_variables(assignment)
        if not unassigned:
            return None
        var = unassigned[0]
        
        # Probar cada valor en el dominio de la variable
        for value in list(csp.domains[var]):
            # Verificar si la asignación es consistente
            if csp.is_consistent(var, value, assignment):
                # Asignar el valor
                csp.assign(var, value, assignment)
                
                # Forward checking
                removed = forward_check(var, value, assignment)
                
                if removed is not None:
                    # Recursión
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                    
                    # Restaurar dominios
                    restore_domains(removed)
                
                # Backtrack: deshacer la asignación
                csp.unassign(var, assignment)
        
        return None
    
    return backtrack({})


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    def values_compatible(xi: str, vi: str, xj: str, vj: str, assignment: dict[str, str]) -> bool:
        """
        Verifica si dos pares variable-valor son compatibles con las restricciones.
        """
        # Crear una asignación temporal con ambos valores
        temp_assignment = dict(assignment)
        temp_assignment[xi] = vi
        temp_assignment[xj] = vj
        
        # Verificar si ambas asignaciones son consistentes
        return csp.is_consistent(xi, vi, {xj: vj}) and csp.is_consistent(xj, vj, {xi: vi})
    
    def revise(xi: str, xj: str, assignment: dict[str, str]) -> bool:
        """
        Elimina valores del dominio de xi que no tienen soporte en el dominio de xj.
        Retorna True si se eliminó algún valor.
        """
        revised = False
        values_to_remove = []
        
        for vi in csp.domains[xi]:
            # Verificar si existe al menos un valor en el dominio de xj que sea compatible
            has_support = False
            for vj in csp.domains[xj]:
                if values_compatible(xi, vi, xj, vj, assignment):
                    has_support = True
                    break
            
            # Si no hay soporte, marcar para eliminar
            if not has_support:
                values_to_remove.append(vi)
                revised = True
        
        # Eliminar valores sin soporte
        for val in values_to_remove:
            csp.domains[xi].remove(val)
        
        return revised
    
    def ac3(assignment: dict[str, str], arcs: list[tuple[str, str]] | None = None) -> bool:
        """
        Algoritmo AC-3 para mantener consistencia de arco.
        Retorna False si algún dominio queda vacío.
        """
        # Si no se especifican arcos, usar todos los arcos
        if arcs is None:
            queue = []
            for xi in csp.variables:
                if xi not in assignment:
                    for xj in csp.get_neighbors(xi):
                        if xj not in assignment:
                            queue.append((xi, xj))
        else:
            queue = list(arcs)
        
        while queue:
            xi, xj = queue.pop(0)
            
            # Si alguna variable ya está asignada, saltar
            if xi in assignment or xj in assignment:
                continue
            
            if revise(xi, xj, assignment):
                # Si el dominio de xi queda vacío, falla
                if not csp.domains[xi]:
                    return False
                
                # Agregar arcos (xk, xi) para todos los vecinos xk de xi (excepto xj)
                for xk in csp.get_neighbors(xi):
                    if xk != xj and xk not in assignment:
                        queue.append((xk, xi))
        
        return True
    
    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        # Si la asignación está completa, retornar la solución
        if csp.is_complete(assignment):
            return assignment
        
        # Seleccionar una variable no asignada
        unassigned = csp.get_unassigned_variables(assignment)
        if not unassigned:
            return None
        var = unassigned[0]
        
        # Probar cada valor en el dominio de la variable
        for value in list(csp.domains[var]):
            # Verificar si la asignación es consistente
            if csp.is_consistent(var, value, assignment):
                # Guardar dominios actuales
                saved_domains = {v: list(csp.domains[v]) for v in csp.variables}
                
                # Asignar el valor
                csp.assign(var, value, assignment)
                
                # Crear arcos para AC-3: todos los arcos (neighbor, var_neighbor)
                arcs = []
                for neighbor in csp.get_neighbors(var):
                    if neighbor not in assignment:
                        for other in csp.get_neighbors(neighbor):
                            if other not in assignment:
                                arcs.append((neighbor, other))
                
                # Ejecutar AC-3
                if ac3(assignment, arcs):
                    # Recursión
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                
                # Backtrack: restaurar dominios y deshacer asignación
                csp.domains = saved_domains
                csp.unassign(var, assignment)
        
        return None
    
    # Ejecutar AC-3 inicial para reducir dominios globalmente
    saved_domains = {v: list(csp.domains[v]) for v in csp.variables}
    if not ac3({}):
        return None
    
    result = backtrack({})
    
    # Si no hay solución, restaurar dominios originales
    if result is None:
        csp.domains = saved_domains
    
    return result


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    def forward_check(var: str, assignment: dict[str, str]) -> dict[str, list[str]] | None:
        """
        Realiza forward checking después de asignar var=value.
        Retorna un diccionario con los valores eliminados de cada dominio,
        o None si algún dominio queda vacío.
        """
        removed: dict[str, list[str]] = {}
        
        # Para cada vecino no asignado
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue
            
            removed[neighbor] = []
            # Verificar cada valor en el dominio del vecino
            for val in list(csp.domains[neighbor]):
                # Si el valor no es consistente con la asignación actual
                if not csp.is_consistent(neighbor, val, assignment):
                    csp.domains[neighbor].remove(val)
                    removed[neighbor].append(val)
            
            # Si el dominio queda vacío, falla
            if not csp.domains[neighbor]:
                return None
        
        return removed
    
    def restore_domains(removed: dict[str, list[str]]) -> None:
        """Restaura los valores eliminados a los dominios."""
        for var, values in removed.items():
            csp.domains[var].extend(values)

    def backtrack(assignment:dict[str, str])->dict [str,str]|None:
        if csp.is_complete(assignment):
            return assignment
        var= min(
            csp.get_unassigned_variables(assignment),
            key= lambda vr: (
                sum(1 for val in csp.domains[vr] if csp.is_consistent(vr, val, assignment)),
                -sum(1 for n in csp.get_neighbors(vr) if n not in assignment)
            )
        )


        ordenados = sorted(
            csp.domains[var], 
            key= lambda val: csp.get_num_conflicts(var, val, assignment)
        )
        for valor in ordenados:
            if not csp.is_consistent(var, valor, assignment):
                continue
            csp.assign(var, valor, assignment)

            eliminar = forward_check(var, assignment)

            if eliminar is not None:
                resultado = backtrack(assignment)
                if resultado is not None:
                    return resultado
                restore_domains(eliminar)

            csp.unassign(var, assignment)

        return None
     
    return backtrack({})
