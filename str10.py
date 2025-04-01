import streamlit as st
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
import matplotlib.pyplot as plt
import pandas as pd
import io

# Function to plot routes
def plot_routes(initial_routes, final_routes, customer_locations):
    """Visualizes ACO and Gurobi-optimized routes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot ACO routes
    for v in initial_routes.keys():
        x_aco = [customer_locations[i][0] for i in initial_routes[v]]
        y_aco = [customer_locations[i][1] for i in initial_routes[v]]
        ax1.plot(x_aco, y_aco, marker='o', linestyle='-', label=f"Vehicle {v}")
        for i, txt in enumerate(initial_routes[v]):
            ax1.annotate(txt, (x_aco[i], y_aco[i]), xytext=(0, 5), textcoords="offset points")

    ax1.scatter(customer_locations[0][0], customer_locations[0][1], c='k', marker='s', s=100, label="Depot")
    ax1.set_title("Initial Routes (ACO)")
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.legend()
    ax1.grid(True)

    # Plot Gurobi routes
    for v in final_routes.keys():
        x_gurobi = [customer_locations[i][0] for i in final_routes[v]]
        y_gurobi = [customer_locations[i][1] for i in final_routes[v]]
        ax2.plot(x_gurobi, y_gurobi, marker='s', linestyle='--', label=f"Vehicle {v}")
        for i, txt in enumerate(final_routes[v]):
            ax2.annotate(txt, (x_gurobi[i], y_gurobi[i]), xytext=(0, 5), textcoords="offset points")

    ax2.scatter(customer_locations[0][0], customer_locations[0][1], c='k', marker='s', s=100, label="Depot")
    ax2.set_title("Optimized Routes (Gurobi)")
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig)

# Function to solve VRP with ACO and Gurobi
def solve_vrp(distance_matrix, num_vehicles, capacities, costs_per_km, demand_list, customer_locations):
    """
    Solves the Vehicle Routing Problem (VRP) using a hybrid ACO-Gurobi approach.
    Returns routes, total distances, and total costs for ACO and Gurobi.
    """
    try:
        # Validate inputs
        if not isinstance(distance_matrix, np.ndarray) or distance_matrix.ndim != 2:
            raise ValueError("Distance matrix must be a 2D numpy array.")
        if not isinstance(num_vehicles, int) or num_vehicles <= 0:
            raise ValueError("Number of vehicles must be a positive integer.")
        if not isinstance(capacities, list) or len(capacities) != num_vehicles:
            raise ValueError("Capacities must be a list with length equal to the number of vehicles.")
        if not isinstance(costs_per_km, list) or len(costs_per_km) != num_vehicles:
            raise ValueError("Costs per km must be a list with length equal to the number of vehicles.")
        if not isinstance(customer_locations, dict):
            raise ValueError("customer_locations must be a dictionary.")

        # Parse demand list
        demands = [0] + [int(d.strip()) for d in demand_list.split(',')]  # Include depot (0 demand)
        if len(demands) != len(distance_matrix):
            raise ValueError(f"Number of demands ({len(demands)}) must match the number of locations in the distance matrix ({len(distance_matrix)}).")

        C = list(range(len(distance_matrix)))  # Customers + depot
        V = list(range(1, num_vehicles + 1))  # Vehicle IDs

        # Ensure capacities and costs match
        Q = {v: capacities[v - 1] for v in V}
        cost_per_km = {v: costs_per_km[v - 1] for v in V}

        # Initialize pheromone matrix
        pheromone = np.ones((len(C), len(C)))

        # ACO Parameters
        n_ants = 30
        n_iterations = 300
        alpha = 1.0  # Pheromone importance
        beta = 3.0  # Distance importance
        evaporation_rate = 0.2

        def aco_vrp(pheromone):
            """ACO heuristic to generate initial solutions."""
            best_routes = {v: [0] for v in V}
            best_costs = {v: float("inf") for v in V}
            best_distances = {v: 0.0 for v in V}

            for _ in range(n_iterations):
                ant_solutions = {ant: {v: [0] for v in V} for ant in range(n_ants)}
                ant_costs = {ant: {v: 0 for v in V} for ant in range(n_ants)}
                ant_distances = {ant: {v: 0 for v in V} for ant in range(n_ants)}

                for ant in range(n_ants):
                    unvisited = set(C[1:])
                    remaining_capacities = Q.copy()
                    current_vehicle = 1

                    while unvisited and current_vehicle <= num_vehicles:
                        current_node = ant_solutions[ant][current_vehicle][-1]
                        probabilities = []
                        candidates = []

                        for j in unvisited:
                            if 0 <= j < len(distance_matrix) and 0 <= current_node < len(distance_matrix):
                                if demands[j] <= remaining_capacities[current_vehicle]:
                                    tau = max(pheromone[current_node, j], 1e-10) ** alpha
                                    eta = (1.0 / (distance_matrix[current_node, j] + 1e-6)) ** beta
                                    probabilities.append(tau * eta)
                                    candidates.append(j)

                        if not candidates:
                            if current_vehicle < num_vehicles:
                                current_vehicle += 1
                                if current_vehicle in ant_solutions[ant]:
                                    ant_solutions[ant][current_vehicle].append(0)
                            else:
                                break
                            continue

                        if not probabilities:
                            break

                        probabilities = np.array(probabilities)
                        sum_prob = sum(probabilities)
                        if sum_prob == 0:
                            if candidates:
                                probabilities = np.ones(len(candidates)) / len(candidates)
                            else:
                                break
                        else:
                            probabilities = probabilities / sum_prob

                        try:
                            next_node = candidates[np.random.choice(len(candidates), p=probabilities)]
                        except ValueError:
                            if candidates:
                                next_node = candidates[0]
                            else:
                                break

                        ant_solutions[ant][current_vehicle].append(next_node)
                        unvisited.remove(next_node)
                        remaining_capacities[current_vehicle] -= demands[next_node]
                        distance = distance_matrix[current_node, next_node]
                        ant_distances[ant][current_vehicle] += distance
                        ant_costs[ant][current_vehicle] += cost_per_km[current_vehicle] * distance

                    # Return to depot
                    for v in V:
                        if len(ant_solutions[ant][v]) > 1:
                            ant_solutions[ant][v].append(0)
                            last_node = ant_solutions[ant][v][-2]
                            if 0 <= last_node < len(distance_matrix):
                                distance = distance_matrix[last_node, 0]
                                ant_distances[ant][v] += distance
                                ant_costs[ant][v] += cost_per_km[v] * distance

                # Update pheromones and best routes
                pheromone_new = pheromone.copy()
                pheromone_new *= (1 - evaporation_rate)
                for ant in range(n_ants):
                    for v in V:
                        route = ant_solutions[ant][v]
                        if len(route) > 1:
                            route_cost = ant_costs[ant][v]
                            for i in range(len(route) - 1):
                                if 0 <= route[i] < len(pheromone_new) and 0 <= route[i + 1] < len(pheromone_new):
                                    pheromone_new[route[i], route[i + 1]] += 1 / (route_cost + 1e-10)
                            if route_cost < best_costs[v]:
                                best_routes[v] = route.copy()
                                best_costs[v] = route_cost
                                best_distances[v] = ant_distances[ant][v]

            return pheromone_new, best_routes, best_costs, best_distances

        def solve_with_gurobi(initial_routes):
            """Gurobi optimization to refine ACO-generated routes."""
            try:
                # Define WLS credentials as environment parameters
                wls_params = {
                    "WLSACCESSID": "ee4f4d29-6bb9-48cd-b563-a67106b53e85",
                    "WLSSECRET": "098633ca-00aa-4763-bd76-2fa207eff5d3",
                    "LICENSEID": 2634762
                }
                # Create Gurobi environment with WLS credentials
                with gp.Env(params=wls_params) as env:
                    model = gp.Model("VRP", env=env)
                    x = model.addVars(len(C), len(C), num_vehicles, vtype=GRB.BINARY, name="x")
                    u = model.addVars(len(C), num_vehicles, vtype=GRB.CONTINUOUS, name="u")

                    # Objective: Minimize total cost
                    model.setObjective(
                        gp.quicksum(
                            cost_per_km[v] * distance_matrix[i, j] * x[i, j, v - 1]
                            for i in C for j in C for v in V if i != j
                        ),
                        GRB.MINIMIZE,
                    )

                    # Constraints
                    for j in C[1:]:
                        model.addConstr(gp.quicksum(x[i, j, v - 1] for i in C for v in V if i != j) == 1)
                    for i in C:
                        for v in V:
                            model.addConstr(
                                gp.quicksum(x[i, j, v - 1] for j in C if i != j) ==
                                gp.quicksum(x[j, i, v - 1] for j in C if i != j)
                            )
                    for v in V:
                        model.addConstr(
                            gp.quicksum(demands[j] * x[i, j, v - 1] for i in C for j in C if i != j) <= Q[v]
                        )
                    for v in V:
                        model.addConstr(gp.quicksum(x[0, j, v - 1] for j in C[1:]) <= 1)
                        model.addConstr(gp.quicksum(x[j, 0, v - 1] for j in C[1:]) <= 1)

                    # Subtour elimination
                    for v in V:
                        for i in C[1:]:
                            for j in C[1:]:
                                if i != j:
                                    model.addConstr(u[i, v - 1] - u[j, v - 1] + len(C) * x[i, j, v - 1] <= len(C) - 1)

                    # Set initial solution from ACO
                    for v in V:
                        route = initial_routes[v]
                        for i in range(len(route) - 1):
                            if 0 <= route[i] < len(C) and 0 <= route[i + 1] < len(C):
                                x[route[i], route[i + 1], v - 1].Start = 1

                    model.optimize()

                    if model.status == GRB.INFEASIBLE:
                        raise ValueError("Gurobi model is infeasible. Check constraints and inputs.")

                    # Extract optimized routes
                    final_routes = {v: [0] for v in V}
                    route_costs = {v: 0 for v in V}
                    route_distances = {v: 0 for v in V}
                    visited = {v: {0} for v in V}

                    for v in V:
                        current_node = 0
                        while True:
                            next_node_found = False
                            for j in C:
                                if j not in visited[v] and x[current_node, j, v - 1].X > 0.5:
                                    final_routes[v].append(j)
                                    visited[v].add(j)
                                    distance = distance_matrix[current_node, j]
                                    route_distances[v] += distance
                                    route_costs[v] += cost_per_km[v] * distance
                                    current_node = j
                                    next_node_found = True
                                    break
                            if not next_node_found:
                                if current_node != 0:
                                    distance = distance_matrix[current_node, 0]
                                    final_routes[v].append(0)
                                    route_distances[v] += distance
                                    route_costs[v] += cost_per_km[v] * distance
                                break

                    return final_routes, route_costs, route_distances

            except gp.GurobiError as e:
                st.error(f"Gurobi error: {str(e)}")
                return initial_routes, {v: 0 for v in initial_routes.keys()}, {v: 0 for v in initial_routes.keys()}

        # Run ACO and Gurobi
        pheromone, initial_routes, aco_costs, aco_distances = aco_vrp(pheromone)
        final_routes, gurobi_costs, gurobi_distances = solve_with_gurobi(initial_routes)

        # Calculate total costs and distances
        total_cost_aco = sum(aco_costs.values()) if aco_costs else float('inf')
        total_distance_aco = sum(aco_distances.values()) if aco_distances else float('inf')
        total_cost_gurobi = sum(gurobi_costs.values()) if gurobi_costs else float('inf')
        total_distance_gurobi = sum(gurobi_distances.values()) if gurobi_distances else float('inf')

        return initial_routes, final_routes, total_distance_aco, total_cost_aco, total_distance_gurobi, total_cost_gurobi

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None, None, None, None, None

def main():
    st.title("Vehicle Routing Problem Solver")

    # File upload for distance matrix
    distance_matrix_file = st.file_uploader("Upload Distance Matrix (Excel file)", type=["xlsx"])
    if distance_matrix_file is not None:
        try:
            distance_matrix_df = pd.read_excel(distance_matrix_file, index_col=0)
            distance_matrix = distance_matrix_df.to_numpy()
        except Exception as e:
            st.error(f"Error reading distance matrix file: {e}")
            return
    else:
        st.warning("Please upload the distance matrix Excel file.")
        return

    # File upload for customer locations
    location_file = st.file_uploader("Upload Customer Locations (Excel file)", type=["xlsx"])
    if location_file is not None:
        try:
            location_df = pd.read_excel(location_file)
            customer_locations = {
                row['CustomerID']: (row['X-Coordinate'], row['Y-Coordinate'])
                for _, row in location_df.iterrows()
            }
        except Exception as e:
            st.error(f"Error reading location file: {e}")
            return
    else:
        st.warning("Please upload the customer locations Excel file.")
        return

    # Input for number of vehicles
    num_vehicles = st.number_input("Number of Vehicles:", min_value=1, step=1, value=1)

    # Input for vehicle capacities
    capacities = []
    for i in range(num_vehicles):
        capacity = st.number_input(f"Capacity of Vehicle {i + 1}:", min_value=1, step=1, value=2400)
        capacities.append(capacity)

    # Input for cost per km for each vehicle
    costs_per_km = []
    for i in range(num_vehicles):
        cost = st.number_input(f"Cost per km for Vehicle {i + 1}:", min_value=0.01, step=0.01, value=9.03)
        costs_per_km.append(cost)

    # Input for demand as comma-separated values
    demand_list = st.text_input("Enter Demand for each customer (comma-separated, e.g., 25,300,170,...):", "25,300,170,60,700,90,65,80,50,90,40,140,60,50")

    # Button to trigger route optimization
    if st.button("Optimize Routes"):
        # Solve VRP and get results
        initial_routes, final_routes, total_distance_aco, total_cost_aco, total_distance_gurobi, total_cost_gurobi = solve_vrp(
            distance_matrix, num_vehicles, capacities, costs_per_km, demand_list, customer_locations
        )

        # Display results
        if initial_routes and final_routes:
            st.header("Results")
            st.subheader("Initial Routes (ACO)")
            for v in range(1, num_vehicles + 1):
                st.write(f"Vehicle {v} route: {' -> '.join(map(str, initial_routes[v]))}")
            st.write(f"Total Distance of ACO Solution: {total_distance_aco:.2f} km")
            st.write(f"Total Cost of ACO Solution: {total_cost_aco:.2f} units")

            st.subheader("Optimized Routes (Gurobi)")
            for v in range(1, num_vehicles + 1):
                st.write(f"Vehicle {v} route: {' -> '.join(map(str, final_routes[v]))}")
            st.write(f"Total Distance of Optimized Gurobi Solution: {total_distance_gurobi:.2f} km")
            st.write(f"Total Cost of Optimized Gurobi Solution: {total_cost_gurobi:.2f} units")

            # Plot routes
            plot_routes(initial_routes, final_routes, customer_locations)
        else:
            st.write("Failed to find a solution. Please check your inputs.")

if __name__ == "__main__":
    main()
