METRICS = ["SW", "CW1", "CW1N", "CW2", "CW3", "CW4", "CW5", "GWc", "GW", "GW1", "GW2"]
import numpy as np
from itertools import combinations
import utils

def SW(p, cities, railways, demands):
    """
    Social Welfare metric used in "Fair Railway Network Design" paper.
    """
    if len(railways) <= 0:
        return -1
    def cost(r):
        return r.time
    def path_cost(p):
        return sum(cost(r) for r in p)
    ij = list(combinations(cities, 2)) # all pairs of cities without repetition
    if p <= -1: # inf
        costs = [path_cost(utils.get_shortest_path(i, j, railways, True)[0]) for (i, j) in ij]
        val = max(costs)
    else:
        val = 0
        for (i, j) in ij:
            path, _ = utils.get_shortest_path(i, j, railways, True)
            k = demands[i.id][j.id]# / (i.pop * j.pop) if i.pop * j.pop != 0 else demands[i.id][j.id]
            val += (k * (path_cost(path) ** p)) # in this case path_cost suits
        val = val ** (1/p)
    return round(val, 2)

def CW1(p, cities, railways, demands, distances):
    """
    This metric is the same as SW except the cost is the time divided by the distance.
    Time should be in minutes and distance in meters or use the same scaling.
    """
    if len(railways) <= 0:
        return -1
    def cost(r):
        if r.dist == 0:
            print("/!\ null distance")
            return -1
        return (r.time / r.dist)
    def path_cost(p):
        return sum(cost(r) for r in p)
    ij = list(combinations(cities, 2))
    if p <= -1: # inf
        vals = []
        for (i, j) in ij:
            path, total_time = utils.get_shortest_path(i, j, railways, True)
            assert(sum(r.time for r in path) == total_time)
            optimal_dist = distances[i.id][j.id] # symmetric
            vals.append(sum(r.time for r in path) / optimal_dist)
        val = max(vals)
    else:
        val = 0
        for (i, j) in ij:
            path, _ = utils.get_shortest_path(i, j, railways, True)
            total_time = sum(r.time for r in path)
            optimal_dist = distances[i.id][j.id] # symmetric
            k = demands[i.id][j.id]# / (i.pop * j.pop) if i.pop * j.pop != 0 else demands[i.id][j.id]
            val += k * ((total_time / optimal_dist) ** p)
        val = val ** (1/p)
    return val

def CW2(p, cities, railways, demands, distances):
    """
    Similar to CW1 but we consider here weighting the demands separately from the norm.
    """
    if len(railways) <= 0:
        return -1
    def cost(r):
        return r.time
    def path_cost(p):
        return sum(cost(r) for r in p)
    ij = list(combinations(cities, 2))
    if p <= -1: # inf
        costs = [path_cost(utils.get_shortest_path(i, j, railways, True)[0]) for (i, j) in ij] # here suits too because of the specific cost
        val = max(costs)
    else:
        val = 0
        for (i, j) in ij:
            path, _ = utils.get_shortest_path(i, j, railways, True)
            total_time = sum(r.time for r in path)
            optimal_dist = distances[i.id][j.id]
            # sum(times) / sum(dist) != sum(time / dist)
            val += (demands[i.id][j.id] / optimal_dist) * (total_time ** p)
        val = val ** (1/p)
    return val

# Another version could consider the waiting times, however it is more relevant to wait for approximated real data inputs
# as it would just be increasing the time factor here otherwise.

def CW1N(p, cities, railways, demands, distances):
    """
    CW1 normalized.
    """
    if len(railways) <= 0:
        return -1
    def cost(r):
        if r.dist == 0:
            print("/!\ null distance")
            return -1
        return (r.time / r.dist)
    def path_cost(p):
        return sum(cost(r) for r in p)
    ij = list(combinations(cities, 2))
    m = len(cities)
    if p <= -1: # inf
        vals = []
        for (i, j) in ij:
            path = utils.get_shortest_path(i, j, railways, True)[0]
            optimal_dist = distances[i.id][j.id]
            vals.append(sum(r.time for r in path) / optimal_dist)
        val = max(vals) / (m*(m-1))
    else:
        val = 0
        for (i, j) in ij:
            path, _ = utils.get_shortest_path(i, j, railways, True)
            total_time = sum(r.time for r in path)
            optimal_dist = distances[i.id][j.id]
            val += demands[i.id][j.id] * ((total_time / optimal_dist) ** p)
        val = val ** (1/p)
        val /= m*(m-1)
    return val

def CW3(p, cities, railways, demands, distances):
    """
    We consider the distance to the max cost.
    """
    if len(railways) <= 0:
        return -1
    def cost(r):
        if r.dist == 0:
            print("/!\ null distance")
            return -1
        return (r.time / r.dist)
    def path_cost(p):
        return sum(cost(r) for r in p)
    ij = list(combinations(cities, 2))
    costs = []
    for (i, j) in ij:
        path = utils.get_shortest_path(i, j, railways, True)[0]
        optimal_dist = distances[i.id][j.id]
        costs.append(sum(r.time for r in path) / optimal_dist)
    max_cost = max(costs)
    if p <= -1: # inf
        vals = list(map(lambda c: np.abs(max_cost - c), costs))
        val = max(vals)
    else:
        val = 0
        for (i, j) in ij:
            path, _ = utils.get_shortest_path(i, j, railways, True)
            total_time = sum(r.time for r in path)
            optimal_dist = distances[i.id][j.id]
            k = demands[i.id][j.id]# / (i.pop * j.pop) if i.pop * j.pop != 0 else demands[i.id][j.id]
            val += k * ((np.abs(max_cost - (total_time / optimal_dist))) ** p)
        val = val ** (1/p)
    return val

def CW4(p, cities, railways, demands):
    """
    CW4 considers a first proposal of acc of a city from another.
    Noted acc(b)_a for accessibility of a from b we use it as weight for the demand.
    """
    if len(railways) <= 0:
        return -1
    def cost(r):
        return r.time
    def path_cost(p):
        return sum(cost(r) for r in p)
    ij = list(combinations(cities, 2))
    if p <= -1: # inf
        costs = [path_cost(utils.get_shortest_path(i, j, railways, True)[0]) for (i, j) in ij] # here suits too because of the specific cost
        val = max(costs)
    else:
        val = 0
        for (i, j) in ij:
            path, _ = utils.get_shortest_path(i, j, railways, True)
            total_time = sum(r.time for r in path)
            access = acc(i, j, railways) # accessibility of j from i
            print("access", access)
            # sum(times) / sum(dist) != sum(time / dist)
            k = demands[i.id][j.id]# / (i.pop * j.pop) if i.pop * j.pop != 0 else demands[i.id][j.id]
            val += (k * access) * (total_time ** p)
        val = val ** (1/p)
    return val

def CW5(p, cities, railways, demands, distances):
    """
    CW5 copies CW4 but divides by distance in the powered cost.
    """
    if len(railways) <= 0:
        return -1
    def cost(r):
        return (r.time / r.dist)
    def path_cost(p):
        return sum(cost(r) for r in p)
    ij = list(combinations(cities, 2))
    if p <= -1: # inf
        vals = []
        for (i, j) in ij:
            path = utils.get_shortest_path(i, j, railways, True)[0]
            optimal_dist = distances[i.id][j.id]
            vals.append(sum(r.time for r in path) / optimal_dist)
        val = max(vals)
    else:
        val = 0
        for (i, j) in ij:
            path, _ = utils.get_shortest_path(i, j, railways, True)
            total_time = sum(r.time for r in path)
            optimal_dist = distances[i.id][j.id]
            access = acc(i, j, railways) # accessibility of j from i
            print("access", access)
            k = demands[i.id][j.id]# / (i.pop * j.pop) if i.pop * j.pop != 0 else demands[i.id][j.id]
            # sum(times) / sum(dist) != sum(time / dist)
            val += (k * access) * ((total_time / optimal_dist) ** p)
        val = val ** (1/p)
    return val

# Of course, there are other variants

# We may consider attractiveness, centrality for contention points, robustness and so on

def acc(city_a, city_b, transitions):
    """
    We propose a first version of accessibility in a railway network for a given city.
    It is the inverse of sum_{path \in paths(a, b)}{dist(path)} / \sigma_{a->b}
    where \sigma_{a->b} denotes the number of distinct paths from a to b. The inverse serves the purpose of following
    an ascending order meaning greater accessibility.
    """
    paths = utils.all_paths(transitions, city_a, city_b)
    sigma = len(paths)
    print("sigma", sigma)
    def path_cost(p):
        return sum(r.dist for r in p) # dist in km
    return sigma / sum(path_cost(p) for p in paths)

# Gini-based indices

def GW1(p, cities, railways, demands, distances, demands_on=False):
    """
    Computes len(cities) Gini indices with y-axis = distance (vol d'oiseau) / time of travel and
    x-axis = cities (without self), for each city.
    Returns the lp norm of the Gini indices.
    """
    cities = set(cities)
    # Generalized Gini Welfare GW_1^p
    def cost(v, vp):
        path, total_time = utils.get_shortest_path(v, vp, railways, time=True)
        assert(total_time == sum(r.time for r in path))
        return distances[v.id][vp.id] / total_time # Distance à vol d'oiseau
        # We consider d/t rather than t/d because it's inversely growing
    gini_indices = []
    for v in cities:
        # 1. Costs are computed directly in A for computation gains
        # 2. Compute A = sum_{v≠v'} (Kv' - cost(v, v')) which is the distance to the equity case
        # v' corresponds to an index in the x-axis on the Lorenz graphic. We count negative costs.
        # 2.1. We need to create the x-axis by ordering costs for cities v' ≠ v
        if demands_on:
            sorted_cities = sorted(list(cities - {v}), key=lambda vp: demands[v.id][vp.id] * cost(v, vp))
        else:
            sorted_cities = sorted(list(cities - {v}), key=lambda vp: cost(v, vp))
        x_axis = {} # Save indices
        for i in range(1, len(sorted_cities)+1): # Indices start at 1
            x_axis[sorted_cities[i-1].id] = i
        # 2.2. Compute sum
        K = 1 # let's fix it to 1, it corresponds to y=(1)v'
        equity_func = lambda _: 1
        if demands_on:
            A = sum(demands[v.id][vp.id] * (equity_func(x_axis[vp.id]) - cost(v, vp)) for vp in (cities - {v}))
        else:
            A = sum((equity_func(x_axis[vp.id]) - cost(v, vp)) for vp in (cities - {v}))
        # 3. Compute Gini index A/(A+B)
        # A+B is actually just the surface between y=Kv' and y=0 (bound in x=max_v' cost(v')), so the surface of the rectangle triangle
        max_city = sorted_cities[-1]
        if demands_on:
            gini_index = A / (((len(cities) - 1) * demands[v.id][max_city.id] * cost(v, max_city)) / 2)
        else:
            gini_index = A / (((len(cities) - 1) * cost(v, max_city)) / 2)
        # 4. We may refine it by multiplying by the demand or something like that.
        # It could also be done before during graphic's building
        gini_indices.append(gini_index)
    if p >= 1:
        return sum(ind ** p for ind in gini_indices) ** (1/p)
    elif p == -1:
        return max(gini_indices)
    else:
        return -1

def GW2(cities, railways, demands, distances, demands_on=False):
    """
    Similar to GW1 but we consider only one Gini index with all pairs of cities as the x-axis.
    """
    cities = set(cities)
    def cost(v, vp):
        path, total_time = utils.get_shortest_path(v, vp, railways, time=True)
        assert(total_time == sum(r.time for r in path))
        return distances[v.id][vp.id] / total_time
    
    ij = list(combinations(cities, 2))

    if demands_on:
        sorted_pairs = sorted(ij, key=lambda pair: demands[pair[0].id][pair[1].id] * cost(pair[0], pair[1]))
    else:
        sorted_pairs = sorted(ij, key=lambda pair: cost(pair[0], pair[1]))

    x_axis = {} # Save indices
    for i in range(1, len(ij)+1): # Indices start at 1
        x_axis[sorted_pairs[i-1]] = i
    
    K = 1 # first value
    equity_func = lambda _: 1
    if demands_on:
        A = sum(demands[i.id][j.id] * (equity_func(x_axis[(i, j)]) - cost(i, j)) for (i, j) in ij)
    else:
        A = sum((equity_func(x_axis[(i, j)]) - cost(i, j)) for (i, j) in ij)
    
    max_city = sorted_pairs[-1]
    if demands_on:
        gini_index = A / ((len(ij) * demands[max_city[0].id][max_city[1].id] * cost(max_city[0], max_city[1])) / 2)
    else:
        gini_index = A / ((len(ij) * cost(max_city[0], max_city[1])) / 2)

    # Here, without refinement with demand there is no p to be used. Otherwise, it could be done in costs
    return gini_index

def GW(cities, railways, demands, distances):
    import matplotlib.pyplot as plt
    """
    This time, we consider the *true* Gini index, with cumulative demands as x-axis and cumulative costs t/d as y-axis.
    """
    cities = set(cities)
    def cost(v, vp):
        path, total_time = utils.get_shortest_path(v, vp, railways, time=True)
        assert(total_time == sum(r.time for r in path))
        print("cost:", distances[v.id][vp.id], "/", total_time, "=", distances[v.id][vp.id] / total_time)
        return distances[v.id][vp.id] / total_time
    
    ij = sorted(list(combinations(cities, 2)), key = lambda p: cost(p[0], p[1]))
    curr_x = 0 # cumulative quantities
    curr_y = 0
    A_total = 0
    AB_total = 0
    x_axis = [0]
    y_axis = [0]
    sc_factor = 1
    a = (sum(demands[v.id][vp.id] * cost(v, vp) for (v, vp) in ij) - cost(ij[0][0], ij[0][1])) / (sum(demands[v.id][vp.id] for (v, vp) in ij) - 1)
    #print("slope", a)
    for (v, vp) in ij:
        tau = demands[v.id][vp.id]
        c = cost(v, vp)
        A_v = (((tau*(tau + 1)) / 2) * (a - c) + a * tau * curr_x) - (tau * curr_y) # Cf. notes
        A_total += A_v

        #plot
        for i in range(0, int(tau/sc_factor)):
           x_axis.append(curr_x + (i+1))
           y_axis.append(curr_y + (i+1) * c)
        curr_x += tau # cumulative demand (population if actually distinct)
        curr_y += tau * c # last point
    
    AB_total = (curr_x * curr_y) / 2 # Surface A + B
    gini_index = A_total / AB_total

    plt.figure(figsize=(6, 6))
    plt.bar(x_axis, y_axis, label=r'$f_i$ (cumulated $\dfrac{d}{t}$)')
    plt.plot([0, max(x_axis)], [0, a*max(x_axis)], color='red', linestyle='--', label='y = x')

    plt.text(10, 25, f'A={round(A_total, 2)}', fontsize=16, ha='center', va='center')
    plt.text(10, 20, f'A+B={round(AB_total, 2)}', fontsize=16, ha='center', va='center')
    
    plt.xlabel('Cumulative demand', fontsize=15)
    plt.ylabel(r'Cumulative $\dfrac{d}{t}$', va='center', labelpad=20, fontsize=15)
    plt.title('Lorenz curve - Gini coefficient', fontsize=15)
    plt.legend()
    plt.grid(axis='y')
    plt.margins(x=0.05, y=0.1)
    plt.savefig("GWp", dpi=300, bbox_inches='tight')
    plt.show()

    return gini_index

def GWc(p, cities, railways, demands, distances):
    import matplotlib.pyplot as plt
    """
    This time, we consider the *true* Gini index, with cumulative demands as x-axis and cumulative costs t/d as y-axis.
    """
    cities = set(cities)
    def cost(v, vp):
        path, total_time = utils.get_shortest_path(v, vp, railways, time=True)
        assert(total_time == sum(r.time for r in path))
        return distances[v.id][vp.id] / total_time
    # The x-axis is based on all pairs of cities. One could consider n Gini indices for each city similarly to what is done above,
    # with the lp-norm computed at the end.
    # We sort the pairs of cities by cost. Actually, I don't really know about it.
    # For now, we are gonna go with the n indices.
    # For a given city v, for each other city v', compute t/d for (v,v'), sort them ascending order, compute cumulated t/d.
    gini_indices = []
    for v in cities:
        curr_x = 0 # cumulative quantities
        curr_y = 0
        A_total = 0
        AB_total = 0
        x_axis = sorted(list(cities - {v}), key=lambda c: cost(v, c)) # Sort by cost in ascending order, relevent to cumulative building for Gini
        a = (sum(demands[v.id][vp.id] * cost(v, vp) for vp in x_axis) - cost(v, x_axis[0])) / (sum(demands[v.id][vp.id] for vp in x_axis) - 1)
        for vp in x_axis:
            # Surface A in the Gini index for all v' given v
            tau = int(demands[v.id][vp.id])
            c = cost(v, vp)
            A_v = (((tau*(tau + 1)) / 2) * (a - c) + a * tau * curr_x) - (tau * curr_y) # Cf. notes
            A_total += A_v

            curr_x += tau # cumulative demand (population if actually distinct)
            curr_y += tau * c # last point
        AB_total = (curr_x * curr_y) / 2 # Surface A + B
        gini_index = A_total / AB_total
        gini_indices.append(gini_index)
    if p >= 1:
        return sum(ind ** p for ind in gini_indices) ** (1/p)
    elif p == -1:
        return max(gini_indices)
    else:
        return -1
        
LATEX_FORMULAS = {
    "SW": r"SW_p(R) = \left\{ \begin{array}{ll} \max_{1 \leq i < j \leq m} \, c_{ij} & \text{if } p \leq -1 (inf) \\ \left( \sum_{1 \leq i < j \leq m} \tau_{ij} \cdot (c_{ij})^p \right)^{1/p} & \text{else} \end{array} \right. \newline \mathrm{where}~c_{ij} = \mathrm{t}_{ij}",

    "CW1": r"CW1_p = \left\{ \begin{array}{ll} \max_{1 \leq i < j \leq m} \, c_{ij} & \text{if } p \leq -1 (inf) \\ \left( \sum_{1 \leq i < j \leq m} \tau_{ij} \cdot (c_{ij})^p \right)^{1/p} & \text{else} \end{array} \right. \newline\quad\quad \mathrm{where}~c{ij} = \frac{\mathrm{t}_{ij}}{\mathrm{d}_{ij}}",

    "CW2": r"CW2_p = \left\{ \begin{array}{ll} \max_{1 \leq i < j \leq m} \, c_{ij} & \text{if } p \leq -1 (inf) \\ \left( \sum_{1 \leq i < j \leq m} \frac{\tau_{ij}}{d_{ij}} \cdot (c(r))^p \right)^{1/p} & \text{else} \end{array} \right. \newline\quad\quad \mathrm{where}~c_{ij} = \frac{\mathrm{t}_{ij}}{\mathrm{d}_{ij}}",

    "CW3": r"CW3_p = \left\{ \begin{array}{ll} \max_{1 \leq i < j \leq m} \, c_{ij} & \text{if } p \leq -1 (inf) \\ \left( \sum_{1 \leq i < j \leq m} \frac{\tau_{ij}}{acc(j)_i} \cdot (c_{ij})^p \right)^{1/p} & \text{else} \end{array} \right. \newline\quad\quad \mathrm{where}~c_{ij} = \frac{\mathrm{t}_{ij}}{\mathrm{d}_{ij}} and acc(j)_i = \frac{\sum_{p \in paths(i, j) dist(p)}}{|paths(i, j)|}",
}