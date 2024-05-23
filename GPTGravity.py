import torch
import torch.optim as optim
import networkx as nx
import numpy as np
import scipy.stats as stats
import pickle

def pickleLoad(fname):
    if len(fname) > 0:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        return []

def weightedMuStd(x, w = None):
    if w is None:
        w = x * 0 + 1
    mu = np.average(x, weights = w)
    sigma = np.sqrt(np.sum(w * (x - mu)**2) / np.sum(w))

    x = (x.copy() - mu)/sigma
    xi = np.argsort(x)
    n = len(x)
    cdf = np.cumsum(w[xi]) / np.sum(w)
    # Calculate the Kolmogorov-Smirnov statistic.
    statistic = np.max(np.abs(cdf - stats.norm.cdf(x[xi]))) #/ np.sqrt(n)
    # Calculate the p-value.
    try:
        p_value = 1 - stats._ksstats.kolmogn(n, statistic)#stats.norm.cdf(1, 0, statistic)
    except:
        p_value = np.nan
    #ks_test = stats.kstest(x, 'norm', weights = w)
    return mu, sigma, p_value

def cdist(x, y):
    return np.sqrt(((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2).sum(axis=2))

def estimatePointArea_(x, pop, mcn = 10000, mcm = 5): #monte carlo simulation for estimating areas of the voronoi cells
    #if maxd == 0:
    d = cdist(x, x)
    nk = 3
    d.sort(axis = 1)
    dmax = d[:, nk]

    min_x = np.min(x, axis=0) - max(dmax)
    max_x = np.max(x, axis=0) + max(dmax)
    tot_area = (max_x - min_x).prod()
    single_area = tot_area / mcn / mcm
    N = x.shape[0]

    # Generate random Monte Carlo points within the bounding box of x
    neighborhood_counts = np.zeros(N)
    cpi = {}

    for i in range(mcm):
        mc_points = np.random.uniform(low=min_x, high=max_x, size=(mcn, x.shape[1]))

        # Calculate all distances from MC points to points in x using broadcasting
        distances = cdist(mc_points, x)
        # Find the index of the closest point in x for each MC point
        cpi[i] = np.argmin(distances, axis=1)
        cpi[i][np.min(distances, axis = 1) > dmax[cpi[i]]] = -1 #masking out mc point that fall too far from x

        # Count occurrences of each point in x being the closest
        neighborhood_counts += np.bincount(cpi[i][cpi[i] >= 0], minlength = N)

    mcpop = np.zeros((mcn, mcm))
    pop_ = pop / neighborhood_counts
    for i in range(mcm):
        mcpop[:, i] = pop_[cpi[i] * (cpi[i] >= 0)] * (cpi[i] >= 0)

    mcpop = mcpop.flatten()

    return neighborhood_counts * single_area, mcpop[mcpop > 0] / single_area


def initPopulation(n, m, seed, params = {}, torus_dist = True):
    def cdist_torus(xy1, xy2, xym): #override cdist with
        xd = abs(xy[:, 0].reshape((-1, 1)) - xy[:, 0].reshape((1, -1)))
        xd = np.minimum(xd, xym[0] - xd)
        yd = abs(xy[:, 1].reshape((-1, 1)) - xy[:, 1].reshape((1, -1)))
        yd = np.minimum(yd, xym[1] - yd)
        l2dist = True
        return (xd ** 2 + yd ** 2) ** 0.5 if l2dist else xd + yd

    np.random.seed(seed)
    pop = np.ones(m) * params.get('init_pop', 1.0)
    dm = int(m ** 0.5)
    xy = np.concatenate([np.arange(0, m).reshape((-1, 1)) % dm, np.arange(0, m).reshape((-1, 1)) // dm], axis = 1)
    if torus_dist:
        d = cdist_torus(xy, xy, xy.max(axis=0) + 1)
    else:
        d = cdist(xy, xy)
    pl = np.zeros(n, dtype = int)
    return pop, xy, d, pl

def arrayXYindexing(a, x, y):
    return a.flatten()[x * a.shape[1] + y]

def simulateRWPopulation(n, m, params, iterN = 100, seed = 1): #Random walk population simulation - explanatory model for log-normal emergence
    #n - number of people to distribute, in addition to m initial places in m slots of the lattice
    #m - lattice size
    #def randomStep(k):
    #    return np.random.choice(neighbors[k])
    def randomStep(I, w = None): #define random step to one of the neighbors with or without neighbor weighting
        if w is None:
            i = (np.random.uniform(size = len(I)) * (neigh_count[I] - 0.000001)).astype(int)
        else:
            neighw = w[neighbors + abs(neighbors) * (neighbors < 0)] * (neighbors >= 0)
            neighcw = neighw.cumsum(axis = 1) / neighw.sum(axis = 1).reshape((-1, 1))
            p = np.random.uniform(size = len(I)).reshape((-1, 1))
            i = (p > neighcw[I, :]).sum(axis = 1)
        return arrayXYindexing(neighbors, I, i)

    pop, xy, d, _ = initPopulation(n, m, seed, params) #initialize the lattice, population on it (constant initial) and distances
    wpop = pop.copy() #work population
    mxy = {tuple(xy[i, :]) : i for i in range(m)}
    xm = max(xy[:, 0]); ym = max(xy[:, 1])
    #neighbors = [np.where(d[i, :] == 1)[0] for i in range(m)]
    neigh_count = (d == 1).sum(axis = 1)
    neighbors = np.array([np.pad(np.where(d[i, :] == 1)[0], (0, 4 - neigh_count[i]), constant_values = (-1, -1)) for i in range(m)])
    I = np.random.choice(range(m), size = n) #random location of individuals
    S = - np.ones((n, 2), dtype=int) #their settlement status (first column - home location, second - work location)
    P = np.zeros((2, 2)) #choice probabilities, state dependant, P[0, 0] - not settled
    pnr = params.get('pnr', 0.01) #not settled, to settle for residence
    pnw = params.get('pnw', pnr) #not settled, to settle for work
    psw = params.get('psw', pnw) #settled for residence, to settle for work
    psr = params.get('psr', psw) #settled for work, to settle for residence
    p_pop = lambda p, pop: 1 - (1 - p) ** pop if p > 0 else abs(p)
    while (S.min(axis = 1) < 0).sum() > 0:
        dm = d <= params.get('rad', 1.0)
        wr0 = dm.sum(axis = 1) - 1
        neigh_pop = (wr0 * pop + np.matmul(dm, pop.reshape(-1, 1)).flatten()) / (wr0 + dm.sum(axis = 1))
        neigh_wpop = (wr0 * wpop + np.matmul(dm, wpop.reshape(-1, 1)).flatten()) / (wr0 + dm.sum(axis = 1))
        ph = (S[:, 0] < 0) * (p_pop(pnr, neigh_pop[I]) * (S[:, 1] < 0) + p_pop(psr, neigh_pop[I]) * (S[:, 1] >= 0))
        pw = (S[:, 1] < 0) * (p_pop(pnw, neigh_wpop[I]) * (S[:, 0] < 0) + p_pop(psw, neigh_pop[I]) * (S[:, 0] >= 0))

        hsettle_ind = np.random.uniform(size = n) < ph
        wsettle_ind = np.random.uniform(size = n) < pw
        pop += np.bincount(I[hsettle_ind], minlength = m)
        wpop += np.bincount(I[wsettle_ind], minlength = m)

        S[hsettle_ind, 0] = I[hsettle_ind]
        S[wsettle_ind, 1] = I[wsettle_ind]
        I[S.min(axis = 1) < 0] = randomStep(I[S.min(axis = 1) < 0]) #, w = pop)
        print('Active actors = {}'.format((S.min(axis = 1) < 0).sum()))
    mob = np.zeros((m, m))
    for i in range(n):
        mob[S[i, 0], S[i, 1]] += 1

    return pop, wpop, mob, xy, d

def simulateCity(m = 900, n = 10000, params = {}, seed = 1):
        pop, jobs, mob, xy, d = simulateRWPopulation(n = n, m = m, params = params, seed = seed, iterN = 10) #simulate jobs
        analyzePopDistr('Synthetic pop', pop = pop, xy = xy, logpop = True)
        analyzePopDistr('Synthetic jobs', pop = jobs, xy = xy, logpop = True)


def analyzePopDistr(city, pop, xy, logpop = True): #analysis of the population and flow distribution for the city based on initial or fitted node coordinates xy
        area, ldens = estimatePointArea_(xy, pop, mcn = 5000, mcm = 100)
        w = ldens
        if logpop:
            ldens = np.log(ldens)
        ldens = np.log(pop[area > 0]) - np.log(area[area > 0]) #density of areas work, while density of individual points does not - sample too large for ks-test?
        w = pop[area > 0]
        #np.concatenate([area.reshape(-1,1), area_.reshape(-1,1)], axis = 1)
        pop_mu, pop_sigma, pop_pv = weightedMuStd(x = ldens, w = w)
        print('{}, pop_mu = {:.4f}, pop_sigma = {:.4f}, pop_pv = {:.7f}'.format(city, pop_mu, pop_sigma, pop_pv))

class GravityModel:
    def __init__(self, locations, weights, populations, initial_alpha=1.0, fine_tune_locations=False, lossType = 'MSE', dpow = 2):
        """
        Initializes the Gravity Model with given locations, weights, populations, and model parameters.

        :param locations: Tensor of shape (n, 2) representing the initial xy coordinates of each location.
        :param weights: Tensor of shape (n,) representing the attractiveness weights of each location.
        :param populations: Tensor of shape (n,) representing the population of each location.
        :param initial_alpha: Initial value for the alpha parameter.
        :param learning_rate: Learning rate for the optimizer.
        :param fine_tune_locations: Boolean to decide if xy coordinates should be fine-tuned.
        """
        self.locations = torch.tensor(locations, requires_grad=fine_tune_locations)
        self.weights = torch.tensor(weights)
        self.populations = torch.tensor(populations)
        self.alpha = torch.tensor([initial_alpha], requires_grad=True)
        self.params = [self.alpha]
        self.lossType = lossType
        self.dpow = dpow
        if fine_tune_locations:
            self.params.append(self.locations)

    def calculate_distances(self):
        """Calculate and return the Euclidean distances between all locations."""
        return torch.cdist(self.locations, self.locations)

    def gravity_matrix(self, distances):
        """
        Construct the gravity model matrix based on the provided distances.
         :param distances: Distance matrix of shape (n, n).
        """
        flows = self.weights * torch.exp(-self.alpha * distances ** self.dpow)
        flow_sums = flows.sum(dim=1, keepdim=True)
        normalized_flows = self.populations[:, None] * flows / flow_sums
        return normalized_flows

    def MSE(self, Y, Ytrue, mask = None, log = True): #log MSE loss for the unconstrained model
        if mask is None:
            mask = Y * 0 + 1
        f = lambda x : torch.log(x + 1) if log else x
        loss = torch.mul(mask , (f(Ytrue) - f(Y)) ** 2).sum() / torch.mul(mask , f(Ytrue) ** 2).sum() #mask.sum()
        return loss

    def binomialB1(self, Y, Ytrue, mask = None): #binomial likelihood loss for the B1 model
        if mask is None:
            mask = Y * 0 + 1
        TM = torch.mul(Y, mask).sum(dim = 1).reshape((-1, 1)) #outflow weights
        PM = torch.div(Y , TM) #normalization to get outflow probabilities
        LL = - torch.mul(mask, torch.mul(Ytrue, torch.log(PM + 1e-8))).sum() / torch.mul(mask, Ytrue).sum() #(mask == 0) + (Ytrue == 0)#add constant to avoid zero probabilities
        return LL

    def loss(self, Y = None, Ytrue = None): #compute loss
        return self.MSE(Y = Y, Ytrue = Ytrue, mask = self.edgebatch, log = self.lossType[:3] == 'log') if self.lossType[-3:] == 'MSE' else self.binomialB1(Y = Y, Ytrue = Ytrue, mask = self.edgebatch)

    def evaluate(self, mask = None, baseline = False): #evaluate model performance on a given sample of edges (mask); baseline will replace the embedding-based attraction Y with a null model Y = 1
        if mask is None:
            mask = np.ones(self.A.shape)
        mask = torch.FloatTensor(mask)
        Ytrue = torch.FloatTensor(self.fullA)
        if baseline == 'const':
            Y = self.W
        elif baseline == 'truemax':
            Y = Ytrue
            Y[Y == 0] = 1e-6
        else:
            Y = self.forward(X = None)
        return self.logMSE(Y = Y, Ytrue = Ytrue, mask = mask, log = self.lossType[:3] == 'log').item() if self.lossType[-3:] == 'MSE' else self.binomialB1(Y = Y, Ytrue = Ytrue, mask = mask).item()

    def fit(self, true_flows, steps=1000, print_every=100, learning_rate=0.01, edgebatching = 0):
        """Fit the model by optimizing the alpha parameter and optionally the locations."""
        true_flows = torch.tensor(true_flows)
        self.optimizer = optim.Adam(self.params, lr=learning_rate)
        self.edgebatch = None
        for step in range(steps):
            self.optimizer.zero_grad()
            distances = self.calculate_distances()
            flows = self.gravity_matrix(distances)
            if edgebatching > 0:
                self.edgebatch = torch.tensor(np.random.uniform(size = flows.shape) < edgebatching)
            loss = self.loss(flows, true_flows)
            if step == 0:
                print(f"Initial loss = {loss.item()}, Alpha = {self.alpha.item()}")
            loss.backward()
            self.optimizer.step()

            if ((step + 1) % print_every == 0) or (step >= steps - 1):
                print(f"Step {step + 1}: Loss = {loss.item()}, Alpha = {self.alpha.item()}")

    def predict(self):
        """Use the trained model to predict the gravity matrix."""
        distances = self.calculate_distances()
        return self.gravity_matrix(distances)

class BatchingGravityModel(GravityModel):
    def fit(self, true_flows, steps=1000, print_every=1, batch_size=32, learning_rate=0.01):
        true_flows = torch.tensor(true_flows)
        self.optimizer = optim.Adam(self.params, lr=learning_rate)
        self.batch_size = batch_size
        distances = self.calculate_distances()
        num_locations = distances.size(0)
        edge_indices = torch.cartesian_prod(torch.arange(num_locations), torch.arange(num_locations))

        for step in range(steps):
            self.optimizer.zero_grad()
            total_loss = 0
            # Shuffle the edge indices to randomize batches
            edge_indices = edge_indices[torch.randperm(edge_indices.size(0))]
            bc = 0
            for i in range(0, edge_indices.size(0), self.batch_size):
                batch_indices = edge_indices[i : i + self.batch_size]
                batch_flows = self.gravity_matrix(distances)[batch_indices[:, 0], batch_indices[:, 1]]
                # Placeholder loss function: sum of batch flows (adjust as needed)
                loss = self.loss(batch_flows, true_flows[batch_indices[:, 0], batch_indices[:, 1]])
                loss.backward()
                total_loss += loss.item()
                bc += 1

            self.optimizer.step()

            if ((step + 1) % print_every == 0) or (step >= steps - 1):
                print(f"Step {step}: Avg Loss = {total_loss / bc}")

#Pick a city to analyze or '' for simulation explaining log-normal pop distribution emergence
city = 'Philadelphia' # 'New York City' #'Philadelphia' #Boston 'Philadelphia' New York City

path = '/Users/stanislav/Desktop/NNClustering/LabCodes/SFdata/'

if len(city) == 0: #simulate city
    simulateCity(m = 400, n = 100000, params = {'pnr' : 0.0001, 'pnw': 0.0004, 'psr': 0.001, 'psw': 0.001, 'rad' : 1.0}, seed = 1)
else: #real city
    LEHD = pickleLoad(path + 'LEHD_cities.pkl')
    nodes = pickleLoad(path + 'LEHD_citynode_attributes.pkl')
    CG = LEHD[city]
    nodes = nodes[city]
    G = nx.DiGraph()
    G.add_weighted_edges_from([(e[0], e[1], e[2]['S000']) for e in CG.edges(data = True)])
    nodepop = dict(G.out_degree(weight="weight")); nodepop = np.array([nodepop[n] for n in G.nodes()])
    nodepop2 = dict(G.in_degree(weight="weight")); nodepop2 = np.array([nodepop2[n] for n in G.nodes()])
    sx = np.cos(np.pi / 180 * 40)
    nodexy = np.array([[nodes[n][0] * sx, nodes[n][1]] for n in G.nodes()])
    #nodearea = np.array([[nodes[n][2]] for n in G.nodes()])

    A = nx.to_numpy_array(G) #mobility matrix

    analyzePopDistr(city, pop = nodepop, xy = nodexy)

    # Instantiate and fit model, enabling location fine-tuning
    model = GravityModel(locations = nodexy, initial_alpha = 1.0, weights = nodepop2, populations = nodepop, fine_tune_locations = False, lossType = 'binomialB1', dpow = 2)
    model.fit(true_flows = A, steps = 500, learning_rate = 0.1, edgebatching = 0) #, batch_size = 1000
    model2 = GravityModel(locations = nodexy, initial_alpha = model.alpha.item(), weights = nodepop2, populations = nodepop, fine_tune_locations = True, lossType = 'binomialB1', dpow = 2)
    model2.fit(true_flows = A, steps = 1000, learning_rate = 0.0003, edgebatching = 0) #, batch_size = 1000
    #model2.fit(true_flows = A, steps = 100, learning_rate = 0.0003, edgebatching = 0)

    analyzePopDistr(city, pop  = nodepop, xy = model2.locations.detach().data.numpy())

    final_flows = model2.predict()