'''Class for foodwebs.'''
import networkx as nx
import numpy as np
import pandas as pd
# from .normalization import normalization_factory

def read_from_SCOR(scor_path):
    '''Reads a TXT file in the SCOR format and returns a FoodWeb object.

    Parameters
    ----------
    scor_path : string
        Path to the foodweb in SCOR format.


    Returns
    -------
    foodweb : foodwebs.Foodweb

    Description
    -------

    The SCOR format defines the graph through a list of edges (flows) and contains other food web data.
    Files in SCOR format look as follows (see examples/data/Richards_Bay_C_Summer.scor):
    --------------------
    title
    #of_all_nodes #of_living_nodes <-- size
    1st node name
    2nd node name
    ...
    1 biomass_of_the_1st_node
    2 biomass_of_the_2nd_node
    ...
    -1
    imports (same format as biomasses)
    -1
    exports (same format as biomasses)
    -1
    respirations (same format as biomasses)
    -1
    flows (in rows, e.g. '1 2 flow_from_1_to_2')
    -1
    --------------------

    Example:
    --------------------
    example_foodweb_1
    2 1
    A
    B
    1 0.00303
    2 0.05
    -1
    1 0.0018666315
    2 0.0
    -1
    1 3.35565e-07
    2 0.0001
    -1
    1 0.09925068
    2 1.45600009
    -1
    1 2 0.002519108
    -1
    --------------------


    '''
    with open(scor_path, 'r', encoding='utf-8') as f:
        print(f'Reading file: {scor_path}')
        title = f.readline().strip()
        size = f.readline().split()

        # check that size line has two values
        if len(size) != 2:
            raise Exception('Invalid SCOR file format.')

        n, n_living = int(size[0]), int(size[1])
        if n_living > n:
            raise Exception('Invalid input. The number of living species \
                             has to be smaller than the number of all nodes.')

        if n <= 0 or n_living <= 0:
            raise Exception('Number of nodes and number of living nodes have to be positive integers.')

        lines = [x.strip() for x in f.readlines()]

        net = pd.DataFrame(index=range(1, n+1))
        net['Names'] = lines[:n]
        net['IsAlive'] = [i < n_living for i in range(n)]
        # reading vector input
        for i, col in enumerate(['Biomass', 'Import', 'Export', 'Respiration']):
            # each section should end with -1
            if lines[(i + 1) * n + i + n] != '-1':
                raise Exception(f'Invalid SCOR file format. {col} section could be wrong, \
                                  the separator -1 could be in a wrong place, names list \
                                  could have wrong length.')

            net[col] = [float(x.split(' ')[1])
                        for x in lines[(i + 1) * n + i: (i + 2) * n + i]]
        # reading the edge/flow list
        flow_matrix = pd.DataFrame(index=range(1, n+1), columns=range(1, n+1))
        for line in [x.split(' ') for x in lines[(i + 2) * n + i + 1:]]:
            if line[0].strip() == '-1':
                break
            flow_matrix.at[int(line[0]), int(line[1])] = float(line[2])
        flow_matrix = flow_matrix.fillna(0.0)
        flow_matrix.index = net.Names
        flow_matrix.columns = net.Names
    return FoodWeb(title=title, flow_matrix=flow_matrix, node_df=net)

def calculate_trophic_levels(food_web):
    '''Calculate the fractional trophic levels of nodes using their the recursive
    relation. This implementation uses diet matrix to improve the numerical
    behavior of computation.

    In matrix form the trophic levels vector t is defined by
    t= 1 for primary producers and non-living nodes
    t= 1 + A t + b for other nodes
    where A_ij is a matrix of a fraction of node i diet a living node j contributes
    b=sum_k A_ik for k = non-living nodes

    Parameters
    ----------
    food web : foodwebs.FoodWeb
        Foodweb object.

    Returns
    -------
    trophic_levels : list
        List of trophic level values.
    '''
    data_size = len(food_web.flow_matrix)

    # the diagonal has the sum of all incoming system flows to the compartment i,
    # except flow from i to i
    A = food_web.get_diet_matrix().transpose()

    tl = pd.DataFrame(food_web.flow_matrix.sum(axis=0), columns=['inflow'])
    # here we identify nodes at trophic level 1
    tl['is_fixed_to_one'] = (tl.inflow <= 0.0) | (np.arange(data_size) >= food_web.n_living)
    tl['data_trophic_level'] = tl.is_fixed_to_one.astype(float)

    # counting the nodes with TL fixed to 1
    if (sum(tl.is_fixed_to_one) != 0):
        # update the equation due to the prescribed trophic level 1 - reduce the dimension of the matrix
        A_tmp = A.loc[~tl['is_fixed_to_one'], ~tl['is_fixed_to_one']]
        A_tmp = A_tmp*-1 + pd.DataFrame(np.identity(len(A_tmp)), index=A_tmp.index, columns=A_tmp.columns)

        B = pd.DataFrame(tl[~tl.is_fixed_to_one].is_fixed_to_one.copy())
        # filling the constants vector with ones - the constant 1 contribution
        B['b'] = 1
        # this is the diet fraction from non-living denoted as b in the function description
        B['b'] = B['b'] + A.loc[~tl['is_fixed_to_one'], tl['is_fixed_to_one']].sum(axis=1)

        A_inverse = np.linalg.pinv(A_tmp)
        tl.loc[~tl['is_fixed_to_one'], 'data_trophic_level'] = np.dot(A_inverse, B['b'])
    else:
        # fails with negative trophic levels = some problems
        np.linalg.pinv(A)
    return tl.data_trophic_level.values

__all__ = [
    'FoodWeb'
]

def calculate_trophic_levels_recursive(food_web, max_iter=1000, tol=1e-3):
    """
    Calculates trophic levels recursively. This method is more stable numerically.
    Parameters: food_web: foodwebs.FoodWeb, Foodweb object., max_iter: maximum number of iterations, tol: computational error tolerance
    Returns vector of TLs. 
    """
    F = food_web.flow_matrix
    n_living = food_web.n_living
    n = F.shape[0]
    trophic_levels = np.full(n, np.nan) 
    
    inflow = F.sum(axis=0)
    trophic_levels[(inflow <= 0.0) | (np.arange(n) >= n_living)] = 1
    
    def iterate_levels(trophic_levels, iteration=0):
        if iteration >= max_iter:
            raise RuntimeError("The maximum number of iterations has been exceeded.")
        
        new_trophic_levels = trophic_levels.copy()
        updated = False
        nominator = 0 
        for i in range(n):
            if np.isnan(trophic_levels[i]):
                #nominator = np.sum(F.iloc[i, :] * trophic_levels)
                denominator = inflow[i]
                for j in range(n):
                    flow = F.iloc[i,j]
                    if flow > 0 and not np.isnan(trophic_levels[j]):
                        nominator = nominator + flow * trophic_levels[j]
   
                if denominator > 0:
                    new_trophic_levels[i] = 1 + nominator / denominator
                    updated = True
                

        if not updated or np.nanmax(abs(new_trophic_levels - trophic_levels)) < tol:
            return new_trophic_levels
        else:
            return iterate_levels(new_trophic_levels, iteration + 1)
    
    return iterate_levels(trophic_levels)
    

class FoodWeb(object):
    '''
    Class defining a food web of an ecosystem.
    It stores species and flows between them with additional data like Biomass.
    '''

    def __init__(self, title, node_df, flow_matrix):
        '''Initialize a foodweb with title, nodes and flow matrix.
            Parameters
            ----------
            title : string
                Name of the foodweb.
            node_df : pd.DataFrame
                Species data respresented in a set of the following columns:
                ['Names', 'IsAlive', 'Biomass', 'Import', 'Export', 'Respiration']
            flow_matrix : pd.DataFrame
                Data containing list of flows between species, adjacency matrix,
                where the intersection between ith column and jth row represents
                flow from node i to j.
            See Also
            --------
            io.read_from_SCOR
        '''
        self.title = title
        self.node_df = node_df.set_index("Names")
        self.flow_matrix = flow_matrix

        self.n = len(self.node_df)
        self.n_living = len(self.node_df[self.node_df.IsAlive])

        if len(flow_matrix) > 1:
            # use local function to avoid circular import
            self.node_df['TrophicLevel'] = calculate_trophic_levels(self)
        self._graph = self._init_graph()

    def _init_graph(self):
        '''Returns networkx.DiGraph initialized using foodweb's flow matrix.'''
        graph = nx.from_pandas_adjacency(self.get_flow_matrix(boundary=True),  create_using=nx.DiGraph)
        nx.set_node_attributes(graph, self.node_df.to_dict(orient='index'))

        exclude_edges = []
        for n in self.node_df.index.values:
            exclude_edges.append((n, 'Import'))
            exclude_edges.append(('Export', n))
            exclude_edges.append(('Respiration', n))
        graph.remove_edges_from(exclude_edges)
        return graph

    def get_diet_matrix(self):
        '''Returns a matrix of system flows express as diet proportions=
        =fraction of node inflows this flow contributes'''
        return self.flow_matrix.div(self.flow_matrix.sum(axis=0), axis=1).fillna(0.0)

    def get_graph(self, boundary=False, mark_alive_nodes=False, normalization=None,
                  no_flows_to_detritus=False):
        '''Returns foodweb as networkx.SubGraph View fo networkx.DiGraph.

        Parameters
        ----------
        boundary : bool, optional (default=False)
            If True, boundary flows will be added to the graph.
            Boundary flows are: Import, Export, and Repiration.
        mark_alive_nodes : bool, optional (default=False)
            If True, nodes, which are not alive will have additional special sign near their name.
        normalization : string, optional (default=None)
            Defines method of graph edges normalization.
            Available options are: 'diet', 'log', 'donor_control',
            'predator_control', 'mixed_control', 'linear' and 'TST'.
        no_flows_to_detritus : bool, optional (default=False)
            If True, fLows to detritus will be excluded from the results.

        Returns
        -------
        subgraph : networkx.SubGraph
            A read-only restricted view of networkx.DiGraph.
        '''
        exclude_nodes = [] if boundary else ['Import', 'Export', 'Respiration']

        exclude_edges = []
        if no_flows_to_detritus:
            not_alive_nodes = self.node_df[~self.node_df.IsAlive].index.values
            exclude_edges = [edge for edge in self._graph.edges() if edge[1] in not_alive_nodes]

        g = nx.restricted_view(self._graph.copy(), exclude_nodes, exclude_edges)
        if mark_alive_nodes:
            # Optionally apply is_alive_mapping if available in this module
            mapper = globals().get('is_alive_mapping')
            if callable(mapper):
                g = nx.relabel_nodes(g, mapper(self))
        # g = normalization_factory(g, norm_type=normalization)
        return g

    def get_flows(self, boundary=False, mark_alive_nodes=False, normalization=None,
                  no_flows_to_detritus=False):
        '''Returns a list of all flows within foodweb.

        Parameters
        ----------
        boundary : bool, optional (default=False)
            If True, boundary flows will be added to the graph.
            Boundary flows are: Import, Export, and Repiration.
        mark_alive_nodes : bool, optional (default=False)
            If True, nodes, which are not alive will have additional special sign near their name.
        normalization : string, optional (default=None)
            Defines method of graph edges normalization.
            Available options are: 'diet', 'log', 'donor_control',
            'predator_control', 'mixed_control', 'linear' and 'tst'.
        no_flows_to_detritus : bool, optional (default=False)
            If True, fLows to detritus will be excluded from the results.

        Returns
        -------
        flows : list of tuples
            List of edges in graph's representation of a foodweb,
            each tuple is in a form of (from, to, weight).

        '''
        return (self.get_graph(boundary, mark_alive_nodes, normalization, no_flows_to_detritus)
                .edges(data=True))

    def get_flow_matrix(self, boundary=False, to_alive_only=False):
        '''Returns the flow (adjacency) matrix.

        Parameters
        ----------
        boundary : bool, optional (default=False)
            If True, boundary flows will be added to the graph.
            Boundary flows are: Import, Export, and Repiration.
        to_alive_only : bool, optional (default=False)
            If True, flow_matrix will include only flows to alive nodes
            (flows to not alive nodes will be 0)

        Returns
        -------
        flows_matrix : pd.DataFrame
            Rows/columns are species, each row/column intersection represents flow
            from ith to jth node.
        '''
        flow_matrix = self.flow_matrix.copy()

        if to_alive_only:
            flow_matrix.transpose()[~self.node_df['IsAlive']] = 0.0

        if not boundary:
            return flow_matrix

        flow_matrix_with_boundary = self.flow_matrix.copy()
        flow_matrix_with_boundary.loc['Import'] = self.node_df.Import.to_dict()
        flow_matrix_with_boundary.loc['Export'] = self.node_df.Export.to_dict()
        flow_matrix_with_boundary.loc['Respiration'] = self.node_df.Respiration.to_dict()
        return (
            flow_matrix_with_boundary
            .join(self.node_df.Import)
            .join(self.node_df.Export)
            .join(self.node_df.Respiration)
            .fillna(0.0))

    def get_links_number(self):
        '''Returns the number of nonzero flows.
        '''
        return self.get_graph(False).number_of_edges()

    def get_flow_sum(self):
        '''Returns the sum of all flows.
        '''
        return self.get_flow_matrix(boundary=True).sum()

    def get_norm_node_prop(self):
        num_node_prop = self.node_df[["Biomass", "Import", "Export", "Respiration"]]
        return num_node_prop.div(num_node_prop.sum(axis=0), axis=1)

    def __str__(self):
        return f'''
                {self.title}\n
                {self.node_df["Biomass"]}\n
                The internal flows matrix: a_ij=flow from i to j\n
                {self.flow_matrix}\n'
                {self.node_df["Import"]}\n
                {self.node_df["Export"]}\n
                {self.node_df["Respiration"]}\n
                {self.node_df["TrophicLevel"]}\n
                '''

    def get_outflows_to_living(self):
        # node's system outflows to living
        # TODO doc
        return self.flow_matrix[self.node_df[self.node_df.IsAlive].index].sum(axis='columns')