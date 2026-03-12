def read_scor_file(file_path):
    """
    Reads a SCOR file and extracts metadata, node names, and flow matrix.
    
    Args:
        file_path (str): Path to the SCOR file.
        
    Returns:
        tuple: (F_matrix, n, nl, node_names) where:
            - F_matrix: pd.DataFrame with flow matrix (n x n)
            - n: int, total number of nodes
            - nl: int, number of living nodes
            - node_names: list of node names
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    line_idx = 0
    
    # Read metadata string (line 0)
    metadata = lines[line_idx].strip()
    line_idx += 1
    
    # Read n and nl
    n, nl = map(int, lines[line_idx].strip().split())
    line_idx += 1
    
    # Read node names (n rows)
    node_names = []
    for i in range(n):
        node_names.append(lines[line_idx].strip())
        line_idx += 1
    
    # Read biomass stocks (n rows with indices and values)
    biomass = [np.nan] * n
    for i in range(n):
        parts = lines[line_idx].strip().split()
        idx = int(parts[0])
        value = float(parts[1])
        biomass[idx-1] = value
        line_idx += 1
    
    # Skip -1 separator
    assert lines[line_idx].strip() == '-1'
    line_idx += 1
    
    # Read imports (n rows)
    imports = [np.nan] * n
    for i in range(n):
        parts = lines[line_idx].strip().split()
        idx = int(parts[0])
        value = float(parts[1])
        imports[idx-1] = value
        line_idx += 1
    
    # Skip -1 separator
    assert lines[line_idx].strip() == '-1'
    line_idx += 1
    
    # Read exports (n rows)
    exports = [np.nan] * n
    for i in range(n):
        parts = lines[line_idx].strip().split()
        idx = int(parts[0])-1
        value = float(parts[1])
        exports[idx-1] = value
        line_idx += 1
    
    # Skip -1 separator
    assert lines[line_idx].strip() == '-1'
    line_idx += 1
    
    # Read respirations (n rows)
    respirations = [np.nan] * n
    for i in range(n):
        parts = lines[line_idx].strip().split()
        idx = int(parts[0])
        value = float(parts[1])
        respirations[idx-1] = value
        line_idx += 1
    
    # Skip -1 separator
    assert lines[line_idx].strip() == '-1'
    line_idx += 1
    
    # Read flow matrix (rows with i, j, F_ji)
    F_matrix = np.zeros((n, n), dtype=float)
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        line_idx += 1
        
        if not line or line == '-1':
            break
        
        parts = line.split()
        i = int(parts[0]) - 1  # Convert to 0-indexed
        j = int(parts[1]) - 1  # Convert to 0-indexed
        flow = float(parts[2])
        F_matrix[j, i] = flow
    
    # Convert to DataFrame
    F_df = pd.DataFrame(F_matrix, index=node_names, columns=node_names)
    
    return n, nl, node_names, biomass, imports, exports, respirations, F_df, metadata
