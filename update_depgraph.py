import re

with open('src/control_panel/orchestrator/depgraph.py', 'r') as f:
    content = f.read()

new_logic = '''
    def _check_for_cycles(self) -> None:
        """Check if the graph contains a cycle."""
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}
        adj: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        
        for edge in self.edges:
            adj[edge.from_op].append(edge.to_op)
            in_degree[edge.to_op] += 1
            
        import collections
        queue = collections.deque([n for n, d in in_degree.items() if d == 0])
        sorted_nodes = []
        while queue:
            u = queue.popleft()
            sorted_nodes.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("DependencyGraph contains a cycle")

    def parallelizable_fraction(self) -> float:
        """
        P in Amdahl's law: cost-weighted fraction of independent nodes.
        Returns a float between 0.0 and 1.0.
        """
        if not self.nodes:
            return 0.0
            
        self._check_for_cycles()
        
        total_time = sum(n.estimated_cost.estimated_runtime_ms for n in self.nodes.values())
        indep_nodes = self.independent_operations()
        
        if total_time == 0:
            return float(len(indep_nodes)) / float(len(self.nodes))

        indep_time = sum(self.nodes[n_id].estimated_cost.estimated_runtime_ms for n_id in indep_nodes)
        return float(indep_time) / float(total_time)
'''

# Remove everything from parallelizable_fraction down
content = re.sub(r'    def parallelizable_fraction\(self\) -> float:.*', new_logic.strip('\n'), content, flags=re.DOTALL)

with open('src/control_panel/orchestrator/depgraph.py', 'w') as f:
    f.write(content)
