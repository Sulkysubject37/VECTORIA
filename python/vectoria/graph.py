class Graph:
    """
    Immutable computation graph.
    """

    def compile(self):
        raise NotImplementedError("Compilation handled by C++ core")
