import time
import numpy as np
import biotite.sequence.phylo as phylo

def test_upgma(tree, upgma_newick):
    ref_tree = phylo.Tree.from_newick(upgma_newick)
    for i in range(len(tree)):
        for j in range(len(tree)):
            assert abs(tree.get_distance(i, j) - ref_tree.get_distance(i, j)) < 1e-3
            assert tree.get_distance(i, j, topological=True) == ref_tree.get_distance(
                i, j, topological=True
            )


def test_neighbor_joining():
    dist = np.array([
        [0, 5, 4, 7, 6, 8],
        [5, 0, 7, 10, 9, 11],
        [4, 7, 0, 7, 6, 8],
        [7, 10, 7, 0, 5, 9],
        [6, 9, 6, 5, 0, 8],
        [8, 11, 8, 9, 8, 0],
    ])
    ref_tree = phylo.Tree(
        phylo.TreeNode([
            phylo.TreeNode([
                phylo.TreeNode([
                    phylo.TreeNode(index=0),
                    phylo.TreeNode(index=1)
                ], [1, 4]),
                phylo.TreeNode(index=2)
            ], [1, 2]),
            phylo.TreeNode([
                phylo.TreeNode(index=3),
                phylo.TreeNode(index=4)
            ], [3, 2]),
            phylo.TreeNode(index=5)
        ], [1, 1, 5])
    )
    test_tree = phylo.neighbor_joining(dist)
    assert test_tree == ref_tree

def test_distances(tree):
    dist = tree.root.distance_to(tree.leaves[0])
    for leaf in tree.leaves:
        assert leaf.distance_to(tree.root) == dist
    assert tree.get_distance(0, 19, True) == 9
    assert tree.get_distance(4, 2, True) == 10

if __name__ == "__main__":
    distances = np.loadtxt("tests/data/sequence/distances.txt", dtype=int)
    with open("tests/data/sequence/newick_upgma.txt") as f:
        newick = f.read().strip()

    start = time.perf_counter()
    # Run tests
    tree = phylo.upgma(distances)
    test_upgma(tree, newick)
    test_neighbor_joining()
    test_distances(tree)

    end = time.perf_counter()
    runtime_ms = (end - start) * 1000
    print(f"python      {runtime_ms:.5f}ms")
