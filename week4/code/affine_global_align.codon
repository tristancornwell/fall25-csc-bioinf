# Affine penalty: a way of scoring contiguous gaps
# higher than discontiguous gaps.
# • gap opening penalty (σ): given to first symbol.
# • gap extension penalty (ε): given to extra symbols.
# opening and extension penalties -5 and -1.
# - Do global alignment, but use a three-level global alignment manhattan graph, where each level has the following respective recurrence relations:

# Recurrence relation for affine gap global alignment:
# loweri, j = maximum of:
#     (loweri1, j) - 1
#     (middlei1, j) - 5

# middlei,j = maximum of:
#     (loweri, j)
#     (middlei1, j1) + Score(vi, wj)
#     (upperi, j)

# upperi,j = maximum of:
#     (upperi, j1) - 1
#     (middlei, j1) - 5

# Scoring parameters
match_score = 3
mismatch_score = -3
gap_open_penalty = -5
gap_extension_penalty = -1

# large negative number for initialization
NEG_INF = -1_000_000

def affine_global_align(seq1: str, seq2: str) -> int:
    n = len(seq1)
    m = len(seq2)

    # Initialize 3 DP matrices
    lower: List[List[int]] = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    middle: List[List[int]] = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    upper: List[List[int]] = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Initialize first row and column
    for i in range(1, n + 1):
        lower[i][0] = gap_open_penalty + (i - 1) * gap_extension_penalty
        middle[i][0] = gap_open_penalty + (i - 1) * gap_extension_penalty
        upper[i][0] = NEG_INF  # illegal state

    for j in range(1, m + 1):
        upper[0][j] = gap_open_penalty + (j - 1) * gap_extension_penalty
        middle[0][j] = gap_open_penalty + (j - 1) * gap_extension_penalty
        lower[0][j] = NEG_INF  # illegal state

    # Fill DP tables
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # lower[i][j] = gap in seq2 (vertical)
            lower[i][j] = max(
                lower[i - 1][j] + gap_extension_penalty,
                middle[i - 1][j] + gap_open_penalty
            )

            # upper[i][j] = gap in seq1 (horizontal)
            upper[i][j] = max(
                upper[i][j - 1] + gap_extension_penalty,
                middle[i][j - 1] + gap_open_penalty
            )

            # middle[i][j] = match/mismatch
            if seq1[i - 1] == seq2[j - 1]:
                score = match_score
            else:
                score = mismatch_score

            middle[i][j] = max(
                lower[i][j],
                middle[i - 1][j - 1] + score,
                upper[i][j]
            )

    # The bottom-right cell of middle matrix contains final global alignment score
    return middle[n][m]