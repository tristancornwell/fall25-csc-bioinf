# 1. Form a 2-D array using the recurrence relation for
# dynamic programming.
# 2. Create array containing “backtracking pointers”.
# 3. After reaching the sink, backtrack to source to
# produce a maximum-weight path.
# 4. Infer the alignment corresponding to this path.

# Recurrence relation:
# mismatch case:
# length(i, j) = maximum of:
# • length(i – 1, j) – 2
# • length(i, j – 1) – 2
# • length(i – 1, j – 1) – 3
# match case:
# length(i, j) = maximum of:
# • length(i – 1, j) – 2
# • length(i, j – 1) – 2
# • length(i – 1, j – 1) + 3

# Scoring parameters
match_score = 3
mismatch_score = -3
gap_penalty = -2

def global_align(seq1: str, seq2: str) -> int:
    n = len(seq1)
    m = len(seq2)

    # DP matrix initialization
    score: List[List[int]] = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Initialize first row and column with gap penalties
    for i in range(1, n + 1):
        score[i][0] = i * gap_penalty
    for j in range(1, m + 1):
        score[0][j] = j * gap_penalty

    # Fill in DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = score[i - 1][j - 1] + match_score
            else:
                diag = score[i - 1][j - 1] + mismatch_score

            up = score[i - 1][j] + gap_penalty
            left = score[i][j - 1] + gap_penalty

            # Take the maximum of diagonal, up, left
            best = diag
            if up > best:
                best = up
            if left > best:
                best = left

            score[i][j] = best

    # Final score (bottom-right cell)
    return score[n][m]
