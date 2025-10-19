# Start with global alignment, but allow alignment of seq2 to
# fit anywhere within seq1 by initializing first column to 0,
# allowing gaps in seq2 at beginning, and taking maximum over 
# last column for allowing seq2 to end anywhere.

# Scoring parameters
match_score = 3
mismatch_score = -3
gap_penalty = -2

def fitting_align(seq1: str, seq2: str) -> int:
    n = len(seq1)
    m = len(seq2)

    # DP matrix: List of Lists of int
    score: List[List[int]] = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Initialize first row: regular gap penalties (for seq2)
    for j in range(1, m + 1):
        score[0][j] = j * gap_penalty

    # Initialize first column: 0s (free gaps in seq1)
    for i in range(1, n + 1):
        score[i][0] = 0

    # Fill in DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = score[i - 1][j - 1] + match_score
            else:
                diag = score[i - 1][j - 1] + mismatch_score

            up = score[i - 1][j] + gap_penalty
            left = score[i][j - 1] + gap_penalty

            # Choose the max of diag, up, left
            best = diag
            if up > best:
                best = up
            if left > best:
                best = left

            score[i][j] = best

    # Find the best alignment score in the last column (aligning all of seq2)
    best_score = score[0][m]
    for i in range(1, n + 1):
        if score[i][m] > best_score:
            best_score = score[i][m]

    return best_score
