# Start with global alignment, then just add zero-weight edge from (source -> every node) and (every node -> sink)

# Recurrence relation for local alignment:
# si, j = maximum of:
# 	0, 
# 	(si1, j) + Score(vi, -),
# 	(si, j1) + Score(-, wj),
# 	(si1, j1) + Score(vi, wj) 
# 	
# where the scores here
# are –2, –2, and either
# +3 or –3 (depending
# on a match vs. a
# mismatch).

# Scoring parameters
match_score = 3
mismatch_score = -3
gap_penalty = -2

def local_align(seq1: str, seq2: str) -> int:
    n = len(seq1)
    m = len(seq2)

    # DP matrix initialization
    score: List[List[int]] = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    max_score = 0  # keep track of maximum score in the matrix

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = score[i - 1][j - 1] + match_score
            else:
                diag = score[i - 1][j - 1] + mismatch_score

            up = score[i - 1][j] + gap_penalty
            left = score[i][j - 1] + gap_penalty

            # Local alignment: allow zero as the floor
            best = diag
            if up > best:
                best = up
            if left > best:
                best = left
            if 0 > best:
                best = 0

            score[i][j] = best

            # Update max score
            if best > max_score:
                max_score = best

    return max_score