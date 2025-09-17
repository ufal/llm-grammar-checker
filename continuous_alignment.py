# this function originates in https://github.com/ufal/asr_latency/blob/main/latency.py

def continuous_levenshtein_alignment(gt_words, asr_words):
    """
    Aligns two sequences of units called 'word' (which can be characters, actually) using a dynamic programming approach 
    similar to Levenshtein minimum edit distance. 
    The innovative adaptation is that it prioritizes continuous sequence of substitutions and copies over interruptions with insertions and deletions.
    We call this approach ``Continous Levenshtein Alignment'' in CUNI submission to IWSLT 2025 Simultaneous task.
    
    Args:
        gt_words (list): List of dictionaries representing ground truth words with 'word' and 'time'.
        asr_words (list): List of dictionaries representing ASR output words with 'word' and 'time'.

    example ('word' is a character):
        gt_words = asr_words = [{'word': 'H', 'time': 1.113}, {'word': 'e', 'time': 1.113}, ... ]
        
    Returns:
        list: A list of tuples, each containing a pair of aligned words from ground truth and ASR, or None for insertion/deletion.

    exaple return:

        [ ({'word': 'H', 'time': 1.113}, {'word': 'H', 'time': 2.6}),
            ...
          (None, {'word': 'e', 'time': 1.113})
        ]

    """

    
    # Initialize a 2D array to store distances
    m = len(gt_words)
    n = len(asr_words)
    # dp = dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # just ofr an easier access
    gt = [w["word"] for w in gt_words]
    asr = [w["word"] for w in asr_words]

    # Fill the Dynamic Programming table, like in the regular Levenshtein minimum edit distance alignment
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                cost = 0 if gt[i-1] == asr[j-1] else 1
                dp[i][j] = min(dp[i][j-1] + 1,      # Insert
                               dp[i-1][j] + 1,      # Delete
                               dp[i-1][j-1] + cost) # Substitute or match



    # for better code readability
    def delete_possible(i,j):
        return dp[i][j] == dp[i-1][j] + 1
    def insert_possible(i,j):
        return dp[i][j] == dp[i][j-1] + 1
    def copy_sub_possible(i,j):
        cost = 0 if gt[i-1] == asr[j-1] else 1
        if dp[i][j] == dp[i-1][j-1] + cost:
            return True
   
    # Backtrack to find the alignment 
    i, j = m, n
    alignment = []

    #### The adaptation for *Continuous* Levenshtein Alignment
    # To find the alignment with most substitutions and matches in the row,
    # we try operations in this order. 
    priorities = ["copy_sub", "delete", "insert"]

    while i > 0 and j > 0:
        for p in priorities:
            if p == "copy_sub" and copy_sub_possible(i,j):
                alignment.append((gt_words[i-1], asr_words[j-1]))
                i -= 1
                j -= 1
                # Whenever the continuation is not possible, we re-order the priorities for the next time.
                new_priorities = ["copy_sub", "delete", "insert"]  # as long as we can copy/sub, we do
                break
            if p == "delete" and delete_possible(i,j):
                alignment.append((gt_words[i-1], None))
                i -= 1
                new_priorities = ["delete", "insert", "copy_sub"]  # otherwise we prefer delete/insert over copy
                break 
            if p == "insert" and insert_possible(i,j):
                alignment.append((None, asr_words[j-1]))
                j -= 1
                new_priorities = ["insert", "delete", "copy_sub"]
                break
        priorities = new_priorities

    # deletions at the end
    while i > 0:
        alignment.append((gt_words[i-1], None))
        i -= 1

    # insertions at the end
    while j > 0:
        alignment.append((None, asr_words[j-1]))
        j -= 1

    # the alignment are ordered bottom-to-top. We reverse them, to be returned in top-to-bottom order
    alignment.reverse()
    return alignment

def to_dict_with_time(tokens):
    return [{"word": t, "time": None} for t in tokens]

def continuous_alignment(orig_tokens, corrected_tokens):
    out = continuous_levenshtein_alignment(to_dict_with_time(orig_tokens), to_dict_with_time(corrected_tokens))
    o = []
    for a,b in out:
        t = (a["word"] if a is not None else None, b["word"] if b is not None else None)
        o.append(t)
    return o