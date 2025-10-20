# Report for Week 4 Deliverable - Tristan Cornwell V00993072
## Steps
1. **Review algorithms and develop plans**: I started by going over the lecture slides and typing out plans for each of the algorithms at the top of the codon files.
2. **Generate test script**: I then got ChatGPT to generate a fasta file parser and a test runner that would take pairs of input files to compare with different algorithms.
3. **Generate algorithms**: Starting with global alignment, I gave my algorithm outlines to ChatGPT and got it to generate the implementation code for them.
4. **Debug algorithms**: I read over my algorithms to ensure they were actually doing what I intended, double-checking anything that seemed off.
5. **Generate proper asserts**: I ran my algorithms while printing the results and got ChatGPT to generate the expected results for each of the test pairs, then added asserts to my test script based on the expected results.
6. **Automation**: I drafted a simple evaluate.sh that ran both my test scripts and printed their runtimes and debugged it using ChatGPT.
7. **CI Intergration**: I added the new evaluate.sh to my CI pipeline. 

## Gotchas
- For me, the most difficult algorithm to get right was the fitting alignment algorithm. It took me a while to figure out the necessary tweaks to allow for any substring of the second sequence to be considered.
- Also for fitting alignment, I spent a while trying to debug something that wasn't a bug, as I was manually calculating the expected results based on the assumption that the wrong string was the second sequence (thus considering substrings of the wrong string)
- The last gotcha for me was the automation of my test scripts. It took a bit to figure out the proper way to format my evaluate.sh so it could be run from the proper directory and access the data files properly.

## Results
As seen in the results of my CI for week 4, the python tests, on average, took longer than the codon ones for tx vs qx comparisons in all algorithms. The results from `global_alignment` can be seen in Table 1, below.
| Method      |      Language  |  Runtime |
|-------------|----------------|----------|
|global-q1    |     codon      | 0.00524521ms|
|global-q1    |     python     | 0.013828277587890625ms
|global-q2    |     codon      | 0.0264645ms|
|global-q2    |     python     | 0.06794929504394531ms|
|global-q3    |     codon      | 0.4282ms|
|global-q3    |     python     | 0.9660720825195312ms|
|global-q4    |     codon      | 0.0140667ms|
|global-q4    |     python     | 0.025033950805664062ms|
|global-q5    |     codon      | 0.120401ms|
|global-q5    |     python     | 0.331878662109375ms|

The most obvious results that highlight codon's efficiency over python, are the mt-human vs mt-orang comparisons. The codon versions run in ~75 seconds for affine alignment and ~25 seconds for all other alignment algorithms, while the python ones fail. When trying to run them locally, they crashed my terminal, and when running on CI, the job times out and fails.

## Time Spent
This deliverable took me roughly 5 hours to complete.
