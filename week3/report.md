# Report for Week 3 Deliverable - Tristan Cornwell V00993072
## Steps
1. **Trim Test File**: I started by trimming the test_phylo.py down to just the required tests for this deliverable
2. **Determine Necessary Implementation Files**: I then manually traced through the tests to determine the necessary files for me to port: tree.py, upgma.py, and nj.py.
3. **Trim Implementation Files**: I was able to remove some of the code in tree.py that wasn't necessary for the tests. This made step 4 as easy as possible.
4. **Codon Translation**: This required the most time. I went back and forth using a combination of ChatGPT, Piazza posts/hints, as well as my own manual debugs based on compilation errors to get all three implementation files and my test file working in codon.
5. **Automation**: I got ChatGPT to generate a simple evaluate.sh that ran both my test scripts and printed their runtimes.
6. **CI Intergration**: I added the biotite install and new evaluate.sh to my CI pipeline. 

## Gotchas
- Between step 2 and 3, I took a while trying to figure out how to set up my directories so that my codon versions of the biotite files would import properly. The easiest solution ended up to be just putting my impolementation files in the same folder as my test files
- For step 4, I didn't find any major gotchas in particular: once I got my initial translated version from ChatGPT, I plugged away at compile-time errors one-by-one, but no individual error caused me an immense amount of grief. I will note that I found the Piazza hints and previous questions very helpful.
- All of the other steps went smoothly. The manual stuff in steps 1 and 2 took a bit of time, but I am very glad I did it without using an LLM, since it gave me much more confidence in my results and a better understanding of the code overall.

## Results
It seems to vary a bit, but both my python and codon runtimes are very low (around 2ms). Often, the python runtime is less than the codon one, which I assume is due to the data being relatively small (size 20 tree and 20x20 matrix for the test inputs of upgma and distances, respectively), so we therefore don't see the speedup that codon would usually provide for much larger data.

## Time Spent
This deliverable took me roughly 7 hours to complete.
