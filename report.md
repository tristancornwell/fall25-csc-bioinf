# Report for Week 1 Deliverable - Tristan Cornwell V00993072
## Steps
1. **My Personal Repository**: set up this repository with the structure outlined in the assignment. Copied over relevant code and data files from Zhongyu Chen's sample repository.
2. **Baseline Assembler**: ran the baseline python assmebler on the provided datasets and made skeleton script to automate it.
3. **Codon Translation**: With the help of ChatGPT (see ai.md), translated the baseline assembler to codon and compared the results with the python version.
4. **Automation**: filled in the skeleton script from step 2 to automate the codon runs as well. Printed the results in a table as specified in the deliverable description.
5. **CI Intergration**: added the script execution to the existing github actions from step 1 and modified the script to work with CI.

## Gotchas
- Steps 1 and 2 didn't really give me any trouble aside from some well-needed refreshers on git operations
- For Step 3, I had issues with typing and imports, as well as runtime flexibility
- Typing: the str|None (| unions) weren't supported in codon; ChatGPT suggested replacing them with Optional[str] or removed entirely, which worked
- Imports: I ran into compile time errors with os.path.join, matplotlib, and sys.argv. I was able to omit matplotlib since it wasn't used, hard-coded myinput data (temporarily) in place of using argv, and used a helper function in place of os.path.join
- Runtime Flexibility: I couldn't find a way to get command line arguments into my codon script so I could avoid hard-coding "dataX" as an input every time. I was instead able to use an environment variable and os.getenv to make it flexible for automation
- For step 4, I was able to get ChatGPT to generate a script for me by providing it the command I was using to run my python and codon assemblers, but I had to tweak it slightly to omit path prefixes in the table output
- For step 5, I had to modify my evaluate.sh to work from the base directory, since I had been running it locally from its own directory. I also had to download its required python libraries (in this case matplotlib) before it ran the assemblers. Setting the -x flag in my script was very helpful for finding these issues. 

## Results
As can be seen in the most recent CI pipeline, these are my results:
| Dataset |  Language |  Runtime (H:MM:SS)  |   N50   |    
|---------|-----------|--------------------|---------|
| data1   |   python  |   0:00:18   |   9990    |   
| data1   |   codon   |   0:00:10   |   9990    |  
| data2   |   python  |   0:00:35   |   9992    | 
| data2   |   codon   |   0:00:18   |   9992    |
| data3   |   python  |   0:00:40   |   9824    |
| data3   |   codon   |   0:00:19   |   9824    |
| data4   |   python  |   0:07:46   |   159255  |
| data4   |   codon   |   0:03:27   |   159255  |

Overall, it's clear through the comparison of these datasets that codon provides roughly a 50% speedup for this algorithm.
I think my results are quite accurate and line up roughly with the baseline provided in the sample repository, but, as noted in the Piazza Updates section, the genomes are unknown, which makes the repo not reproducible as-is. 
This deliverable also gave me a good idea for things to watch out for when translating code in the future (especially import errors, I spent the majority of my time debugging these).

