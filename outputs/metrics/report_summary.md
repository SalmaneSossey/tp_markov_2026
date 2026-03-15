# Report Notes

## Data Source
- Train URL: https://www.gutenberg.org/cache/epub/11/pg11.txt (fallback)
- Same-style URL: https://www.gutenberg.org/cache/epub/12/pg12.txt (fallback)
- Different-style URL: https://www.gutenberg.org/cache/epub/1342/pg1342.txt (fallback)

## Preprocessing Choices
- Lowercased all text.
- Kept only letters a-z and spaces.
- Collapsed repeated whitespace to one space.
- Added `^` start marker and `$` end marker.

## Top Transitions
- e ->  : 19
- h -> e: 16
- i -> n: 15
-   -> t: 13
- d ->  : 12
- e -> r: 12
- t -> h: 12
-   -> o: 11
- r ->  : 11
-   -> a: 9

## Generation Examples
- Order-1 top-k sample: 
- Order-3 top-k sample: 

## Perplexity Comparison
- Order-1 train perplexity: 10.5106
- Order-1 same-style perplexity: 12.9558
- Order-1 different-style perplexity: 15.4019
- Order-1 same-style perplexity: 12.9558
- Order-3 same-style perplexity: 20.5901
- Order-5 same-style perplexity: 26.4130

## Order-1 vs Order-3 Discussion
- Higher-order models keep longer local patterns but become much sparser.
- Order-3 used 318 states out of 24389 possible histories.

## Relation to LLMs
- Markov chains only condition on a short fixed history, unlike modern LLMs with much longer context windows.
- Perplexity is still useful as a shared evaluation idea, even though the models are much simpler here.

## Limitations
- Restricted vocabulary removes punctuation and capitalization.
- Character-level outputs are locally plausible but often globally incoherent.
- Higher orders improve short patterns but increase sparsity quickly.
