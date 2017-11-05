import numpy as np
test_doc_authors = [0,1,2]
author_guess = [0,1,3]

print(np.mean([1 if x==y else 0 for (x,y) in list(zip(test_doc_authors,author_guess))]))
print(np.equal(test_doc_authors,author_guess).mean())
