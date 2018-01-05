chain_probs = []
chains = 2
for c in range(chains):
    # accuracies = train_test()
    # probs = cross_validation(2)
    probs = [0,1,2]
    chain_probs.append(probs)

print(chain_probs)
