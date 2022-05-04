import numpy as np
import matplotlib.pyplot as plt

##------- Heavyside function -------##
def sigmoid(zk):
    # zk = W*xk
    return 1/(1 + np. exp(-zk))
##----------------------------------##

