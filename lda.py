from typing import List
import numpy as np
import random


def lda_gen(vocabulary:List[str], alpha: np.ndarray, beta: np.ndarray, xi: int) -> List[str]:
    topic_proportions = np.random.dirichlet(alpha)
    doc_len = np.random.poisson(xi)
    words = []
    for j in vocabulary:
        topic = np.random.multinomial(doc_len, topic_proportions)
        index = np.where(topic == max(topic))[0][0]
        word_proportions = np.random.multinomial(len(vocabulary), beta[index])
        words.append(random.choices(vocabulary, word_proportions)[0])
    return words