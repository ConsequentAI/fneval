from typing import List, Dict

class FewShot:
    def __init__(self, prb, outcome):
        self.prb = prb
        self.outcome = outcome

FewShotAnswerSamples = [
    FewShot(
        prb = "What is $\left(\\frac{7}{8}\\right)^3 \cdot \left(\\frac{7}{8}\\right)^{-3}$?",
        outcome = "$1$",
    ),
    FewShot(
        prb = "In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?",
        outcome = "$15$",
    ),
    FewShot(
        prb = "Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$",
        outcome = "$\sqrt{59}$",
    ),
    FewShot(
        prb = "The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?",
        outcome = "$\\frac{1}{32}$",
    ),
    FewShot(
        prb = "The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?",
        outcome = "$181$",
    ),
    FewShot(
        prb = "Calculate $6 \\cdot 8\\frac{1}{3}",
        outcome = "$16$",
    ),
    FewShot(
        prb = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?",
        outcome = "$2$",
    ),
    FewShot(
        prb = "How many zeros are at the end of the product 25 $\\times$ 240?",
        outcome = "$3$",
    ),
]
