from typing import List, Dict, Optional

class FewShot:
    def __init__(self, prb: str, outcome: str, sol: Optional[str] = None):
        self.prb = prb
        self.outcome = outcome
        self.sol = sol

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

CoTFewShotAnswerSamples = [
    FewShot(
        prb = "What is $\left(\\frac{7}{8}\\right)^3 \cdot \left(\\frac{7}{8}\\right)^{-3}$?",
        outcome = "$1$",
        sol = "Let's think about this step by step. We have the left part of the product raised to the power of 3, and the right part raised to the power of -3. Raising x to a negative power is the same as the 1/x, so we have a product of x and 1/x, which equals 1.",
    ),
    FewShot(
        prb = "In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?",
        outcome = "$15$",
        sol = "Let's think about this step by step. This is a simple combinations problem, and the choice is 6 choose 4, which equals 15.",
    ),
    FewShot(
        prb = "Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$",
        outcome = "$\sqrt{59}$",
        sol = "Let's think about this step by step. We need to find the distance in 3D space, which is equal to the square of difference in each coordinate, summed, and then sqrt of that. In this case, that is sqrt of (3^2 + 7^2 + (-1)^2) which is sqrt of 9 + 49 + 1, which equals \sqrt{59}.",
    ),
    FewShot(
        prb = "The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?",
        outcome = "$\\frac{1}{32}$",
        sol = "Let's think about this step by step. 15 can be made using 7 and 8 rolled. There are 8 times 8 ways of rolling the dies of which only two end up with a 7 and 8 on the face. So the probability is 2/64 which is \frac{1}{32}.",
    ),
    FewShot(
        prb = "The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?",
        outcome = "$181$",
        sol = "Let's think about this step by step. This is an arithmetic sequence with difference 9, so its nth term is given by 1+9(n-1), so its 21st terms is 1+9x20 = 181.",
    ),
    FewShot(
        prb = "Calculate $6 \\cdot 8\\frac{1}{3}",
        outcome = "$16$",
        sol = "Let's think about this step by step. This equals 6 times 8 divided by 3, which is 2 times 8, or 16.",
    ),
    FewShot(
        prb = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?",
        outcome = "$2$",
        sol = "Let's think about this step by step. None of the bits after the 3rd matter in this case, so the binary remainder is 10_2, which is 2 in decimal.",
    ),
    FewShot(
        prb = "How many zeros are at the end of the product 25 $\\times$ 240?",
        outcome = "$3$",
        sol = "Let's think about this step by step. 25 is 100 by 4, and 4 cleanly divides 24, so two zeros from 100 and one from the other number give us 3 zeros.",
    ),
]
