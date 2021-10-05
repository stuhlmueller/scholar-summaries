abstract_to_claims = """
Accurately list all of the conclusions of the following studies:

Abstract of study 1: "Background and aims: Creatine is a supplement used by sportsmen to increase athletic performance by improving energy supply to muscle tissues. It is also an essential brain compound and some hypothesize that it aids cognition by improving energy supply and neuroprotection. The aim of this systematic review is to investigate the effects of oral creatine administration on cognitive function in healthy individuals. Methods: A search of multiple electronic databases was performed for the identification of randomized clinical trials (RCTs) examining the cognitive effects of oral creatine supplementation in healthy individuals. Results: Six studies (281 individuals) met our inclusion criteria. Generally, there was evidence that short term memory and intelligence/reasoning may be improved by creatine administration. Regarding other cognitive domains, such as long‐term memory, spatial memory, memory scanning, attention, executive function, response inhibition, word fluency, reaction time and mental fatigue, the results were conflicting. Performance on cognitive tasks stayed unchanged in young individuals. Vegetarians responded better than meat‐eaters in memory tasks but for other cognitive domains no differences were observed. Conclusions: Oral creatine administration may improve short‐term memory and intelligence/reasoning of healthy individuals but its effect on other cognitive domains remains unclear. Findings suggest potential benefit for aging and stressed individuals. Since creatine is safe, future studies should include larger sample sizes. It is imperative that creatine should be tested on patients with dementias or cognitive impairment. HIGHLIGHTSOral creatine supplementation improves memory of healthy adults.Findings suggest potential benefit for aging and stressed individuals.Future trials should investigate the effect of creatine administration on individuals with dementia or mild cognitive impairment."

Conclusions of study 1 (one sentence each):
- Short term memory and intelligence/reasoning may be improved by creatine administration
- Regarding other cognitive domains, such as long‐term memory, spatial memory, memory scanning, attention, executive function, response inhibition, word fluency, reaction time and mental fatigue, the results of creatine administration were conflicting
- Performance on cognitive tasks stayed unchanged in young individuals after creatine administration
- Vegetarians responded better than meat‐eaters in memory tasks but for other cognitive domains no differences were observed after creatine administration
- Oral creatine administration may improve short‐term memory and intelligence/reasoning of healthy individuals but its effect on other cognitive domains remains unclear
- Creatine has potential benefits for aging and stressed individuals
- Creatine should be tested on patients with dementias or cognitive impairment
- Oral creatine supplementation improves memory of healthy adults
- Findings suggest potential benefit of creatine administration for aging and stressed individuals
- Future trials should investigate the effect of creatine administration on individuals with dementia or mild cognitive impairment

Abstract of study 2: "{text}"

Conclusions of study 2 (one sentence each):
-""".strip()

answer_to_question = """For each answer state exactly the question that it answers:

Answer: Oral creatine administration may improve short‐term memory and intelligence/reasoning of healthy individuals but its effect on other cognitive domains remains unclear
Question: Does oral creatine administration improve memory and intelligence?

====

Answer: Resource management techniques determine which modules of analytics applications are pushed to each edge device to minimize the latency and maximize the throughput
Question: What techniques are used to manage resources of analytics applications?

====

Answer: Forward inference techniques such as sequential Monte Carlo and particle Markov chain Monte Carlo for probabilistic programming can be implemented in any programming language by creative use of standardized operating system functionality including processes, forking, mutexes, and shared memory
Question: How can I implement forward inference techniques such as SMC and particle MCMC?

====

Answer: {answer}
Question:"""

answer_to_subquestion = """What is the more specific question that each of the following answers is referring to:

General question: What are the effects of creatine on cognition?
Answer: Oral creatine administration may improve short‐term memory and intelligence/reasoning of healthy individuals but its effect on other cognitive domains remains unclear
More specific question: Does oral creatine administration improve memory and intelligence?

====

General question: What are the most effective management techniques?
Answer: Resource management techniques determine which modules of analytics applications are pushed to each edge device to minimize the latency and maximize the throughput
More specific question: What techniques are used to manage resources of analytics applications?

====

General question: How can I implement a probabilistic programming language?
Answer: Forward inference techniques such as sequential Monte Carlo and particle Markov chain Monte Carlo for probabilistic programming can be implemented in any programming language by creative use of standardized operating system functionality including processes, forking, mutexes, and shared memory
More specific question: How can I implement forward inference techniques such as SMC and particle MCMC?

====

General question: {question}
Answer: {answer}
More specific question:"""
