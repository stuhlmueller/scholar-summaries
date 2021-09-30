conclusions_prompt = """
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
