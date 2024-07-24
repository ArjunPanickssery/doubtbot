import json

FILENAME = 'results_v2.json'
LINEGRAPHNAME = 'result_v2.png'
DISTPLOTNAME = 'difference_v2.png'

f = open(FILENAME)
r = json.load(f)
pre = []
post = []

# f2 = open(FILENAME)
# r2 = json.load(f)
for i in range(len(r)):
    pre.append(r[i]['judge_confidence_pre_critique'])
    post.append(r[i]['judge_confidence_post_critique'])

import matplotlib.pyplot as plt


x = list(range(len(pre)))

plt.figure(figsize=(10, 6))
plt.plot(x, pre, label='Pre', marker='o')
plt.plot(x, post, label='Post', marker='s')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Pre vs Post Values')
plt.legend()
plt.grid(True)

plt.savefig(LINEGRAPHNAME)

import seaborn as sns

difference= [pre[i]-post[i] for i in range(len(post))]

plt.figure(figsize=(10, 6))
sns.histplot(difference, kde=True)
plt.title('Distribution of Differences (Post - Pre)')
plt.xlabel('Difference')
plt.ylabel('Frequency')

plt.axvline(x=sum(difference)/len(difference), color='r', linestyle='--')

plt.savefig(DISTPLOTNAME)

