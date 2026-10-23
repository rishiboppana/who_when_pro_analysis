# Index check: does the labelled agent speak at the labelled event?

| dataset/framework | n | agent at step | agent at step-1 | agent at step+1 |
|---|---|---|---|---|
| WW_AG | 126 | 123/126 | 22/106 | 17/125 |
| WW_HC | 58 | 56/58 | 20/58 | 16/51 |
| WWPRO/macnet | 261 | 261/261 | 0/109 | 109/261 |
| WWPRO/magentic-one | 811 | 811/811 | 0/722 | 0/357 |
| WWPRO/mathchat | 920 | 920/920 | 0/920 | 0/195 |
| WWPRO/metagpt | 81 | 81/81 | 0/14 | 0/81 |

A framework is kept only if 'agent at step' is clearly the best offset (decided by the user at STOP 1).
