import json, glob, os, re, sys, difflib
sys.path.insert(0, "/Users/user/Desktop/thesis/error_localization/prompt_A_benchmark/src")
O="/private/tmp/claude-501/-Users-user-Desktop-thesis-who-and-when-pro-analysis/2a21fd4a-fc52-4e4f-9e6b-a0bbdd9c277a/scratchpad/orig/Who&When/"
L="/Users/user/Desktop/thesis/error_localization/who_and_when_dataset/Who&When/"
def ld(b,s): return {f:json.load(open(b+s+"/"+f)) for f in sorted(os.listdir(b+s), key=lambda x:int(re.sub(r"\D","",x)))}
# 1. label differences repo vs local copy (what the old run scored against)
for s in ["Algorithm-Generated","Hand-Crafted"]:
    a,b=ld(O,s),ld(L,s); diffs=[]
    for f in a:
        for k in ("mistake_agent","mistake_step"):
            if a[f][k]!=b[f][k]: diffs.append((f,k,a[f][k],b[f][k]))
        gt_a=a[f].get("ground_truth", a[f].get("groundtruth")); gt_b=b[f].get("ground_truth", b[f].get("groundtruth"))
        if gt_a!=gt_b: diffs.append((f,"gt",gt_a,gt_b))
        if a[f]["history"]!=b[f]["history"]: diffs.append((f,"history"))
    print(s,"label/gt/history diffs repo vs local:",len(diffs)); print("  ",diffs[:20])
# 2. index convention re-check (repo copy): does history[ms] belong to mistake_agent?
for s,key in [("Algorithm-Generated","name"),("Hand-Crafted","role")]:
    d=ld(O,s); m0=m1=0
    for r in d.values():
        ms=int(r["mistake_step"]); ag=r["mistake_agent"].lower()
        m0+= ag in str(r["history"][ms].get(key,"")).lower()
        m1+= ms>=1 and ag in str(r["history"][ms-1].get(key,"")).lower()
    print(s,"agent matches history[ms]:",m0,"history[ms-1]:",m1,"of",len(d))
# 3. faithfulness: old frozen A.txt rendering vs original all_at_once string
from render import build_prompt_A
def orig_prompt(problem, gt, history, hc):
    ia="role" if hc else "name"
    chat="\n".join(f"{e.get(ia,'Unknown Agent')}: {e.get('content','')}" for e in history)
    return ("You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
      f"The problem is:  {problem}\n" f"The Answer for the problem is: {gt}\n"
      "Identify which agent made an error, at which step, and explain the reason for the error. "
      "Here's the conversation:\n\n"+chat+"\n\nBased on this conversation, please predict the following:\n"
      "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
      "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
      """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
      "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
      "3. The reason for your prediction."
      "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n")
r=ld(O,"Algorithm-Generated")["1.json"]
rec={"problem":r["question"],"ground_truth":r["ground_truth"],"history":r["history"],"subset":"Algorithm-Generated"}
a=build_prompt_A(rec); o=orig_prompt(r["question"],r["ground_truth"],r["history"],False)
print("old A prompt == repo all_at_once prompt:", a==o, "| lengths", len(a), len(o))
sm=difflib.SequenceMatcher(None,a,o); print("char similarity ratio", round(sm.ratio(),4))
pre=lambda s:s.split("Here")[0]; print("prefix up to 'Here' equal:", pre(a)==pre(o)); print(repr(pre(a)[-60:])); print(repr(pre(o)[-60:]))
