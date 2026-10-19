# Research differentiation — Round 1

Sources inspected on 2026-10-18. This is a focused design review, not a systematic
literature survey and not a novelty claim. No external implementation was copied.

| Primary source | Relevant mechanism | Implication for AXIOM |
|---|---|---|
| [Reflexion paper](https://arxiv.org/abs/2303.11366) and [official repository](https://github.com/noahshinn/reflexion) | Verbal feedback stored in episodic memory, explicitly without weight updates. | A text memory baseline is essential. A parameter change alone is also insufficient: test changed choices and outcomes. |
| [Voyager paper](https://arxiv.org/abs/2305.16291), [official skill manager source](https://github.com/MineDojo/Voyager/blob/main/voyager/agents/skill.py) | Growing executable skill library and embedding retrieval. Source uses OpenAI embeddings and Chroma; stored skill names index program descriptions. | Reuse the separation of skills and retrieval as a design idea, but keep versions explicitly and train a small local selector. Do not equate library growth with parameter learning. |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | Experience records, retrieval, reflection and planning for believable behavior. | Event-backed persistent state is useful engineering; behavioral believability does not establish this project's learning criterion. |
| [Li et al., contextual bandits](https://arxiv.org/abs/1003.0146) | Context-conditioned action choice from observed reward, computationally efficient linear methods. | Use a disjoint-arm LinUCB comparator and shared logged action outcomes. Our exhaustive simulator evaluation is not claimed as unbiased off-policy evaluation from arbitrary production logs. |
| [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682) | Replay helps preserve older behavior under changing distributions. | Include old/new phase probes, replay and exact rollback. A toy menu study cannot establish long-horizon continual learning. |
| [Avalanche official project](https://github.com/ContinualAI/avalanche) | PyTorch continual-learning research infrastructure. | Relevant for later broad experiments; adding it now would exceed the minimal dependency and scope budget. |

## Narrow proposed contribution

An auditable, provider-independent learning interface: it distinguishes recording,
validation and training, measures ranking changes, and exposes rollback. The present
linear algorithm is standard; the architecture is an engineering hypothesis. Whether
it beats memory retrieval, LinUCB or an MLP is an empirical question.

## Falsification and confounds

1. Parameter freezing must preserve exactly the old digest and ranking for the same
   memory and observation. Otherwise the claimed learning attribution is invalid.
2. Same replay data and test tasks across policies; randomized logged actions avoid
   training AXIOM on more informative outcomes than comparators.
3. Semantic transfer must be reported separately from fixed vocabulary knowledge.
   Unseen names/layouts are not unseen concepts, real software, or natural language.
4. Inquiry returns synthetic, noisy hints at charged costs. This does not establish
   internet search ability or realistic human teaching behavior.
5. Negative transfer and distribution-shift regressions remain in the report.
6. Five independent training seeds give only a preliminary uncertainty estimate;
   episodes from the same learner are not independent experimental replicates.
