# Protected LLM-SR search adapter

The repository already vendors the official LLM-SR implementation. Its
multi-island experience buffer derives from the Apache-2.0 DeepMind
FunSearch implementation.

The protected adapter reuses the official tempered-softmax sampling function
and the multi-island/cluster/reset design, but does not reuse the original
full-training-response evaluator or final best-program selector.

Only development-stage formula identities, scalar scores and score signatures
may enter the adapter. Admission, posterior-update, reporting, test/OOD and
held-out response values are structurally absent.

Candidates sampled from the buffer receive no epistemic authority. They must
still pass all-domain definedness, Global Protected Admission and the
independent posterior-update transaction.

