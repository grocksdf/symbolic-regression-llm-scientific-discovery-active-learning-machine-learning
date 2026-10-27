# Localized Gap refinement contract

This is a response-free generation-language and validity contract. It does not
modify historical experiments or authorize a real run.

The LLM supplies only a smooth gate \(g(x)\), a low-complexity correction
\(\Delta(x)\), and a rationale. Code binds protocol identity, round, island,
candidate identity, parent identity and action, then constructs

\[
f_{\mathrm{new}}(x)=f_{\mathrm{parent}}(x)+g(x)\Delta(x).
\]

On the registered covariate domain, \(g(x)\) must be finite and lie in
\([0,1]\). Using no responses, the candidate must satisfy

\[
\operatorname{RMS}_{\mathrm{non-target}}
  (f_{\mathrm{new}}-f_{\mathrm{parent}})
\le
\operatorname{RMS}_{\mathrm{target}}
  (f_{\mathrm{new}}-f_{\mathrm{parent}}),
\]

with strictly positive target-region change. This certificate grants no
epistemic authority. A surviving candidate must still pass independent Global
Protected Admission.

