# Admit-then-independent-update transaction

The transaction uses four ordered, disjoint development roles:

1. fit: freezes the original Bank posterior;
2. Gap audit: generates the search diagnostic;
3. Admission: decides whether a candidate structure may enter the Bank;
4. posterior update: updates the expanded Bank after admission.

Admission responses are not reused for posterior weighting. The posterior
update role is opened only after a candidate is admitted. Reporting remains a
fifth independent role.

For an expanded Bank \(H^+=H_0\cup\{h\}\),

\[
p(H^+\mid D_{\rm fit},D_{\rm update})
\propto
p(H^+\mid D_{\rm fit})p(D_{\rm update}\mid H^+,D_{\rm fit}).
\]

No admitted candidate, crossed role identity, empty update role, test/OOD, or
held-out input is accepted. This is correctness only and does not authorize a
real experiment.

