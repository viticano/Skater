**********
Overview
**********

'''''''''''''''''''''''''''''


skate
~~~~~~~~~~~~~~~~

skate is a python package for model agnostic interpretation of predictive models. With skate, you can unpack the internal
mechanics of arbitrary models; as long as you can obtain inputs, and use a function to obtain outputs, you can use skate to learn
about the models internal decision criteria.

So that models produced by different learning algorithms, implementations, and environments can be compared, the skate philosophy
is that all models should be evaluated as black boxes; decision criteria are inferred based on input perturbations and observing
corresponding outputs.

Interpretation algorithms for all but the most trivial models must make some simplifications
when representing a model's decision criteria. These simplifications can be with respect to:

1. `feature scope`: the number of features for which we explain model behavior at a time.
2. `domain`: the region of the input space for which explain model behavior
3. `detail/fidelity`: the level of aggregation performed (across features or the domain)

Partial dependence, marginal plots, and similar explanations generally simplify on feature scope;
they provide faithful approximations on small feature subsets.

LIME and anchor LIME simplify on domain and fidelity, as they provide regressions or trees
respectively, on small regions of the domain.

Feature importance simplifies on detail, as it provides scalars for all features
across the entire domain.
