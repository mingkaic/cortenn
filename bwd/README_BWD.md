# Backwards (BWD)

Backwards implements a Grader ADE graph traveler to generate partial
derivatives using Automatic Differentiation (AD).

Every Grader requires an instance of iRuleSet. Ruleset defines manditory
opcodes for sum and production operations as well as chain rules for each
operation
