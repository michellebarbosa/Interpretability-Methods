ELI5 for each Method

LIME: Pokes the model with random variations of your input (delete some words/pixels/features), sees how the prediction changes, then draws a simple straight-line approximation through those results. Fast, intuitive, but the explanation can shift depending on how you randomly sample — run it twice, get slightly different weights.

SHAP: Based on a math concept from game theory (Shapley values) — think of each feature as a "player" in a team, and SHAP fairly divides credit for the prediction among all of them by checking every possible combination of features being present/absent. Slower, but more consistent and theoretically guaranteed to add up correctly (all feature contributions sum exactly to the prediction).

Integrated Gradients: looks inside the model. Uses backpropagation to compute actual gradients — how sensitive the output is to each pixel — at many points along a path from a blank image to the real one, then sums them up.