---
layout: post
title: Solving an ML Research Problem
description: structured approach to solving a machine learning research problem
tags: machine-learning research problem-solving
minute: 5
---

## **Study the Problem**

- Describe the problem in detail
- Identify inputs and outputs of the system
- Identify the business context and goals.
- Identify the constraints and assumptions.
- Identify the stakeholders and ensure that everyone is on the same page about all the above.

## **Define the Metric**

- Define a single value performance metric for evaluation.
- Ensure that the metric is aligned with your goals.
- When there are multiple values that need to be optimized, two options:
    - (preferred) A composite metric that combines all the values into a single value. Weighted sum where the weights are determined by the importance of the values.
    - A hierarchy of metrics. Pick the most indicative value as your main metric for success. Then incrementally focus on the other values to break ties.

## **Study the Data**

"Data is the Kind of Kings"

- Identify available datasets
- Identify potential data sources - open, internal, 3rd party, web, crowdsourcing, etc.
- Form a clear picture of the data requirements for quality and quantity.
- Consider licenses, ToS, regional regulations, etc.
- **Re-iterate on data regularly** as you know more through the project.
- Split your dataset into training, validation and test sets. (and a comparison set but more on that later)
- Ensure that the validation and test sets are representative.

## **List Solutions**

- Find and list possible methods to solve the problem.
- List the pros and cons of each method and be specific.
- Order the list based on Return on Investment (ROI).
- Trust your intuition and experience. If you don't have it, get it.
- You need creativity, intuition, experience, and domain knowledge.

**Note:** Read and think a lot, be a researcher. Build the expert intuition. I'm sure you get stuck at some point without it.

## **Create Evaluation Pipeline**

ATM we know the literature, possible solutions, the data, and the metric. Now we need to create a pipeline to evaluate the solutions.

We do it before we start the experiments with a clear outlook on the problem. We also don't want to be biased with the effort we put later on to the experiments.

- Implement the evaluation pipeline.
- Make sure that the pipeline is fast, precise, fair, and bug-free.
- Consider updating the pipeline if it was created earlier by contrasting it with the current knowledge.
- Go over the pipeline with the team and stakeholders.

## **Establish Baseline**

- Alternatives for a baseline ranked by preference
    - Use the currently deployed model
    - Use the SOTA model
    - Use the SOTA paper metrics (only good for greenfield projects)
- Compute evaluation metrics on your test set
- Assess clearly what indicates that you beat the baseline

**Do not trust papers (sorry).** Values in papers are mostly misleading in practical cases.

## **Pick the Best ROI Solution**

- Review the list of potential solutions and their pros/cons
- Assess each solution based on expected performance and implementation cost
- Consider both short-term gains and long-term potential
- Select the solution with the highest return on investment (ROI)
- Justify your choice to stakeholders and the team

## **Experiment**

- Implement the highest ROI solution
- Test as much as possible.
    - ML people don't like testing but a simple test might save you weeks if not months.
- Be clear about the inputs, outputs, and expected results of the experiment.
- Log as much as possible. Performance metrics, runtime metrics, data metrics, visualizations, etc.
- Keep your experiments reproducible
    - Save artifacts; model checkpoints, training, and model configs.
    - Set the experiment seed for numerical consistency between different runs.
    - Save environment setup information (CUDA version, python modules, …)
    - Create a code base that enables reproducible runs with ease.
    - Make sure your data loader works deterministically based on the seed.
- Debug thoroughly if anything seems suspicious.
    - You'll be surprised by the bugs and problems you detect this way.
- Vary key parameters and collect performance data.
- Use the evaluation pipeline to assess the results
- Document all experiments, including failures
- Compare results against the baseline to see where you are, previous runs to see how to proceed, and project goals to see whether you are on the right track.

**Using a comparison split.** It is hard to compare successive training runs when data collection happens in parallel. Create a specific data split that stays constant (**comparison split**). Run validation on this split alongside your validation set. It'd help you compare models fairly and justify what works as you progress. Change the comparison set when the validation curve and comparison curve diverge which signals a change in data distribution. And you need to run all your previous compared models on this new set.

**Don't be obsessed with a solution.** Our goal is solving the problem, not making your favorite solution work.

**YOLO runs with big models with big changes.** Sometimes you can't go incremental because of budget, date, time, etc. In my experience, success of YOLO runs correlated with the experience and programming skills of the person.

**Codebase.** Make sure that you have a codebase that is modular, easy, and robust enough to perform the tasks above.

**Automate as much as possible.** We are lazy and we get lazier as we spend more time on a project. Therefore, automate any step you take as soon as possible to reduce the cognitive load of repetitive TODOs.

## **Iterate**

- Analyze experimental results to identify areas for improvement
- Brainstorm modifications or new approaches based on insights
- Implement promising changes and re-run experiments
- Continuously refine the model and approach
- Regularly update stakeholders on progress and findings
- Be prepared to pivot if the current approach is not giving the desired results
- Consider when to stop iterating based on diminishing returns or project constraints

**Always think about data first before changing your model.** Getting more high-quality data is the most effective way to improve performance unless the limitation is inherently about model capacity.
