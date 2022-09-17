
## My takes after years of ML R&D and OSS dev.

- Don't create API for ML, just copy&paste .
- Test every line
- If you aren’t sure about the design, test first.
- Document everything. Be clear, and avoid abbreviations.
- Always keep your experiments reproducible (lineage, data, code, baseline)

### Copy & paste ML, don't abstract
Abstracting ML code sacrifices expressiveness, increases coupling and aggravates maintenance. These might be ok for regular software. But things are different for ML. I am sure if you'r in ML, you spend hours for a simple change in a library to try a new paper. I feel your pain 😄.

ML is too fast, any API is out-dated from its inception. We see a similar pattern with well-known ML libraries (Transition from Theano -> Tensorflow -> PyTorch -> JAX...). It is not only the engineering also model architectures are changing quite swiftly. There are new layers, new attention methods everyday. In an API, every new method would make a new argument, new config field, new if...else statement. This makes the code overly complicated without realizing.

It also makes research difficualt. When you need to try something new, you don't only worry about the core performance metric but you need to be compatible with an API, make sure you don't break anything. This introduces complexity and cognitive load for people that you don't want in an effective research team. You might find your team discussion the API more than the problem that they really need to solve. They might also get lazy to deal with the compat issues and ignore things they might like to try otherwise.

Being said that, I am not advacating no abstraction at all policy (As always gray is a better color). In my experience, what works the best, you define abstractions over fundamental components and define how these componenet should interact/communicate and let people go implementing each component by only focusing on the problem at hand. For instance, if you have a model code that needs to be exported and deployed there are 3 main components. Model implementation, exporter, deployer. As long as we precisely define how the model should input the exporter, what the exported outputs to the deployer, we are free to implement each one of those as we like. However, in this case, we must heavily test the code. Again TEST the code!!

Another problem with abstraction is model life-cycle. Let's say you have a core library shared by your models. You also have N different models that are
currently running. You implement a new model, update your library from v1 to v2. Then you deploy that model using v2 but then how do you keep it compatible with
all the other previous models. How about if the new model has some breaking changes. You either add if...else that would keep things compatible that stays there forever or package model with the right version of your library that would make reproducibility almost impossible after a while. The first option would make your code a lot longer for no reason and makes things harder to test, understand and reproduce. The second option, defies the whole purpose of a library. What is the point of a library if I need to run a different version of it everytime?

@🐸Coqui we experienced all these problems and start transitioning to the following internally and with our open-source code. We define fundemental components; data loader, model implementation, model exporter, model predictor, model service, etc. We define precisely what each component inputs/outputs. We setup a data loader API for demanding IO operations. Each model implementation is a single python file, including all the steps of that model (definition, training, prediction, logging). We copy and paste when we need to use something already implemented and make changes freely without worrying about compatibility but with extensive unitesting.

This brute approach solves all the issues I mentioned. Our research team can solely work on improving the model performance, share/reproduce experiments easily, communicate efficiently. Each model life-cycle is independed from research to deployment. We can easily stage/drop models. Efficiently optimize each model for deployment with model specific tricks. Most importantly, we focus on real problems not problems causes by problems. But we test!!



Overview of ML code that starts from research to deployment
Data backend
Model implementation
Model training
Model export
Model deployment


### Test every line
Testing is tedious but also a must, especially for code that sees production.