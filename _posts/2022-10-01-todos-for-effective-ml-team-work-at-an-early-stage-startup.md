---
layout: post
title: TODOs for Effective ML teamwork at an early-stage startup
description: My take on things to do for effective team work for solving Machine Learning problems at an early stage company.
summary: Don't create APIs for ML, just copy&paste. Test every line. If you aren't sure about the design, test first. Always keep your experiments reproducible (lineage, data, code, baseline). Document everything. Be clear, and avoid abbreviations.
tags: tts machine-learning
minute: 7
---

TLDL,

- Don't create APIs for ML, just copy&paste.
- Test every line.
- If you aren't sure about the design, test first.
- Always keep your experiments reproducible (lineage, data, code, baseline).
- Document everything. Be clear, and avoid abbreviations.

### Copy & paste ML, don't abstract
Abstracting ML code sacrifices expressiveness, increases coupling, and aggravates maintenance. These might be ok for regular software. But things are different for ML. I am sure you know how it feels to waste hours trying to match the API when you want to implement an ML trick. I know your pain 😄. APIs and abstractions are bad for fast-paced ML R&D.

ML is too fast, and any API is outdated from its inception. We see a similar pattern with well-known ML libraries (Transition from Theano -> Tensorflow -> PyTorch -> JAX...). It is not only the engineering; also, model architectures are swiftly changing. There are new layers, and new attention methods every day. In an API, every new paper would make a new argument, new config field, or if...else statement. It makes the code overly complicated without realizing it.

It also makes research difficult. When you need to try something new, you don't only worry about the core performance metric, but you need to be compatible with an API and make sure you don't break anything. It introduces complexity and cognitive load that you don't want in a functional team. You might find your team discussing the API more than the problem they need to solve. Worse, they might also get lazy to deal with compatibility issues and avoid trying new ideas they'd try otherwise.

That said, I am not advocating an absolute no abstraction policy (As always, gray is a better color). In my experience, what works best is you define abstractions over fundamental components, define how they should interact/communicate, and let people implement each one by only focusing on the problem at hand. For instance, if you have a model code that needs to be exported and deployed, there are three main components. Model implementation, exporter, deployer. As long as we precisely define how the model should input the exporter and what the exported outputs for the deployer are, we are free to implement each one in isolation. However, in this case, we must heavily test our code. So, again, we need to TEST the code!!

Another problem with abstraction is the model life cycle. Let's say you have a core library shared by your models. You also have N different models that are
currently running. You implement a new model and update your library from v1 to v2. Then you deploy that model using v2, but how do you keep it compatible with
all the previous models? What if the new model has some breaking changes? Either add if...else that would keep things consistent but stay there forever or package the model with the correct version of your library that would make reproducibility almost impossible after a while. The first option would make your code a lot longer for no reason and makes things harder to test, understand and reproduce. The second option defies the whole purpose of a library. What is the point of a library if I need to run a different version/abstraction of it every time?

@[🐸Coqui](https://coqui.ai), we experienced all these problems and started transitioning to the following internally and with our open-source code. First, we define fundamental components; data loader, model implementation, model exporter, model predictor, model service, etc. Then, we determine precisely what each part inputs/outputs. We only create a shared API for the data loader since it is essential to optimize demanding IO operations. Each model implementation is a single python file, including all the steps of that model from inception to having the final working checkpoint (definition, training, prediction, logging). We copy and paste when we need to reuse something from a different model and make changes freely without worrying about compatibility.

This brute approach solves all the issues I mentioned. Our research team can solely work on improving the model performance, share/reproduce experiments quickly, and communicate efficiently. Each model life-cycle is independent, starting from research to deployment. We can easily stage/drop models. Efficiently optimize each model for deployment with model-specific tricks. Most importantly, we focus on real problems, not problems caused by problems. But we test!!


### Test every line
Testing is tedious but also a must, especially for code that goes into production. Therefore, yes, we C&P code and prefer less abstraction, but we unit test every line of the code and ensure every function works as intended and every layer inputs and outputs correctly with the right shape and values. We also run training steps for models and try to overfit a small batch of instances.

All these tests are not only crucial for deployment, but they also ensure that your research is on track. You try new ideas and methods without worrying about the legacy code. I am sure you also experienced that you tried a recent paper that didn't work. After a couple of weeks, you realized that it was not the paper but an older bug in the code. "Test every line" mentality helps alleviate that.

Testing for better design. Sometimes, I need to implement something, but I don't know how. One way I found helpful (I am sure there is already a name for it) is writing the tests first (Maybe test-driven development?). It helps you think like a user and see like a user. For instance, [🐸 TTS's](https://github.com/coqui-ai/TTS) API was garbage initially since I did not use my code but mainly developed it. Increasing the level of testing made me see the important factors and adapt the API accordingly, which led to a nice bump in Github stars. (It is now not perfect but certainly better.)

Testing for bugs. A bug can be in the data pipeline, model implementation, or deployment backend. Securing the data pipeline from bugs is vital since problems here permeate the whole ML cycle. Model bugs are hard to figure out. They might require domain knowledge and experience with the model architecture. Even if you test everything, I think you always find new bugs in your models. It is essential not to ignore these bugs and create new test cases that cover them, not only for that model but for all the models that might have the same bug. Deployment tests are probably easier but less accessible since we don't interact with the deployment code as frequently as the other components. You must monitor your model and set alarms for specific metric shifts in the best scenario. But it is not always possible, especially in the early stages, when you are about to get your MVP out. In that case, you should have a set of inputs and outputs that check extreme conditions and runtime measures.

Testing versions. Most ML code depends on libraries. Those libs get updated, and things change without notice,  break your code. It is an ideal practice to dock on specific versions, but it doesn't save you once and for all. For instance, you use a particular library that runs on python 3.6, but you need to migrate your code to python 3.10. You need to upgrade the library for that. So how do we ensure that the new version of the library does not break our code? The answer is easy. You need to test.

Testing saves you time. Although testing seems time-consuming, I think it saves more time than it consumes in the long run. It gets more apparent when you start working on more complex ML systems. Fixing bugs, implementing new ideas and models, adding new data resources, etc., gets to be more manageable.

In our team, we test every function and layer of model implementations. If we find a new bug in a model, we cover it directly in tests in all the models. We run the most intensive tests for the code shared among models since anything busted there creates the biggest buzz. For model export and deployment, we test, at the very least, extreme conditions and pass samples from which we know what to expect. And use a tool that helps us monitor important performance and runtime metrics.

I know testing is tedious. But the world functions thanks to tested software :).


### Document, document, document...

Communication in a team. In ML, we have different problems to solve with many possible solutions. In general, among all possible solutions, we select one and push it forward. But it leads to a situation where only a subset of people (generally one person) really know what the winning solution really is. It is a problem. How can you work effectively as a team if all the people who need to know it don't know it? So we need to document our work and document it as extensively and redundantly as possible. As the person who implemented it, anything that looks redundant to you might be critical for others to understand or use.

Remote work and time zones. [🐸Coqui](https://coqui.ai) is a fully remote company. We work from different timezones and places. It makes documentation more crucial for effective team work. If something is not documented, the team might waste 24 hours just to communicate that piece. 24 hours is 6 months for an early-stage startup. To save that time, we need to write good documentation.

Writing is not enough. We should also make it visible and reachable. It is useless if there are docs, but they are not accessible at the right time. I think there is no universal solution for documentation. Still, I suggest talking with the team and figuring your way out together as incrementally creating a system that works for you.

Things you need to document. It might sound extreme, but I'd say document everything; code, paper, meeting, discussion, thought, company, ideas (successful or unsuccessful). Some experienced devs do not like commenting on the code because, they say, code is self-descriptive. I think we should comment code because it helps regardless. It gets you onboard quickly, helps you understand polylithic things faster, and doesn't assume mastery. Also, in most cases, you need to see documentation when you are writing code. But, then, what is more, easier than seeing the docs in place. So comment on your code, please.

Communication between teams. There are different teams for different things that need to communicate. A critical part of that communication is documentation, especially with growing team sizes & numbers. It may not seem essential for early-stage companies, but something that is not documented now will never be documented. Thus, starting with docs from day one is something you should consider. Along the same lines of the API discussion above, I think we should consider teams like different fundamental components of your pipeline and define the I/O relation. Then, extensively document how and whats of this I/O.

Communicating with users. Users need docs and probably writing for users is the most challenging task since probably everyone at your early-stage startup is too technical to be a layman user. I also think writing docs for users is an art form. You must be aware that people don't read docs they view. And things that solve their immediate problem should be viewable, recognizable, and actionable. For instance, users of a library or API should immediately recognize what is needed to be copy&pasted to make things work. If you don't have someone who professionally writes user docs, then I think the best approach is to wait for people to ask something and convert your answers to docs like fixing bugs and flipping them unitests. If you try to predict what you need to write, in general, you write too many things, people totally ignore and rather than dealing with its cognitive load, they ask you directly. In that case, you need to replicate your work, thinking RTFD. If you think RTFD always, then you better make your docs more effective.

With my team at [🐸Coqui](https://coqui.ai), we use GitHub issues for all the documentation work. We post in separate "issues" for things that are part of a process and recently started to use the Wiki for shared content. The good thing about using issues is they are referable and linkable from everywhere; code, planning dashboard, releases, etc. I should also say it is cheaper. For user docs of our open-source code, we do what everyone does and use sphinx. But I'm also curious to try the [FastAI way](https://www.fast.ai/posts/2019-11-27-nbdev.html), where you do it all in the Notebooks.


### Keep things reproducible

ML teams experiment with different models and ideas. Each experiment takes time and resources. Therefore, it is crucial to make these experiments sharable and reproducible for effective collaboration for constant improvement.

Reproducibility means, in our terms, producing the exact same outputs every time using the exact same inputs. It is only possible when the same **code** is run on the same **dataset** with the same experiment **environment** (software & hardware).

"Code" means to be able to share the experiment code with all the changes that are done with the experiment. One way to achieve this is to ensure you have no uncommitted changes before running the experiment. Although it sounds reasonable, it is hard to do in practice. Therefore, I suggest logging uncommitted changes in the experiment so you are aware of those in the worst case. It helps to use an experiment tracker that does it automatically.

"Dataset" means using the exact same version of a dataset in the experiment and loading samples in the same order. For instance, to reproduce a training run, creating the same data batch for each step as in the original experiment is essential. It is pretty likely that based on the randomization of the samples, you can get a model that performs better| or worse. "Dataset" is also crucial for debugging because it might cause an error at any random step of training, and such things are difficult to unfold.

"Environment" means using the same deterministic environment (software & hardware). The environment comprises libraries, programming language, and hardware. We need to log all these in a way that is recreatable. In the software, one important trick is to ensure deterministic model runs. For that, we need to use the same random seed and prevent specific randomized actions of the underlying code (CUDA, Torch, TF, Numpy, etc.). For logging the environment, experiment trackers can do that for you, and I suggest using one of them (e.g., ClearML, MLFlow).

(For deterministic execution, here is a segment of code you can use with PyTorch training.)

```python
random.seed(training_seed)
os.environ["PYTHONHASHSEED"] = str(training_seed)
np.random.seed(training_seed)
torch.manual_seed(training_seed)
torch.cuda.manual_seed(training_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

At [🐸Coqui](https://coqui.ai), we use ClearML (open-source) for experiment tracking that log all the necessary details. It even helps for rerunning experiments with a single click. We also use our own open-source PyTorch based [👟Trainer](https://github.com/coqui-ai/Trainer). It is simple, optimized for performance, experiment logging, and reproducibility. It ticks all the boxes for us.

### Final words

There is no specific practice for ML teams as widespread as regular Software teams. I think it is partially because of the unripe MLOps tools and practices. With better tools, software, and practices, it gets easier for teams to be more efficient. So keep an eye out and watch what is happening in the field.

Before I finish, I also thank my team @🐸 (Aya, Edresson, Logan, Julian). You are the best!!

Finally, most of the content reflects my experience at [🐸Coqui](https://coqui.ai) and open-source development (🐸TTS). I just wanted to share what has worked for me and us. Let me know your comments.

### Links

- 🐸[Coqui.ai](https://coqui.ai): Best TTS and Voice Cloning in town
- 🐸[TTS](https://github.com/coqui-ai/TTS): Our open-source TTS library.
- 👟[Trainer](https://github.com/coqui-ai/Trainer): A DL trainer based on PyTorch that we implement most practices above.

Here are some links I use for learning about MLOps and ML  practices; (Let me know if you have some other gems. )

- [MLOps Community](https://mlops.community/)
- [FastAI Blog](https://www.fast.ai/)
- [Coursera](https://www.coursera.org/lecture/machine-learning-business-professionals/build-successful-ml-teams-EP7xF)
