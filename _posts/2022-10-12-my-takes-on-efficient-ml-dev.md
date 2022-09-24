
## My take on effective ML work at a early stage startup

- Don't create API for ML, just copy&paste .
- Test every line
- If you aren't sure about the design, test first.
- Document everything. Be clear, and avoid abbreviations.
- Always keep your experiments reproducible (lineage, data, code, baseline)

### Copy & paste ML, don't abstract
Abstracting ML code sacrifices expressiveness, increases coupling and aggravates maintenance. These might be ok for regular software. But things are different for ML. If you're in ML, I am sure you spend hours for a simple change in a library to try a new paper. I feel your pain 😄.

ML is too fast, and any API is outdated from its inception. We see a similar pattern with well-known ML libraries (Transition from Theano -> Tensorflow -> PyTorch -> JAX...). It is not only the engineering; also, model architectures are changing quite swiftly. There are new layers, new attention methods every day. In an API, every new paper would make a new argument, new config field, and new if...else statement. It makes the code overly complicated without realizing it.

It also makes research difficult. When you need to try something new, you don't only worry about the core performance metric, but you need to be compatible with an API, and make sure you don't break anything. It introduces complexity and cognitive load that you don't want in a functional team. You might find your team discussing the API more than the problem they need to solve. Worse, they might also get lazy to deal with compatibility issues and avoid trying new ideas they'd try otherwise.

That said, I am not advocating no abstraction policy (As always, gray is a better color). In my experience, what works best is you define abstractions over fundamental components, define how they should interact/communicate, and let people implement each one by only focusing on the problem at hand. For instance, if you have a model code that needs to be exported and deployed, there are three main components. Model implementation, exporter, deployer. As long as we precisely define how the model should input the exporter, and what the exported outputs for the deployer are, we are free to implement each one in isolation. However, in this case, we must heavily test our code. So, again, we need to TEST the code!!

Another problem with abstraction is the model life-cycle. Let's say you have a core library shared by your models. You also have N different models that are
currently running. You implement a new model, and update your library from v1 to v2. Then you deploy that model using v2, but how do you keep it compatible with
all the previous models? What if the new model has some breaking changes? Either add if...else that would keep things compatible but stay there forever or package model with the right version of your library that would make reproducibility almost impossible after a while. The first option would make your code a lot longer for no reason and makes things harder to test, understand and reproduce. The second option defies the whole purpose of a library. What is the point of a library if I need to run a different version of it every time?

@🐸Coqui we experienced all these problems and started transitioning to the following internally and with our open-source code. First, we define fundamental components; data loader, model implementation, model exporter, model predictor, model service, etc. Then, we determine precisely what each component inputs/outputs. We only create a shared API for the data loader since it is essential to be optimized for demanding IO operations. Each model implementation is a single python file, including all the steps of that model from inception to having the final working checkpoint (definition, training, prediction, logging). We copy and paste when we need to reuse something from a different model and make changes freely without worrying about compatibility.

This brute approach solves all the issues I mentioned. Our research team can solely work on improving the model performance, share/reproduce experiments quickly, and communicate efficiently. Each model life-cycle is independent, starting from research to deployment. We can easily stage/drop models. Efficiently optimize each model for deployment with model-specific tricks. Most importantly, we focus on real problems, not problems caused by problems. But we test!!


### Test every line
Testing is tedious but also a must, especially for code that goes into production. Therefore, yes, we C&P code and prefer less abstraction but we definetely
unit test everline of the code and make sure every function works as intended, every layer inputs and outputs correctly with the right shape and values. For models
we also run training runs and try to overfit a small batch of instances.

All these tests are not only important for deployment, they also make sure that your research is on track. You try new ideas and methods without worrying about the
legacy code. I am sure, you also experienced that you tried a new paper and it didn't work but after a couple of weeks you realized that it was not the new method that was not working but a older bug that made it ineffective. This kind of extendsive testing alleviates it.

Testing for better design. Sometimes, I need to implement something but I don't know how. So one way I found useful (I am sure there is already a name for it) is writing the tests first. It helps you think like a user and really see how that code would be used and what needed is to make it right. For instance, 🐸 TTS's API was
a garbagge initially since I did not really use my code but mostyle develop. After I start testing, it made me see that kind of user intraction and API are right for the purpose. (It is not perfect now but certainly way better.)

Testing for bugs. A bug can be in the data, model or deployment side of the things. Data is important since any bug permeates along the whole pipeline. Model bugs are hard to figure, requires a lot of domain knowledge and sometime interactive debugging. Even if you test everything, I think you always find bugs. It is important to add cases to your tests that cover new bugs and try to replicate those for other models, tests and components. Deployment tests are probably easier but mostly less transparent. In the best scenario, you need to monitor your model, set alarms against certain unexpected metric changes. But it is not always possible especially in the early stages. In that case, at least you should have a set of inputs, outputs that check extreme conditions and you know what to expect. Then assert that those conditions are always met in your tests.

Testing for version updates. Many ML code is depends on different libraries. They get updated and things can change then break your code. It is of couse a good practise to anchor on specific versions but it doesn't save you in all cases. For instance, you use a specific version of a library that runs on python 3.6 but you need to migrate to python 3.10 and you need to upgrade the library for that. How we can make sure that the new version of that library does not break our code. Answer is easy. Testing.

In our team, we create test classes that match all the functions of the model code. As I said above, we try to keep all the code in a single file that releates to a model that solves a particular problem. So we create a clone test file that match every line of that model file. Also, if we find a new bad in a model, we directly think about other models that might be effected and implement new tests not for that model only but all the other as well. For the code shared among models, we run the most intensive tests since anything broken there would damage the most. For model export and deployment we test extreme conditions and pass samples that we know what to expect as output. More advance monitoring is on its way too.

Testing is boring. We don't like it. But the world functions by the account of heros who test code :).


### Document, document, document...

Communication in a team. In ML we have different problems to solve by many possible solutions. In general, among all possible solutions we select one, and push it forward. But it leads to a situation that only subset of people (generally one person) who really know what the winner solution reall is. It is a problem. How can you work effectively as a team if all the people who need to know it don't know it. We need ot document our work and document it as extensively and redudantly as possible. Anything that looks redudant to you, as the person who created the solution, might be key for others to understand or use it.

Remote work and timezones. 🐸Coqui is a fully remote company. We work from different timezones and places. It makes documentation more crucial for an effective team work. If there is something not documented, team might waste 24 hours just for communicating the right piece. 24 hours is 6 months for an early-stage startup. To save that time, we need to write good documentation.

Writing is not enough. We should also make is visible and reachable. If there is docs but they are not reachable at the right time, it is useless. I think there is no one solution for that but I suggest talking with the team and figure your way out together as incrementally creating a system that works for everyone.

Things you need to document. It might sound extreme but document everything; code, paper, meeting, discussion, thought, company, ideas (that work or not work). Some adept devs do not like commenting code because code is self-descriptive. I think we should comment code because it makes onboarding easier, searching easier, automating documentation easier and in most cases you need documentation when you are writing code then what is more easier then cheking the docs in the same place. I am also not good at understading my code after some while. So comments let me warm up easier when I need to revisit the code. So comment you code please.

Communication between teams. There are different teams for different things and what makes inter-team work functional is documentation. I think we should consider teams like different fundemantal components of your ML pipeline and define the I/O relation. Then, extensively document the I/O relation.

Communicating with users. Users need docs too and probably writing for user is the hardest task since probably everyone at your early-stage startup is too technical to be a layman user. Therefore, I think writing docs for users is a form of art. You need to be aware that people don't read docs they view it. And things that solve their immediate problem should be viewable, recognizable and actionable. For instance, if you have open-source lib. users, they should immediately recognize what is needed to be copy&paste to make things work. If you don't have someone who profesionally write user docs, then I think the best approach is to wait people ask things and convert your answers to docs like fixing bugs and converting them unitests above. If you try to predict what you need to write, in general you write too many things, people totally ignore and rather than dealing with its cognitive load, they ask you directly in an issue. In that case, you need to replicate your work and thinking RTFD. If you think RTFD always then you better fidn ways to make your docs more effective.

With my team at 🐸Coqui, we use Github issues for all the documenation work. We post in separate issues for things that are part of a process and use the Wiki for common content. The good think about using issues is they are refereable and linkable from everywhere of our work; code, planing dashboard, releases, etc. I should also say, it is cheaper. For user docs of our open-source code, we do what all people do and use sphinx. But I'm also curious to try the [FastAI way](https://www.fast.ai/posts/2019-11-27-nbdev.html) where you do it all in the Notebooks.


### Keep things easily producible
