---
layout: post
title: "Brief History of Machine Learning"
description: "!My subjective ML timeline(https://web"
tags: 
minute: 12
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

[![My subjective ML timeline](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/test.jpg)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/test.jpg)

My subjective ML timeline (click for larger)

Since the initial standpoint of science, technology and AI, scientists following Blaise Pascal and Von Leibniz ponder about a machine that is intellectually capable as much as humans. Famous writers like Jules  
Verne , Frank Baum (Wizard of OZ), Marry Shelly (Frankenstein), George Lucas (Star Wars) dreamed artificial beings resembling human behaviors or even more, swamp humanized skills in different contexts.

[![Pascal's machine performing subtraction and summation - 1642](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/Arts_et_Metiers_Pascaline_dsc03869.jpg)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/Arts_et_Metiers_Pascaline_dsc03869.jpg)

Pascal's machine performing subtraction and summation - 1642

Machine Learning is one of the important lanes of AI which is very spicy hot subject in the research or industry. Companies, universities devote many resources to advance their knowledge. Recent advances in the field propel very solid results for different tasks, comparable to human performance (98.98% at [Traffic Signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=results&subsubsection=ijcnn) - higher than human-).

Here I would like to share a crude timeline of Machine Learning and sign some of the milestones by no means complete. In addition, you should add "up to my knowledge" to beginning of any argument in the text.

First step toward prevalent ML was proposed by **Hebb**, in 1949, based on a neuropsychological learning formulation. It is called **Hebbian Learning** theory. With a simple explanation, it pursues correlations between nodes of a Recurrent Neural Network (RNN). It memorizes any commonalities on the network and serves like a memory later. Formally, the argument states that;

> Let us assume that the persistence or repetition of a reverberatory activity (or "trace") tends to induce lasting cellular changes that add to its stability.… When an [axon](http://en.wikipedia.org/wiki/Axon "Axon") of cell *A* is near enough to excite a cell *B* and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that *A'*s efficiency, as one of the cells firing *B*, is increased.[1]

![us__en_us__ibm100__700_series__checkers__620x350](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/11/us__en_us__ibm100__700_series__checkers__620x350.jpg)

Arthur Samuel

**In 1952**, **Arthur Samuel** at IBM, developed a program playing **Checkers**. The program was able to observe positions and learn a implicit model that gives better moves for the latter cases. Samuel played so many games with the program and observed that the program was able to play better in the course of time.

With that program Samuel confuted the general providence dictating machines cannot go beyond the written codes and learn patterns like human-beings. He coined “machine learning, ” which he defines as;

> a field of study that gives computer the ability without being explicitly programmed.

![](http://csis.pace.edu/~ctappert/srd2011/photos/Rosenblatt-ratlab.jpg)

F. Rosenblatt

**In 1957**, **Rosenblatt's** **Perceptron** was the second model proposed again with neuroscientific background and it is more similar to today's ML models. It was a very exciting discovery at the time and it was practically more applicable than Hebbian's idea. Rosenblatt introduced the Perceptron with the following lines;

> The perceptron is designed to illustrate some of the fundamental properties of intelligent systems in general, without becoming too deeply enmeshed in the special, and frequently unknown, conditions which hold for particular biological organisms.[2]

After 3 years later, **Widrow [4]** engraved **Delta Learning rule** that is then used as practical procedure for Perceptron training. It is also known as **Least Square** problem. Combination of those two ideas creates a good linear classifier. However, Perceptron's excitement was hinged by **Minsky**[3] in 1969 . He proposed the famous **XOR** problem and the inability of Perceptrons in such linearly inseparable data distributions. It was the Minsky's tackle to NN community. Thereafter, NN researches would be dormant up until 1980s

![](http://www.cs.ru.nl/~ths/rt2/col/h10/draw-LTUdecis.GIF)

XOR problem which is nor linearly seperable data orientation

There had been not to much effort until the intuition of **Multi-Layer Perceptron (MLP)** was suggested by **Werbos[6]** in 1981 with NN specific **Backpropagation(BP)** algorithm, albeit BP idea had been proposed before by **Linnainmaa [5]** in 1970 in the name "reverse mode of automatic differentiation". Still BP is the key ingredient of today's NN architectures. With those new ideas, NN researches accelerated again. In 1985 - 1986 NN researchers successively presented the idea of **MLP**with practical **BP** training (Rumelhart, Hinton, Williams [7] -  Hetch, Nielsen[8])

[![From Hetch and Nielsen [7]](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/Hetch_Nielsen_NN.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/Hetch_Nielsen_NN.png)

From Hetch and Nielsen [8]

At the another spectrum, a very-well known ML algorithm was proposed by **J. R. Quinlan [9]** in 1986 that we call **Decision Trees**, more specifically **ID3** algorithm. This was the spark point of the another mainstream ML.  Moreover, ID3 was also released as a software able to find more real-life use case with its simplistic rules and its clear inference, contrary to still black-box NN models.

After ID3, many different alternatives or improvements have been explored by the community (e.g. ID4, Regression Trees, CART ...) and still it is one of the active topic in ML.

[![From Quinlan []](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/Quinlan_ID3.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/Quinlan_ID3.png)

From Quinlan [9]

One of the most important ML breakthrough was **Support Vector Machines** (Networks) (SVM), proposed by **Vapnik and Cortes[10]** in **1995** with very strong theoretical standing and empirical results. That was the time separating the ML community into two crowds as NN or SVM advocates. However the competition between two community was not very easy for the NN side  after **Kernelized** version of SVM by **near 2000s** .(I was not able to find the first paper about the topic), SVM got the best of many tasks that were occupied by NN models before. In addition, SVM was able to exploit all the profound knowledge of convex optimization, generalization margin theory and kernels against NN models. Therefore, it could find large push from different disciplines causing very rapid theoretical and practical improvements.

[![From Vapnik and Cortes [10]](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/SVM_Vapnik.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/05/SVM_Vapnik.png)

From Vapnik and Cortes [10]

NN took another damage by the work of Hochreiter's thesis [40] in 1991 and **Hochreiter et. al.[11] in 2001**, showing the gradient loss after the saturation of NN units as we apply BP learning. Simply means, it is redundant to train NN units after a certain number of epochs owing to saturated units hence NNs are very inclined to over-fit in a short number of epochs.

Little before, another solid ML model was proposed by **Freund and Schapire**in **1997** prescribed with boosted ensemble of weak classifiers called **Adaboost.** This work also gave the Godel Prize to the authors at the time. Adaboost trains weak set of classifiers that are easy to train, by giving more importance to hard instances. This model still the basis of many different tasks like face recognition and detection. It is also a realization of **PAC  (Probably Approximately Correct)** learning theory. In general, so called weak classifiers are chosen as simple decision stumps (single decision tree nodes). They introduced Adaboost as ;

> The model we study can be interpreted as a broad, abstract extension of the well-studied on-line prediction model to a general decision-theoretic setting...[11]

Another ensemble model explored by **Breiman** [12] in **2001** that ensembles multiple decision trees where each of them is curated by a random subset of instances and each node is selected from a random subset of features. Owing to its nature,  it is called **Random Forests(RF)**. RF has also theoretical and empirical proofs of endurance against over-fitting. Even AdaBoost shows weakness to over-fitting and outlier instances in the data, RFis more robust model against these caveats.(For more detail about RF, refer to [my old post.](http://www.erogol.com/randomness-randomforests/)). RF shows its success in many different tasks like Kaggle competitions as well.

> Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large[12]

As we come closer today, a new era of NN called **Deep Learning** has been commerced. This phrase simply refers NN models with many wide successive layers. The 3rd rise of NN has begun roughly in  **2005** with the conjunction of many different discoveries from past and present by  recent mavens Hinton, LeCun, Bengio, Andrew Ng and other valuable older researchers. I enlisted some of the important headings (I guess, I will dedicate complete post for Deep Learning specifically) ;

* GPU programming
* Convolutional NNs [18][20][40]
  + Deconvolutional Networks [21]
* Optimization algorithms
  + Stochastic Gradient Descent [19][22]
  + BFGS and L-BFGS [23]
  + Conjugate Gradient Descent [24]
  + Backpropagation [40][19]
* Rectifier Units
* Sparsity [15][16]
* Dropout Nets [26]
  + Maxout Nets  [25]
* Unsupervised NN models [14]
  + Deep Belief Networks [13]
  + Stacked Auto-Encoders [16][39]
  + Denoising NN models [17]

With the combination of all those ideas and non-listed ones, NN models are able to beat off state of art at very different tasks such as Object Recognition, Speech Recognition, NLP etc. However, it should be noted that this absolutely does not mean, it is the end of other ML streams. Even Deep Learning success stories grow rapidly , there are many critics directed to training cost and tuning exogenous parameters of  these models. Moreover, still SVM is being used more commonly owing to its simplicity. (said but may cause a huge debate 🙂 )

Before finish, I need to touch on one another relatively young ML trend. After the growth of WWW and Social Media, a new term, **BigData** emerged and affected ML research wildly. Because of the large problems arising from BigData , many strong ML algorithms are useless for reasonable systems (not for giant Tech Companies of course). Hence, research people come up with a new set of simple models that are dubbed **Bandit Algorithms [27 - 38]** (formally predicated with **Online** **Learning**)that makes learning easier and adaptable for large scale problems.

I would like to conclude this infant sheet of ML history. If you found something wrong (you should 🙂 ), insufficient or non-referenced, please don't hesitate to warn me in all manner.

# **References ----**

[1] Hebb D. O., The organization of behaviour.New York: Wiley & Sons.

[2]Rosenblatt, Frank. "The perceptron: a probabilistic model for information storage and organization in the brain." *Psychological review* 65.6 (1958): 386.

[3]Minsky, Marvin, and Papert Seymour. "Perceptrons." (1969).

[4]Widrow, Hoff "Adaptive switching circuits." (1960): 96-104.

[5]S. Linnainmaa. The representation of the cumulative rounding error of an algorithm as a Taylor  
expansion of the local rounding errors. Master’s thesis, Univ. Helsinki, 1970.

[6] P. J. Werbos. Applications of advances in nonlinear sensitivity analysis. In Proceedings of the 10th  
IFIP Conference, 31.8 - 4.9, NYC, pages 762–770, 1981.

[7] Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. *Learning internal representations by error propagation*. No. ICS-8506. CALIFORNIA UNIV SAN DIEGO LA JOLLA INST FOR COGNITIVE SCIENCE, 1985.

[8] Hecht-Nielsen, Robert. "Theory of the backpropagation neural network." *Neural Networks, 1989. IJCNN., International Joint Conference on*. IEEE, 1989.

[9] Quinlan, J. Ross. "Induction of decision trees." *Machine learning* 1.1 (1986): 81-106.

[10] Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks." *Machine learning* 20.3 (1995): 273-297.

[11] Freund, Yoav, Robert Schapire, and N. Abe. "A short introduction to boosting."*Journal-Japanese Society For Artificial Intelligence* 14.771-780 (1999): 1612.

[12] Breiman, Leo. "Random forests." *Machine learning* 45.1 (2001): 5-32.

[13] Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." *Neural computation* 18.7 (2006): 1527-1554.

[14] Bengio, Lamblin, Popovici, Larochelle, "Greedy Layer-Wise  
Training of Deep Networks", NIPS’2006

[15] Ranzato, Poultney, Chopra, LeCun " Efficient Learning of  Sparse Representations with an Energy-Based Model ", NIPS’2006

[16] Olshausen B a, Field DJ. Sparse coding with an overcomplete basis set: a strategy employed by V1? *Vision Res*. 1997;37(23):3311–25. Available at: http://www.ncbi.nlm.nih.gov/pubmed/9425546.

[17] Vincent, H. Larochelle Y. Bengio and P.A. Manzagol, [Extracting and Composing Robust Features with Denoising Autoencoders](http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/217), Proceedings of the Twenty-fifth International Conference on Machine Learning (ICML‘08), pages 1096 - 1103, ACM, 2008.

[18] Fukushima, K. (1980). Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biological Cybernetics, 36, 193–202.

[19] LeCun, Yann, et al. "Gradient-based learning applied to document recognition."*Proceedings of the IEEE* 86.11 (1998): 2278-2324.

[20] LeCun, Yann, and Yoshua Bengio. "Convolutional networks for images, speech, and time series." *The handbook of brain theory and neural networks*3361 (1995).

[21] Zeiler, Matthew D., et al. "Deconvolutional networks." *Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on*. IEEE, 2010.

[22] S. Vishwanathan, N. Schraudolph, M. Schmidt, and K. Mur- phy. Accelerated training of conditional random fields with stochastic meta-descent. In International Conference on Ma- chine Learning (ICML ’06), 2006.

[23] Nocedal, J. (1980). ”Updating Quasi-Newton Matrices with Limited Storage.” Mathematics of Computation 35 (151): 773782. doi:10.1090/S0025-5718-1980-0572855-

[24] S. Yun and K.-C. Toh, “A coordinate gradient descent method for l1- regularized convex minimization,” Computational Optimizations and Applications, vol. 48, no. 2, pp. 273–307, 2011.

[25] Goodfellow I, Warde-Farley D. Maxout networks. *arXiv Prepr arXiv …*. 2013. Available at: http://arxiv.org/abs/1302.4389. Accessed March 20, 2014.

[26] Wan L, Zeiler M. Regularization of neural networks using dropconnect. *Proc …*. 2013;(1). Available at: http://machinelearning.wustl.edu/mlpapers/papers/icml2013\_wan13. Accessed March 13, 2014.

[27] [Alekh Agarwal](http://www.cs.berkeley.edu/~alekh/), [Olivier Chapelle](http://olivier.chapelle.cc/), [Miroslav Dudik](http://www.cs.cmu.edu/~mdudik/), [John Langford](http://hunch.net/~jl), [A Reliable Effective Terascale Linear Learning System](http://arxiv.org/abs/1110.4198), 2011

[28] [M. Hoffman](http://www.cs.princeton.edu/~mdhoffma/), [D. Blei](http://www.cs.princeton.edu/~blei/), [F. Bach](http://www.di.ens.fr/~fbach/), [Online Learning for Latent Dirichlet Allocation](http://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf), in Neural Information Processing Systems (NIPS) 2010.

[29] [Alina Beygelzimer](http://hunch.net/~beygel), [Daniel Hsu](http://cseweb.ucsd.edu/~djhsu/), [John Langford](http://hunch.net/~jl), and [Tong Zhang](http://stat.rutgers.edu/home/tzhang/) [Agnostic Active Learning Without Constraints](http://arxiv.org/abs/1006.2588) NIPS 2010.

[30] [John Duchi](http://www.cs.berkeley.edu/~jduchi/), [Elad Hazan](http://ie.technion.ac.il/~ehazan/), and [Yoram Singer](http://www.magicbroom.info/About.html), [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.html), JMLR 2011 & COLT 2010.

[31] [H. Brendan McMahan](http://www.cs.cmu.edu/~mcmahan/), [Matthew Streeter](http://www.cs.cmu.edu/~matts/), [Adaptive Bound Optimization for Online Convex Optimization](http://arxiv.org/abs/1002.4908), COLT 2010.

[32] [Nikos Karampatziakis](http://www.cs.cornell.edu/~nk/) and [John Langford](http://hunch.net/~jl), [Importance Weight Aware Gradient Updates](http://arxiv.org/abs/1011.1576) UAI 2010.

[33] [Kilian Weinberger](http://www.cse.wustl.edu/~kilian/), [Anirban Dasgupta](http://research.yahoo.com/Anirban_Dasgupta/), [John Langford](http://hunch.net/~jl), [Alex Smola](http://alex.smola.org/), [Josh Attenberg](http://www.linkedin.com/in/joshattenberg), [Feature Hashing for Large Scale Multitask Learning](http://arxiv.org/pdf/0902.2206), ICML 2009.

[34] [Qinfeng Shi](http://users.cecs.anu.edu.au/~qshi/), [James Petterson](http://users.cecs.anu.edu.au/~jpetterson/), [Gideon Dror](http://www2.mta.ac.il/~gideon/), [John Langford](http://hunch.net/~jl), [Alex Smola](http://alex.smola.org/), and [SVN Vishwanathan](http://www.stat.purdue.edu/~vishy/), [Hash Kernels for Structured Data](http://hunch.net/~jl/projects/hash_reps/hash_kernels/hashkernel.pdf), AISTAT 2009.

[35] [John Langford](http://hunch.net/~jl), [Lihong Li](http://www.research.rutgers.edu/~lihong/), and [Tong Zhang](http://stat.rutgers.edu/home/tzhang/), [Sparse Online Learning via Truncated Gradient](http://hunch.net/~jl/projects/interactive/sparse_online/paper_sparseonline.pdf), NIPS 2008.

[36] [Leon Bottou](http://leon.bottou.org/), [Stochastic Gradient Descent](http://leon.bottou.org/projects/sgd), 2007.

[37] [Avrim Blum](http://www.cs.cmu.edu/~avrim/), [Adam Kalai](http://www.cs.cmu.edu/~akalai/), and [John Langford](http://hunch.net/~jl) [Beating the Holdout: Bounds for KFold and Progressive Cross-Validation](http://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf). COLT99 pages 203-208.

[38] [Nocedal, J.](http://www.ece.northwestern.edu/faculty/Nocedal_Jorge.html) (1980). "Updating Quasi-Newton Matrices with Limited Storage". Mathematics of Computation 35: 773–782.

[39] D. H. Ballard. Modular learning in neural networks. In AAAI, pages 279–284, 1987.

[40] S. Hochreiter. Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis, Institut f ̈ur In-  
formatik, Lehrstuhl Prof. Brauer, Technische Universit ̈at M ̈unchen, 1991. Advisor: J. Schmidhuber.
