---
layout: post
title: "Some possible Matrix Algebra libraries based on C/C++"
description: "I've gathered the following from online research so far:

I've used Armadillo(http://arma"
tags: c linear algebra machine learning
minute: 4
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

I've gathered the following from online research so far:

I've used [Armadillo](http://arma.sourceforge.net/) a little bit, and found the interface to be intuitive enough, and it was easy to locate binary packages for Ubuntu (and I'm assuming other Linux distros). I haven't compiled it from source, but my hope is that it wouldn't be too difficult. It meets most of my design criteria, and uses dense linear algebra. It can call LAPACK or MKL routines.

I've heard good things about [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), but haven't used it. It [claims to be fast](http://eigen.tuxfamily.org/index.php?title=Benchmark), uses templating, and supports dense linear algebra. It doesn't have LAPACK or BLAS as a dependency, but appears to be able to do everything that LAPACK can do (plus some things LAPACK can't). A lot of projects use Eigen, which is promising. It has a binary package for Ubuntu, but as a header-only library it's trivial to use elsewhere too.

The [Matrix Template Library](http://www.simunova.com/node/33) version 4 also looks promising, and uses templating. It supports both dense and sparse linear algebra, and can call [UMFPACK](http://www.cise.ufl.edu/research/sparse/umfpack/) as a sparse solver. The features are somewhat unclear from their website. It has a binary package for Ubuntu, downloadable from their web site.

[PETSc](http://www.mcs.anl.gov/petsc/), written by a team at Argonne National Laboratory, has access to sparse and dense linear solvers, so I'm presuming that it can function as a matrix library. It's written in C, but has C++ bindings, I think (and even if it didn't, calling C from C++ is no problem). The documentation is incredibly thorough. The package is a bit overkill for what I want to do now (matrix multiplication and indexing to set up mixed-integer linear programs), but could be useful as a matrix format for me in the future, or for other people who have different needs than I do.

[Trilinos](http://trilinos.sandia.gov/index.html), written by a team at Sandia National Laboratory, provides object-oriented C++ interfaces for dense and sparse matrices through its Epetra component, and templated interfaces for dense and sparse matrices through its Tpetra component. It also has components that provide linear solver and eigensolver functionality. The documentation does not seem to be as polished or prominent as PETSc; Trilinos seems like the Sandia analog of PETSc. PETSc can call some of the Trilinos solvers. Binaries for Trilinos are available for Linux.

[Blitz](http://sf.net/projects/blitz/) is a C++ object-oriented library that has Linux binaries. It doesn't seem to be actively maintained (2012-06-29: a new version has just appeared yesterday!), although the mailing list is active, so there is some community that uses it. It doesn't appear to do much in the way of numerical linear algebra beyond BLAS, and looks like a dense matrix library. It uses templates.

[Boost::uBLAS](http://www.boost.org/doc/libs/1_48_0/libs/numeric/ublas/doc/index.htm) is a C++ object-oriented library and part of the Boost project. It supports templating and dense numerical linear algebra. I've heard it's not particularly fast.

The [Template Numerical Toolkit](http://math.nist.gov/tnt/history.html) is a C++ object-oriented library developed by NIST. Its author, Roldan Pozo, seems to contribute patches occasionally, but it doesn't seem to be under active development any longer (last update was 2010). It focuses on dense linear algebra, and provides interfaces for some basic matrix decompositions and an eigenvalue solver.

[Elemental](http://code.google.com/p/elemental/), developed by Jack Poulson, is a distributed memory (parallel) dense linear algebra software package written in a style similar to [FLAME](http://z.cs.utexas.edu/wiki/flame.wiki/FrontPage). For a list of features and background on the project, see his [documentation](http://elemental.googlecode.com/hg/doc/build/html/index.html). FLAME itself has an associated library for sequential and shared-memory dense linear algebra, called [libflame](http://z.cs.utexas.edu/wiki/flame.wiki/libflame/), which appears to be written in object-oriented C. Libflame looks a lot like LAPACK, but with better notation underlying the algorithms to make development of fast numerical linear algebra libraries more of a science and less of a black art.

There are other libraries that can be added to the list; if we're counting sparse linear algebra packages as "matrix libraries", the best free one I know of in C is [SuiteSparse](http://www.cise.ufl.edu/research/sparse/SuiteSparse/), which is programmed in object-oriented style. I've used SuiteSparse and found it fairly easy to pick up; it depends on BLAS and LAPACK for some of the algorithms that decompose sparse problems into lots of small, dense linear algebra subproblems. The lead author of the package, Tim Davis, is incredibly helpful and a great all-around guy.

The [Harwell Subroutine Libraries](http://www.hsl.rl.ac.uk/) are famous for their sparse linear algebra routines, and are free for academic users, though you have to go through this process of filling out a form and receiving an e-mail for each file that you want to download. Since the subroutines often have dependencies, using one solver might require downloading five or six files, and the process can get somewhat tedious, especially since the form approval is not instantaneous.

There are also other sparse linear algebra solvers, but as far as I can tell, [MUMPS](http://graal.ens-lyon.fr/MUMPS/index.php?page=home) and other packages are focused mostly on the solution of linear systems, and solving linear systems is the least of my concerns right now. (Maybe later, I will need that functionality, and it could be useful for others.)



### Related posts:

1. [What is "long long" type in c++?](http://www.erogol.com/what-is-long-long-type-in-c/ "What is \"long long\" type in c++?")
2. [Our ECCV2014 work "ConceptMap: Mining noisy web data for concept learning"](http://www.erogol.com/eccv2014-work-conceptmap-mining-noisy-web-data-concept-learning/ "Our ECCV2014 work \"ConceptMap: Mining noisy web data for concept learning\"")
3. [Extracting a sub-vector at C++](http://www.erogol.com/extracting-sub-vector-c/ "Extracting a sub-vector at C++")
4. [ML Work-Flow (Part 5) – Feature Preprocessing](http://www.erogol.com/ml-work-flow-part-5-feature-processing/ "ML Work-Flow (Part 5) – Feature Preprocessing")