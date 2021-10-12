# Effects of Behaviour Adaptation on the Spread of Infectious Disease on Networks with Community Structures

The study investigates the influence of awareness in the spread of infectious disease on networks with community structures. Awareness is defined as the reaction of individuals to the presence of infections in their environment. As the number of infections rises, they adjust their behaviour such that the probability of becoming infected decreases. We distinguish between local, community and global awareness, that is, awareness of the number of infected among one's direct neighbours (local awareness), one's community (community awareness) and the entire network (global awareness). The impact of awareness is studied on an SIS epidemic model using stochastic simulations and the mean-field approach. The results are reported for two major characteristics of an epidemic: the epidemic prevalence and the epidemic threshold. As expected, each of these three types of awareness reduces the epidemic prevalence. Interestingly, the epidemic threshold is lowered only by the local awareness and possibly by the community awareness when the communities are small.

## HierarchicalConfigurationModel.py

This file contains the class HierarchicalConfigurationModel that is used to generate a Hierarchical Configuration Model with given in-degree and out-degree sequences using the erased Configuration Model algorithm.

## unitls.py

This file contains various functions that are useful in the various operations of this project.

## numerical.py

This file contains the class numerical.SIS which models an SIS epidemic model with awareness on a given Hierarchical Configuration Model. This model can then be solved using the mean-field approach for the steady-state and approximate the epidemic threshold.

## simulations.py 

This file contains the class simulations.SIS which models an SIS epidemic model with awareness on a given Hierarchical Configuration Model. Then, we can run Gillespie-style simulations and approximate the epidemic threshold.

## awareness_models_simulations.py

This file contains the awareness class which can be used to define alternative (non-linear or discontinuous) awareness models for the simulations.

## networks folder

This folder contains the three network case studies that were examined in this project for the relationship between awareness and epidemic prevalence or epidemic threshold.
