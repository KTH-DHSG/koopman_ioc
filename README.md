# Inverse-Optimal-Control

Dependencies:
Casadi for Matlab & Python (download from here: https://web.casadi.org/get/).

## To Do

- [x] Re-check all the old code and materials and clean it.
- [x] Finish coding a full Koopman IOC example.
- [X] Analyze Koopman IOC solution, derive assumptions and Theorem and update overleaf to have a full explanation.
- [x] Develop solution to fix unicycle dynamics.
- [?] Develop solution to fix obstacle avoidance.
- [?] Code and test a Human Motion Prediction experiment.
- [X] Write a full paper.

## Summary

### inverse_lqr
This folder contains the Python code for the code of **iLQR**, i.e, the algorithm to learn the cost function Q of a LQR problem (linear system dynamics) for two different tests (an holonomic mobile robot and a pendulum). See pages 1-2 of Notes IOC.pdf in the documentation. It also contains a discontinued example using unicycle dynamics which did not work since the dynamics are nonlinear due to time-varying B.

* The subfolder *inverse_lqr_withR* contains the algorithm to also learn the cost function R together with Q for the same two tests. See page 3 of Notes IOC.pdf in the documentation. 

### koopman_ioc
This folder contains the Python code for different versions of **Koopman** algorithms to learn the system dynamics and then apply IOC on those to learn the cost functions. The subfolders are:

* ***bilinear_pmp_example***: contains the code for the bilinear Koopman in z(x) and z(x)*u with the adjusted bilinear PMP instead of iLQR. See pages 20-29 of Notes IOC.pdf in the documentation. **This is the current algorithm in use and the most updated code**. 

    * The subfolder *hmp_example* includes the use of unicycle dynamics and obstacle avoidance for human motion prediction, see pages 33-38 of Notes IOC.pdf in the documentation. 

    * The subfolder *aug_hmp_example* includes the use augmented Koopman of unicycle dynamics and obstacle avoidance for human motion prediction. This code is **DEPRECATED**.

* *forward_comparison*: contains the code to test the koopman with input-constrained LQR, but only the forward comparison (thus no IOC). *forward_comparison.py* is the basic case, see pages 8-10 of Notes IOC.pdf in the documentation. *forward_comparison_ref.py* is the case using error dynamics LQR, see page 11 of Notes IOC.pdf in the documentation. *forward_comparison_mix.py* is the case mixing constraints, see pages 12-19 of Notes IOC.pdf in the documentation.

* *koopman_lqr_example*: contains the code for the original corrected example of Koopman with state z(x) and control v(x,u) and assuming naive linear lifted dynamics. See pages 4-7 of Notes IOC.pdf in the documentation.

* *seg_bilinear_pmp_example*: contains the code for a bilinear Koopman in z(x) and z(x)*u, but instead of using iLQR we use the nIOC code. See pages 30-32 of Notes IOC.pdf in the documentation.

* *UNCONTROLLABLE_example_32*: contains the code for the original example with one uncontrollable mode, thus the controllability assumption is not fulfilled and IOC is not possible.

### nonlinear_ioc
This folder contains both the MATLAB and Python version of the code for **nIOC**, i.e., the algorithm to learn the unknown weights for a series of known features of a cost function with nonlinear system dynamics. See reference [3] from latex.