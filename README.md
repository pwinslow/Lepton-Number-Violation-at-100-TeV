# Lepton Number Violation in the 100 TeV Frontier
This repository serves to store and share codes associated with the LNV@100TeV project. The interplay between high energy searches at future O(100 TeV) hadron colliders and precision low energy tests of lepton number violation is studied with the goal of determining which future experimental program will best probe generic new physics models incorporating LNV. The codes stored here are relevant only to high energy searches for lepton number violation, i.e., simulation and analysis of high energy collider data. Calculations relevant to low energy nuclear processes will only appear later in publication (links will be provided once available).
 
# Background
Total lepton number remains an accidental global symmetry of the Standard Model of particle physics at low temperatures, i.e., less than O(100 GeV). Its' true status is one of the major questions driving current research at the interface of nuclear and particle physics as well as cosmology. Experimental searches for violations of total lepton number symmetry search for elementary particle reactions that cannot occur unless it is not conserved.     

At high energy collider experiments, such as the Large Hadron Collider, protons are accelerated to speeds close to the speed of light and then forced to collide. Since the initial state of the collision contains no leptons (just two protons), we can look for evidence of total lepton number violation by searching for statistically significant signals of non-zero total lepton number in the collision debris. To be specific, we are interested in an event signature characterized by two leptons (in this case electrons or positrons) with the same charge in the final state. We simulate such events using the open source software packages MadGraph (simulating hard scattering in proton-proton collisions), PYTHIA (simulating hadron showers from the underlying quarks and gluons in the collisions), and DELPHES (simulating the response of a possible future collider with a center-of-mass energy of 100 TeV). We then employ machine learning techniques to maximally separate signal from background, allowing us to focus on a highly non-linear multivariate region of the data within which the discovery potential is maximized.    

# Codes   

In the following, I list the codes used to accomplish this in the order in which they were used.   

## Data Production   

10^6 events at 100 TeV were simulated for all background sources as well as a single benchmark parameter point for the signal. All sources considered are

* Irreducible backgrounds: W<sup>&plusmn;</sup>W<sup>&plusmn;</sup>jj (leptonic weak decays), W<sup>&plusmn;</sup>Zjj (leptonic weak decays), and ZZjj (leptonic weak decays)
* Reducible backgrounds: **Charge-fake** --> tt~ (leptonic decays with charge-flip probabilities applied), &#947;Zjj (leptonic decays with charge-flip probabilities applied), **Jet-fake** --> W<sup>&plusmn;</sup>W<sup>&plusmn;</sup>jj (hadronic weak decays with jet-fake probabilities applied), tt~ (hadronic decays with jet-fake probabilities applied), single top (leptonic decay with jet-fake probability applied), pure QCD (4j with jet-fake probabilities applied)
* Signal: e<sup>&plusmn;</sup>e<sup>&plusmn;</sup>jj. Benchmark point is chosen as the current limit imposed by the GERDA experiment, i.e., both particle masses (couplings) fixed to 1 TeV (0.202).

*Note: both charge states were included for each source!* All data from the above simulations can be found on the titan cluster in the path **/data2/pwinslow/100TeV_Fixed_Results**.

A scan of the model parameter space was also performed. Both masses and couplings were assumed to be equal to each other and the scan was composed a grid of 200 points covering a mass (couping) range of [250 GeV, 5 TeV] ([0.2, 2]). All data from the scan simulations can be found on the titan cluster in the path **/data2/pwinslow/100TeV_Scan_Results**.

From the scripts described below, **Sub_scripts.sh**, **Pythia_Delphes_script.sh**, and **full_LHCO_wrangler.py** were used specifically for generation of 10^6 events for each of the various backgrounds and the signal benchmark point while **Scan_Cleanup.sh** and **mlrun.sh** were used to perform all simulations for the scan of the parameter space.

#### Fixed_Sub_script.sh   

The purpose of this script is to submit parallel runs of Madgraph5 (for monte carlo calculation of parton level cross sections) to the titan cluster PBS (Portable Batch System) resource manager. 

#### Pythia_Delphes_script.sh   

The purpose of this script is to 1) run PYTHIA on existing parton level events to simulate parton showering with matching effects accounted for and 2) to run the delphes detector simulator using our own specialized detector card simulating a O(100 TeV) detector environment.   

#### full_LHCO_wrangler.py   

The purpose of this script is to collect data generated from parallel computation of parton level cross sections into one dat file. This file should contain the average matched cross section (after PYTHIA) and all events over all runs. The script also creates a repository to store this dat file and all root files for sharing with collaborators.   

#### PW_FCC_delphes_card.tcl   

This card provides the currently estimated specs for a proposed detector at a O(100 TeV) future circular hadron collider. We have highly borrowed from the card created by Heather Gray and Filip Moortgat for FCC detector studies.   

#### Scan_Cleanup.sh

The purpose of this script is to reduce and collect the results of the parameter scan performed by the Scan_Sub_script.sh script. After running, each parameter point should be associated with a single event file.   

#### mlrun.sh   

The purpose of this script is to run the machine learning pipeline on each parameter point of the scan. The result is a dataset which assigns a statistical significance to each parameter point.  

## Data Analysis   

#### Wrangler.py   

This script transforms raw LHCO data to a set of features for machine learning and also provides a method for plotting probability densities.   

#### MLpipeline.py   

This script performs a machine learning pipeline on a given data file. The pipeline trains a random forest classifier which is then used to optimize signal and background separation, maximizing statistical significance of any signal.   

#### Analysis.ipynb   

This is a python notebook (similar to a mathematica notebook). It contains quite a bit of explanatory text for those who are interested in the details of what is involved in the machine learning pipeline. It also contains the final results in the form of two plots towards the end of the notebook. A detailed explanation of these results is contained within. 

## Final Results   

The two plots representing the final results are included indepedently in png format in the main repo with the file names **luminosity_plot.png** and **param_scan.png**.
