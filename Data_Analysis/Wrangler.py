########################################################################################################
# This script transforms raw LHCO data to a set of features for machine learning and also provides a   #
# method for plotting probability densities.							       #
########################################################################################################


# Analysis imports
from __future__ import division
from math import cosh, sinh, cos, sin, factorial
import numpy as np
from numpy import linalg as LA
import pandas as pd
from pandas import Series, DataFrame
from random import random
import itertools
#import re
import os


# Define combinatoric factor for jet fake probabilities
def nCk(n, k):
	return factorial(n) / (factorial(n-k) * factorial(k))
	
# Define charge flip probabilities based on pseudo-rapidities
def CF_probs(eta):
	
	if eta < -2.5:
		prob = 0
		
	elif (-2.5 <= eta) and (eta <= -2):
		prob = 8.9e-2
		
	elif (-2 < eta) and (eta <= -1.52):
		prob = 4.4e-2
		
	elif (-1.52 < eta) and (eta <= -1.37):
		prob = 0
		
	elif (-1.37 < eta) and (eta <= -.8):
		prob = 1.8e-2
		
	elif (-.8 < eta) and (eta <= 0):
		prob = .7e-2
		
	elif (0 < eta) and (eta <= .8):
		prob = .2e-2
		
	elif (.8 < eta) and (eta <= 1.37):
		prob = 1.9e-2
		
	elif (1.37 < eta) and (eta <= 1.52):
		prob = 0
		
	elif (1.52 < eta) and (eta <= 2):
		prob = 3.9e-2
		
	elif (2 < eta) and (eta <= 2.5):
		prob = 8.45e-2
		
	elif (2.5 < eta):
		prob = 0
	
	return prob



class Wrangler(object):

	'''

	This class transforms the raw LHCO data to a set of features for machine learning and also provides a method for plotting 
	probability densities.

	'''

	__version__ = 'Beta_1.0'

	# Initialize the class
	def __init__(self, Ne_1 = None, Np_1 = None, Ne_2 = None, Np_2 = None, Nj = None, Nb = None):

		self.Ne_1, self.Np_1, self.Ne_2, self.Np_2, self.Nj, self.Nb = Ne_1 or 0, Np_1 or 0, Ne_2 or 0, Np_2 or 0, Nj or 0, Nb or 0


		'''

		Attributes:
		-----------

			Ne_1 and Ne_2: Upper bounds on minimum number of electrons
			Np_1 and Np_2: Upper bounds on minimum number of positrons
			Nj: Upper bound on minimum number of jets
			Nb: Required number of b-jets
			
			Two attributes are required to specify the upper bounds on electrons and positrons since the same-sign lepton signature includes both 
			e- e- and e+ e+ events. Specifically, we filter events based on an or statement of the form

					if (Num_e >= Ne_1, Num_p >= Np_1, Num_j >= Nj, Num_b >= Nb) or (Num_e >= Ne_2, Num_p >= Np_2, Num_j >= Nj, Num_b >= Nb):
						keep event
					else:
						toss event

			where Ne_1, Np_1 = 2, 0 and Ne_2, Np_2 = 0, 2 covers both e- e- and e+ e+ cases.

		Methods:
		--------

			Three_Vec: Translate raw kinematic information into 3-momenta.
			
			Four_Vec: Translate raw kinematic information into 4-momenta.
			
			Inv_Mass: Caculate invariant mass of a 4-vector.
			
			cos_theta_star: Calculate the polar angle of two objects in the Collins-Soper frame (Phys. Rev. D 16 (1977) 2219-2225).
			
			Sphericity_Tensor: Create a sphericity tensor based on a given 3-momenta.
			
			Event_Shape_Variables: Calculate event shape variables, i.e., sphericity, transverse sphericity, aplanarity, and planarity.
			
			Delta_R: Returns angular distance between two final state particles.

			Feature_Architect: Takes a LHCO event file and performs two jobs, (1) imposes basic selection cuts to ensure signature purity and (2) outputs data 
			in the form of a list of raw features which conforms to the basic format of a design matrix which can be directly used as input into classification 
			algorithms. Each data instance corresponds to a given event while the list of raw features within a given instance is as follows:
															
				e_Num = Number of electrons
				p_Num = Number of positrons
				H_T_leptons = Total transverse momentum of all leptons
				H_T_jets = Total transverse momentum of all jets
				Delta_R_leptons = Angular distance between the two leading leptons
				Delta_R_jets = Angular distance between the two leading jets
				Delta_R_leptonjet = Angular distance between the leading lepton and jet
				dilepton_mass = Invariant mass of the leading leptons
				dijet_mass = Invariant mass of the leading jets
				dileptonjet_mass = Invariant mass of the leading lepton and jet
				cos_leptons = Polar angle between the two leading leptons in the Collins-Soper frame
				cos_jets = Polar angle between the two leading jets in the Collins-Soper frame
				cos_leptonjet = Polar angle between the leading lepton and jet in the Collins-Soper frame
				MET = Missing transverse momentum
				S_leptons = The sphericity of all leptons
				TS_leptons = The transverse sphericity of all leptons
				AP_leptons = The aplanarity of all leptons
				P_leptons = The planarity of all leptons
				S_jets = The sphericity of all jets
				TS_jets = The transverse sphericity of all jets
				AP_jets = The aplanarity of all jets
				P_jets = The planarity of all jets
				S_global = The sphericity of the full event
				TS_global = The transverse sphericity of the full event (otherwise known as the circularity)
				AP_global = The aplanarity of the full event
				P_global = The planarity of the full event
			
			Hist_Constructor: Constructs probability density plots based on a specified bin structure, cross section, and weight
		'''


	
	def Three_Vec(self, eta, phi, pT):
		
		p1, p2, p3 = pT * sin(phi), pT * cos(phi), pT * sinh(eta)
		
		return np.array([p1, p2, p3])
		
		
	
	def Four_Vec(self, eta, phi, pT):
		
		p0, p1, p2, p3 = pT * cosh(eta), pT * sin(phi), pT * cos(phi), pT * sinh(eta)
		
		return np.array([p0, p1, p2, p3])
		
		
		
	def Inv_Mass(self, p):

		p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
		
		inv_mass = np.sqrt( p0**2 - p1**2 - p2**2 - p3*2 )
		
		return inv_mass
		
		
		
	def cos_theta_star(self, eta_1, phi_1, pT_1, eta_2, phi_2, pT_2):
		
		pT_12 = LA.norm( np.array([pT_1 * sin(phi_1) + pT_2 * sin(phi_2), pT_1 * cos(phi_1) + pT_2 * cos(phi_2)]) )
		p_12 = self.Four_Vec( eta_1, phi_1, pT_1 ) + self.Four_Vec( eta_2, phi_2, pT_2 )
		m_12 = self.Inv_Mass( p_12 )
		
		cos_theta = ( np.abs( sinh( eta_1 - eta_2 ) ) / np.sqrt( 1 + ( pT_12 / m_12 )**2 ) ) * ( 2 * pT_1 * pT_2 / m_12**2 )
		
		return cos_theta
		
		
	
	def Sphericity_Tensor(self, p):
		
		px, py, pz = p[0], p[1], p[2]
		
		M_xyz = np.array([[px**2, px * py, px * pz], [py * px, py**2, py * pz], [pz * px, pz * py, pz**2]])
		
		return M_xyz
		
	
	
	def Event_Shape_Variables(self, S_tensor):
		
		# Calculate eigenvalues of sphericity matrices
		lmbda = LA.eigvals(S_tensor).real
		
		# Normalize and sort set of eigenvalues
		lmbda = np.sort(lmbda / LA.norm(lmbda))[::-1]
		
		# Define event shape variables
		Sphericity = 1.5 * ( lmbda[1] + lmbda[2] )
		Trans_Sphericity = 2 * lmbda[1] / ( lmbda[0] + lmbda[1] )
		Aplanarity = 1.5 * lmbda[2]
		Planarity = lmbda[1] - lmbda[2]
		
		return Sphericity, Trans_Sphericity, Aplanarity, Planarity
		
		
		
	def Delta_R(self, eta_1, phi_1, eta_2, phi_2):
		
		delta_eta = eta_1 - eta_2
		
		if phi_1 - phi_2 < - np.pi:

			delta_phi = (phi_1 - phi_2) + 2 * np.pi
			
		elif phi_1 - phi_2 > np.pi:
			
			delta_phi = (phi_1 - phi_2) - 2 * np.pi
			
		else:
			
			delta_phi = phi_1 - phi_2
			
		return np.sqrt( delta_eta**2 + delta_phi**2 )
		
		
	
	def Feature_Architect(self, lhco_file):
    
		with open(lhco_file, 'r+') as File:
			delphes_file = File.readlines()

		# Read off average matched cross section
		#sigma = float(re.findall("-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", delphes_file[0])[0])
		sigma = float( delphes_file[0].split(':')[1].strip() )
		
		# Initialize the design matrix for the given dataset
		X = []

		# Initialize all analysis features
		e_Num, L_e_pT, NL_e_pT, L_e_eta, L_e_phi, NL_e_eta, NL_e_phi = 0, 0, 0, 0, 0, 0, 0
		p_Num, L_p_pT, NL_p_pT, L_p_eta, L_p_phi, NL_p_eta, NL_p_phi = 0, 0, 0, 0, 0, 0, 0
		j_Num, L_j_pT, L_j_eta, L_j_phi, NL_j_pT, NL_j_eta, NL_j_phi = 0, 0, 0, 0, 0, 0, 0
		NNL_j_pT, NNL_j_eta, NNL_j_phi, NNNL_j_pT, NNNL_j_eta, NNNL_j_phi = 0, 0, 0, 0, 0, 0
		H_T_jets = 0
		b_Num = 0
		MET = 0
		
		# Initialize sphericity tensors for the global, hadronic, and leptonic geometries
		Sph_global = np.zeros((3,3), dtype = 'float')
		Sph_jets = np.zeros((3,3), dtype = 'float')
		Sph_leptons = np.zeros((3,3), dtype = 'float')

		CF_Flag = False # Initialize charge flip flag
		CF_prob_sum = 0 # Initialize a total charge flip probability
		
		JF1_Flag = False # Initialize single jet fake flag
		JF1_prob = (0.5/5000) * nCk(3, 1) # Define single jet fake probability
		
		JF2_Flag = False # Initialize double jet fake flag
		JF2_prob = (0.5/5000)**2 * nCk(4, 2) # Define double jet fake probability
		
		# Begin event collection
		line = 5 # Skip header info
		Tot_event_counter, event_counter = 0, 0 # Initialize counters for total and retained events
		while line < len(delphes_file):
   
			# Collect info from a given event
			if float(delphes_file[line].strip().split()[0]) != 0:

				# Collect state info
				state_info = [float(i) for i in delphes_file[line].strip().split()]
        
				# If electron, collect electron info
				if state_info[1] == 1 and state_info[6] == -1:
					e_Num += 1
					if state_info[4] > L_e_pT:
						L_e_pT = state_info[4]
						L_e_eta = state_info[2]
						L_e_phi = state_info[3]
					if NL_e_pT < state_info[4] < L_e_pT:
						NL_e_pT = state_info[4]
						NL_e_eta = state_info[2]
						NL_e_phi = state_info[3]
                
				# If positron, collect positron info
				if state_info[1] == 1 and state_info[6] == 1:
					p_Num += 1
					if state_info[4] > L_p_pT:
						L_p_pT = state_info[4]
						L_p_eta = state_info[2]
						L_p_phi = state_info[3]
					if NL_p_pT < state_info[4] < L_p_pT:
						NL_p_pT = state_info[4]
						NL_p_eta = state_info[2]
						NL_p_phi = state_info[3]
	                
				# If jet, collect jet info. Note: jets with pT too small to be mistaken for leptons still contribute to H_T and S_jets.
				if state_info[1] == 4 and state_info[7] == 0:
					j_Num += 1
					if state_info[4] > L_j_pT:
						L_j_pT = state_info[4]
						L_j_eta = state_info[2]
						L_j_phi = state_info[3]
					elif NL_j_pT < state_info[4] and state_info[4] < L_j_pT:
						NL_j_pT = state_info[4]
						NL_j_eta = state_info[2]
						NL_j_phi = state_info[3]
					elif NNL_j_pT < state_info[4] and state_info[4] < NL_j_pT:
						NNL_j_pT = state_info[4]
						NNL_j_eta = state_info[2]
						NNL_j_phi = state_info[3]
					elif NNNL_j_pT < state_info[4] and state_info[4] < NNL_j_pT:
						NNNL_j_pT = state_info[4]
						NNNL_j_eta = state_info[2]
						NNNL_j_phi = state_info[3]
					elif state_info[4] < NNNL_j_pT:
						H_T_jets += state_info[4]
						Sph_jets += self.Sphericity_Tensor(self.Three_Vec(state_info[2], state_info[3], state_info[4]))
						
	                
				# If b jet, collect b jet info
				if state_info[1] == 4 and state_info[7] > 0:
					b_Num += 1

	                
				# If MET, collect MET info
				if state_info[1] == 6:
					MET += state_info[4]
	            						

				# Increment line number									
				line += 1
				
			else:
	        
				# Impose basic selection cuts. Signature: 2 same-sign leptons + (>= 2 jets) + (== 0 b-jets).
				if (e_Num >= self.Ne_1 and p_Num >= self.Np_1 and j_Num >= self.Nj and b_Num == self.Nb) \
				or (e_Num >= self.Ne_2 and p_Num >= self.Np_2 and j_Num >= self.Nj and b_Num == self.Nb):
	        
					# Once all requirements are passed then also increment retained event count
					event_counter += 1
						
					# Virtually all training features depend on which type of background is being currently analyzed. Because of this, 
					# we'll analyse each case individually below.
					
					# Diboson selection cuts
					if self.Ne_1 == 2 and self.Np_1 == 0 and self.Ne_2 == 0 and self.Np_2 == 2 and self.Nj == 2 and self.Nb == 0:
						
						# Check if same-sign leptons are electrons or positrons. For ZZ events there's likely an equal number of e's 
						# and p's. In that case, take the same-sign pair to be the one with largest L_pT.
						if (e_Num > p_Num) or ((e_Num == p_Num) and (L_e_pT > L_p_pT)):
							
							H_T_leptons = L_e_pT + NL_e_pT
							
							Delta_R_leptons = self.Delta_R(L_e_eta, L_e_phi, NL_e_eta, NL_e_phi)
							Delta_R_leptonjet = self.Delta_R(L_e_eta, L_e_phi, L_j_eta, L_j_phi)
							
							dilepton_p = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(NL_e_eta, NL_e_phi, NL_e_pT)
							dilepton_mass = self.Inv_Mass(dilepton_p)
							dileptonjet_p = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dileptonjet_mass = self.Inv_Mass(dileptonjet_p)
							
							cos_leptons = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, NL_e_eta, NL_e_phi, NL_e_pT)
							cos_leptonjet = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, L_j_eta, L_j_phi, L_j_pT)
							
							Sph_leptons = self.Sphericity_Tensor(self.Three_Vec(L_e_eta, L_e_phi, L_e_pT)) + \
									   self.Sphericity_Tensor(self.Three_Vec(NL_e_eta, NL_e_phi, NL_e_pT))
							
						elif (e_Num < p_Num) or ((e_Num == p_Num) and (L_e_pT < L_p_pT)):
							
							H_T_leptons = L_p_pT + NL_p_pT
							
							Delta_R_leptons = self.Delta_R(L_p_eta, L_p_phi, NL_p_eta, NL_p_phi)
							Delta_R_leptonjet = self.Delta_R(L_p_eta, L_p_phi, L_j_eta, L_j_phi)
							
							dilepton_p = self.Four_Vec(L_p_eta, L_p_phi, L_p_pT) + self.Four_Vec(NL_p_eta, NL_p_phi, NL_p_pT)
							dilepton_mass = self.Inv_Mass(dilepton_p)
							dileptonjet_p = self.Four_Vec(L_p_eta, L_p_phi, L_p_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dileptonjet_mass = self.Inv_Mass(dileptonjet_p)
							
							cos_leptons = self.cos_theta_star(L_p_eta, L_p_phi, L_p_pT, NL_p_eta, NL_p_phi, NL_p_pT)
							cos_leptonjet = self.cos_theta_star(L_p_eta, L_p_phi, L_p_pT, L_j_eta, L_j_phi, L_j_pT)
							
							Sph_leptons = self.Sphericity_Tensor(self.Three_Vec(L_p_eta, L_p_phi, L_p_pT)) + \
									   self.Sphericity_Tensor(self.Three_Vec(NL_p_eta, NL_p_phi, NL_p_pT))
															
								
						H_T_jets += L_j_pT + NL_j_pT + NNL_j_pT + NNNL_j_pT
						
						Delta_R_jets = self.Delta_R(L_j_eta, L_j_phi, NL_j_eta, NL_j_phi)
						
						dijet_p = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
						dijet_mass = self.Inv_Mass(dijet_p)
						
						cos_jets = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NL_j_eta, NL_j_phi, NL_j_pT)
						
						Sph_jets += self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								 self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT)) + \
								 self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)) + \
								 self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_global = Sph_leptons + Sph_jets
						
						S_leptons, TS_leptons, AP_leptons, P_leptons = self.Event_Shape_Variables(Sph_leptons)
						S_jets, TS_jets, AP_jets, P_jets = self.Event_Shape_Variables(Sph_jets)
						S_global, TS_global, AP_global, P_global = self.Event_Shape_Variables(Sph_global)
						
								
							
					# Charge-flip selection cuts
					if self.Ne_1 == 1 and self.Np_1 == 1 and self.Ne_2 == 1 and self.Np_2 == 1 and self.Nj == 2 and self.Nb == 0:
						
						CF_Flag = True # Set charge flip flag
						
						H_T_leptons = L_e_pT + L_p_pT
						H_T_jets += L_j_pT + NL_j_pT + NNL_j_pT + NNNL_j_pT
							
						Delta_R_leptons = self.Delta_R(L_e_eta, L_e_phi, L_p_eta, L_p_phi)
						Delta_R_jets = self.Delta_R(L_j_eta, L_j_phi, NL_j_eta, NL_j_phi)
						
						dilepton_p = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(L_p_eta, L_p_phi, L_p_pT)
						dilepton_mass = self.Inv_Mass(dilepton_p)
						dijet_p = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
						dijet_mass = self.Inv_Mass(dijet_p)
						
						cos_leptons = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, L_p_eta, L_p_phi, L_p_pT)
						cos_jets = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NL_j_eta, NL_j_phi, NL_j_pT)
						
						Sph_leptons = self.Sphericity_Tensor(self.Three_Vec(L_e_eta, L_e_phi, L_e_pT)) + \
								   self.Sphericity_Tensor(self.Three_Vec(L_p_eta, L_p_phi, L_p_pT))
						Sph_jets += self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								 self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT)) + \
								 self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)) + \
								 self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_global = Sph_leptons + Sph_jets
						
						S_leptons, TS_leptons, AP_leptons, P_leptons = self.Event_Shape_Variables(Sph_leptons)
						S_jets, TS_jets, AP_jets, P_jets = self.Event_Shape_Variables(Sph_jets)
						S_global, TS_global, AP_global, P_global = self.Event_Shape_Variables(Sph_global)
						
						if L_e_pT > L_p_pT:
							
							# Adjust lepton numbers
							e_Num += 1
							p_Num -= 1
							
							# Increment charge flip probability
							CF_prob_sum += CF_probs(L_p_eta)
							
							Delta_R_leptonjet = self.Delta_R(L_e_eta, L_e_phi, L_j_eta, L_j_phi)
							
							dileptonjet_p = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dileptonjet_mass = self.Inv_Mass(dileptonjet_p)
							
							cos_leptonjet = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, L_j_eta, L_j_phi, L_j_pT)
							
						else:
							
							# Adjust lepton numbers
							e_Num -= 1
							p_Num += 1
							
							# Increment charge flip probability
							CF_prob_sum += CF_probs(L_e_eta)
							
							Delta_R_leptonjet = self.Delta_R(L_p_eta, L_p_phi, L_j_eta, L_j_phi)
							
							dileptonjet_p = self.Four_Vec(L_p_eta, L_p_phi, L_p_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dileptonjet_mass = self.Inv_Mass(dileptonjet_p)
							
							cos_leptonjet = self.cos_theta_star(L_p_eta, L_p_phi, L_p_pT, L_j_eta, L_j_phi, L_j_pT)
											
						
					# Single jetfake selection cuts
					if self.Ne_1 == 1 and self.Np_1 == 0 and self.Ne_2 == 0 and self.Np_2 == 1 and self.Nj == 3 and self.Nb == 0:						
						
						JF1_Flag = True # Set single jet fake flag
						
						# Adjust number of jets
						j_Num -= 1
						
						
						if (e_Num > p_Num) or ((e_Num == p_Num) and (L_e_pT > L_p_pT)):
							
							# Adjust lepton number
							e_Num += 1
							
							H_T_leptons_1 = L_e_pT + L_j_pT
							H_T_leptons_2 = L_e_pT + NL_j_pT
							H_T_leptons_3 = L_e_pT + NNL_j_pT
							H_T_leptons = np.mean([H_T_leptons_1, H_T_leptons_2, H_T_leptons_3])
							
							Delta_R_leptons_1 = self.Delta_R(L_e_eta, L_e_phi, L_j_eta, L_j_phi)
							Delta_R_leptons_2 = self.Delta_R(L_e_eta, L_e_phi, NL_j_eta, NL_j_phi)
							Delta_R_leptons_3 = self.Delta_R(L_e_eta, L_e_phi, NNL_j_eta, NNL_j_phi)
							Delta_R_leptons = np.mean([Delta_R_leptons_1, Delta_R_leptons_2, Delta_R_leptons_3])
							
							Delta_R_leptonjet_1 = self.Delta_R(L_e_eta, L_e_phi, NL_j_eta, NL_j_phi)
							Delta_R_leptonjet_2 = self.Delta_R(L_e_eta, L_e_phi, L_j_eta, L_j_phi)
							Delta_R_leptonjet_3 = self.Delta_R(L_e_eta, L_e_phi, L_j_eta, L_j_phi)
							Delta_R_leptonjet = np.mean([Delta_R_leptonjet_1, Delta_R_leptonjet_2, Delta_R_leptonjet_3])
													
							dilepton_p_1 = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dilepton_mass_1 = self.Inv_Mass(dilepton_p_1)
							dilepton_p_2 = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
							dilepton_mass_2 = self.Inv_Mass(dilepton_p_2)
							dilepton_p_3 = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)
							dilepton_mass_3 = self.Inv_Mass(dilepton_p_3)
							dilepton_mass = np.mean([dilepton_mass_1, dilepton_mass_2, dilepton_mass_3])
							
							dileptonjet_p_1 = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
							dileptonjet_mass_1 = self.Inv_Mass(dileptonjet_p_1)
							dileptonjet_p_2 = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dileptonjet_mass_2 = self.Inv_Mass(dileptonjet_p_2)
							dileptonjet_p_3 = self.Four_Vec(L_e_eta, L_e_phi, L_e_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dileptonjet_mass_3 = self.Inv_Mass(dileptonjet_p_3)
							dileptonjet_mass = np.mean([dileptonjet_mass_1, dileptonjet_mass_2, dileptonjet_mass_3])
							
							cos_leptons_1 = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, L_j_eta, L_j_phi, L_j_pT)
							cos_leptons_2 = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, NL_j_eta, NL_j_phi, NL_j_pT)
							cos_leptons_3 = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, NNL_j_eta, NNL_j_phi, NNL_j_pT)
							cos_leptons = np.mean([cos_leptons_1, cos_leptons_2, cos_leptons_3])
							
							cos_leptonjet_1 = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, NL_j_eta, NL_j_phi, NL_j_pT)
							cos_leptonjet_2 = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, L_j_eta, L_j_phi, L_j_pT)
							cos_leptonjet_3 = self.cos_theta_star(L_e_eta, L_e_phi, L_e_pT, L_j_eta, L_j_phi, L_j_pT)
							cos_leptonjet = np.mean([cos_leptonjet_1, cos_leptonjet_2, cos_leptonjet_3])
							
							Sph_leptons_1 = self.Sphericity_Tensor(self.Three_Vec(L_e_eta, L_e_phi, L_e_pT)) + \
									     self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT))
							Sph_leptons_2 = self.Sphericity_Tensor(self.Three_Vec(L_e_eta, L_e_phi, L_e_pT)) + \
									     self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT))
							Sph_leptons_3 = self.Sphericity_Tensor(self.Three_Vec(L_e_eta, L_e_phi, L_e_pT)) + \
									     self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT))
							Sph_leptons = (Sph_leptons_1 + Sph_leptons_2 + Sph_leptons_3) / 3
								
						elif (e_Num < p_Num) or ((e_Num == p_Num) and (L_e_pT < L_p_pT)):
							
							# Adjust lepton number
							p_Num += 1
							
							H_T_leptons_1 = L_p_pT + L_j_pT
							H_T_leptons_2 = L_p_pT + NL_j_pT
							H_T_leptons_3 = L_p_pT + NNL_j_pT
							H_T_leptons = np.mean([H_T_leptons_1, H_T_leptons_2, H_T_leptons_3])
							
							Delta_R_leptons_1 = self.Delta_R(L_p_eta, L_p_phi, L_j_eta, L_j_phi)
							Delta_R_leptons_2 = self.Delta_R(L_p_eta, L_p_phi, NL_j_eta, NL_j_phi)
							Delta_R_leptons_3 = self.Delta_R(L_p_eta, L_p_phi, NNL_j_eta, NNL_j_phi)
							Delta_R_leptons = np.mean([Delta_R_leptons_1, Delta_R_leptons_2, Delta_R_leptons_3])
							
							Delta_R_leptonjet_1 = self.Delta_R(L_p_eta, L_p_phi, NL_j_eta, NL_j_phi)
							Delta_R_leptonjet_2 = self.Delta_R(L_p_eta, L_p_phi, L_j_eta, L_j_phi)
							Delta_R_leptonjet_3 = self.Delta_R(L_p_eta, L_p_phi, L_j_eta, L_j_phi)
							Delta_R_leptonjet = np.mean([Delta_R_leptonjet_1, Delta_R_leptonjet_2, Delta_R_leptonjet_3])
													
							dilepton_p_1 = self.Four_Vec(L_p_eta, L_p_phi, L_p_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dilepton_mass_1 = self.Inv_Mass(dilepton_p_1)
							dilepton_p_2 = self.Four_Vec(L_p_eta, L_p_phi, L_p_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
							dilepton_mass_2 = self.Inv_Mass(dilepton_p_2)
							dilepton_p_3 = self.Four_Vec(L_p_eta, L_p_phi, L_p_pT) + self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)
							dilepton_mass_3 = self.Inv_Mass(dilepton_p_3)
							dilepton_mass = np.mean([dilepton_mass_1, dilepton_mass_2, dilepton_mass_3])
							
							dileptonjet_p_1 = self.Four_Vec(L_p_eta, L_p_phi, L_p_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
							dileptonjet_mass_1 = self.Inv_Mass(dileptonjet_p_1)
							dileptonjet_p_2 = self.Four_Vec(L_p_eta, L_p_phi, L_p_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dileptonjet_mass_2 = self.Inv_Mass(dileptonjet_p_2)
							dileptonjet_p_3 = self.Four_Vec(L_p_eta, L_p_phi, L_p_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
							dileptonjet_mass_3 = self.Inv_Mass(dileptonjet_p_3)
							dileptonjet_mass = np.mean([dileptonjet_mass_1, dileptonjet_mass_2, dileptonjet_mass_3])
							
							cos_leptons_1 = self.cos_theta_star(L_p_eta, L_p_phi, L_p_pT, L_j_eta, L_j_phi, L_j_pT)
							cos_leptons_2 = self.cos_theta_star(L_p_eta, L_p_phi, L_p_pT, NL_j_eta, NL_j_phi, NL_j_pT)
							cos_leptons_3 = self.cos_theta_star(L_p_eta, L_p_phi, L_p_pT, NNL_j_eta, NNL_j_phi, NNL_j_pT)
							cos_leptons = np.mean([cos_leptons_1, cos_leptons_2, cos_leptons_3])
							
							cos_leptonjet_1 = self.cos_theta_star(L_p_eta, L_p_phi, L_p_pT, NL_j_eta, NL_j_phi, NL_j_pT)
							cos_leptonjet_2 = self.cos_theta_star(L_p_eta, L_p_phi, L_p_pT, L_j_eta, L_j_phi, L_j_pT)
							cos_leptonjet_3 = self.cos_theta_star(L_p_eta, L_p_phi, L_p_pT, L_j_eta, L_j_phi, L_j_pT)
							cos_leptonjet = np.mean([cos_leptonjet_1, cos_leptonjet_2, cos_leptonjet_3])
							
							Sph_leptons_1 = self.Sphericity_Tensor(self.Three_Vec(L_p_eta, L_p_phi, L_p_pT)) + \
									     self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT))
							Sph_leptons_2 = self.Sphericity_Tensor(self.Three_Vec(L_p_eta, L_p_phi, L_p_pT)) + \
									     self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT))
							Sph_leptons_3 = self.Sphericity_Tensor(self.Three_Vec(L_p_eta, L_p_phi, L_p_pT)) + \
									     self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT))
							Sph_leptons = (Sph_leptons_1 + Sph_leptons_2 + Sph_leptons_3) / 3
					
							
						H_T_jets_1 = NL_j_pT + NNL_j_pT + NNNL_j_pT
						H_T_jets_2 = L_j_pT + NNL_j_pT + NNNL_j_pT
						H_T_jets_3 = L_j_pT + NL_j_pT + NNNL_j_pT
						H_T_jets += np.mean([H_T_jets_1, H_T_jets_2, H_T_jets_3])
						
						Delta_R_jets_1 = self.Delta_R(NL_j_eta, NL_j_phi, NNL_j_eta, NNL_j_phi)
						Delta_R_jets_2 = self.Delta_R(L_j_eta, L_j_phi, NNL_j_eta, NNL_j_phi)
						Delta_R_jets_3 = self.Delta_R(L_j_eta, L_j_phi, NL_j_eta, NL_j_phi)
						Delta_R_jets = np.mean([Delta_R_jets_1, Delta_R_jets_2, Delta_R_jets_3])
						
						dijet_p_1 = self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT) + self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)
						dijet_mass_1 = self.Inv_Mass(dijet_p_1)
						dijet_p_2 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)
						dijet_mass_2 = self.Inv_Mass(dijet_p_2)
						dijet_p_3 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
						dijet_mass_3 = self.Inv_Mass(dijet_p_3)
						dijet_mass = np.mean([dijet_mass_1, dijet_mass_2, dijet_mass_3])
						
						cos_jets_1 = self.cos_theta_star(NL_j_eta, NL_j_phi, NL_j_pT, NNL_j_eta, NNL_j_phi, NNL_j_pT)
						cos_jets_2 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NNL_j_eta, NNL_j_phi, NNL_j_pT)
						cos_jets_3 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NL_j_eta, NL_j_phi, NL_j_pT)
						cos_jets = np.mean([cos_jets_1, cos_jets_2, cos_jets_3])
						
						Sph_jets_1 = self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT)) + \
									  self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)) + \
									  self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_jets_2 = self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_jets_3 = self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_jets += (Sph_jets_1 + Sph_jets_2 + Sph_jets_3) / 3
						
						Sph_global = Sph_leptons + Sph_jets
						
						S_leptons, TS_leptons, AP_leptons, P_leptons = self.Event_Shape_Variables(Sph_leptons)
						S_jets, TS_jets, AP_jets, P_jets = self.Event_Shape_Variables(Sph_jets)
						S_global, TS_global, AP_global, P_global = self.Event_Shape_Variables(Sph_global)					
					
					
					# Double jetfake selection cuts
					if self.Ne_1 == 0 and self.Np_1 == 0 and self.Ne_2 == 0 and self.Np_2 == 0 and self.Nj == 4 and self.Nb == 0:
						
						JF2_Flag = True # Set double jet fake flag
						
						# Adjust number of jets, electrons, and positrons
						j_Num -= 2
						if random() < 0.5:
							e_Num += 2
						else:
							p_Num += 2
						
						
						H_T_leptons_1 = L_j_pT + NL_j_pT
						H_T_leptons_2 = L_j_pT + NNL_j_pT
						H_T_leptons_3 = L_j_pT + NNNL_j_pT
						H_T_leptons_4 = NL_j_pT + NNL_j_pT
						H_T_leptons_5 = NL_j_pT + NNNL_j_pT
						H_T_leptons_6 = NNL_j_pT + NNNL_j_pT
						H_T_leptons = np.mean([H_T_leptons_1, H_T_leptons_2, H_T_leptons_3, H_T_leptons_4, H_T_leptons_5, H_T_leptons_6])
						
						H_T_jets_1 = NNL_j_pT + NNNL_j_pT
						H_T_jets_2 = NL_j_pT + NNNL_j_pT
						H_T_jets_3 = NL_j_pT + NNL_j_pT
						H_T_jets_4 = L_j_pT + NNNL_j_pT
						H_T_jets_5 = L_j_pT + NNL_j_pT
						H_T_jets_6 = L_j_pT + NL_j_pT
						H_T_jets += np.mean([H_T_jets_1, H_T_jets_2, H_T_jets_3, H_T_jets_4, H_T_jets_5, H_T_jets_6])
						
						Delta_R_leptons_1 = self.Delta_R(L_j_eta, L_j_phi, NL_j_eta, NL_j_phi)
						Delta_R_leptons_2 = self.Delta_R(L_j_eta, L_j_phi, NNL_j_eta, NNL_j_phi)
						Delta_R_leptons_3 = self.Delta_R(L_j_eta, L_j_phi, NNNL_j_eta, NNNL_j_phi)
						Delta_R_leptons_4 = self.Delta_R(NL_j_eta, NL_j_phi, NNL_j_eta, NNL_j_phi)
						Delta_R_leptons_5 = self.Delta_R(NL_j_eta, NL_j_phi, NNNL_j_eta, NNNL_j_phi)
						Delta_R_leptons_6 = self.Delta_R(NNL_j_eta, NNL_j_phi, NNNL_j_eta, NNNL_j_phi)
						Delta_R_leptons = np.mean([Delta_R_leptons_1, Delta_R_leptons_2, Delta_R_leptons_3, Delta_R_leptons_4, 
											Delta_R_leptons_5, Delta_R_leptons_6])
											
						Delta_R_jets_1 = self.Delta_R(NNL_j_eta, NNL_j_phi, NNNL_j_eta, NNNL_j_phi)
						Delta_R_jets_2 = self.Delta_R(NL_j_eta, NL_j_phi, NNNL_j_eta, NNNL_j_phi)
						Delta_R_jets_3 = self.Delta_R(NL_j_eta, NL_j_phi, NNL_j_eta, NNL_j_phi)
						Delta_R_jets_4 = self.Delta_R(L_j_eta, L_j_phi, NNNL_j_eta, NNNL_j_phi)
						Delta_R_jets_5 = self.Delta_R(L_j_eta, L_j_phi, NNL_j_eta, NNL_j_phi)
						Delta_R_jets_6 = self.Delta_R(L_j_eta, L_j_phi, NL_j_eta, NL_j_phi)
						Delta_R_jets = np.mean([Delta_R_jets_1, Delta_R_jets_2, Delta_R_jets_3, Delta_R_jets_4, Delta_R_jets_5, 
										  Delta_R_jets_6])						
						
						Delta_R_leptonjet_1 = self.Delta_R(L_j_eta, L_j_phi, NNL_j_eta, NNL_j_phi)
						Delta_R_leptonjet_2 = self.Delta_R(L_j_eta, L_j_phi, NL_j_eta, NL_j_phi)
						Delta_R_leptonjet_3 = self.Delta_R(L_j_eta, L_j_phi, NL_j_eta, NL_j_phi)
						Delta_R_leptonjet_4 = self.Delta_R(NL_j_eta, NL_j_phi, L_j_eta, L_j_phi)
						Delta_R_leptonjet_5 = self.Delta_R(NL_j_eta, NL_j_phi, L_j_eta, L_j_phi)
						Delta_R_leptonjet_6 = self.Delta_R(NNL_j_eta, NNL_j_phi, L_j_eta, L_j_phi)
						Delta_R_leptonjet = np.mean([Delta_R_leptonjet_1, Delta_R_leptonjet_2, Delta_R_leptonjet_3, Delta_R_leptonjet_4, 
											 Delta_R_leptonjet_5, Delta_R_leptonjet_6])

						dilepton_p_1 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
						dilepton_mass_1 = self.Inv_Mass(dilepton_p_1)
						dilepton_p_2 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)
						dilepton_mass_2 = self.Inv_Mass(dilepton_p_2)
						dilepton_p_3 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						dilepton_mass_3 = self.Inv_Mass(dilepton_p_3)
						dilepton_p_4 = self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT) + self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)
						dilepton_mass_4 = self.Inv_Mass(dilepton_p_4)
						dilepton_p_5 = self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT) + self.Four_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						dilepton_mass_5 = self.Inv_Mass(dilepton_p_5)
						dilepton_p_6 = self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT) + self.Four_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						dilepton_mass_6 = self.Inv_Mass(dilepton_p_6)
						dilepton_mass = np.mean([dilepton_mass_1, dilepton_mass_2, dilepton_mass_3, dilepton_mass_4, dilepton_mass_5, 
										   dilepton_mass_6])						
						
						dijet_p_1 = self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT) + self.Four_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						dijet_mass_1 = self.Inv_Mass(dijet_p_1)
						dijet_p_2 = self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT) + self.Four_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						dijet_mass_2 = self.Inv_Mass(dijet_p_2)
						dijet_p_3 = self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT) + self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)
						dijet_mass_3 = self.Inv_Mass(dijet_p_3)
						dijet_p_4 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						dijet_mass_4 = self.Inv_Mass(dijet_p_4)
						dijet_p_5 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)
						dijet_mass_5 = self.Inv_Mass(dijet_p_5)
						dijet_p_6 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
						dijet_mass_6 = self.Inv_Mass(dijet_p_6)
						dijet_mass = np.mean([dijet_mass_1, dijet_mass_2, dijet_mass_3, dijet_mass_4, dijet_mass_5, dijet_mass_6])
						
						dileptonjet_p_1 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)
						dileptonjet_mass_1 = self.Inv_Mass(dileptonjet_p_1)
						dileptonjet_p_2 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
						dileptonjet_mass_2 = self.Inv_Mass(dileptonjet_p_2)
						dileptonjet_p_3 = self.Four_Vec(L_j_eta, L_j_phi, L_j_pT) + self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT)
						dileptonjet_mass_3 = self.Inv_Mass(dileptonjet_p_3)
						dileptonjet_p_4 = self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
						dileptonjet_mass_4 = self.Inv_Mass(dileptonjet_p_4)
						dileptonjet_p_5 = self.Four_Vec(NL_j_eta, NL_j_phi, NL_j_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
						dileptonjet_mass_5 = self.Inv_Mass(dileptonjet_p_5)
						dileptonjet_p_6 = self.Four_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT) + self.Four_Vec(L_j_eta, L_j_phi, L_j_pT)
						dileptonjet_mass_6 = self.Inv_Mass(dileptonjet_p_6)
						dileptonjet_mass = np.mean([dileptonjet_mass_1, dileptonjet_mass_2, dileptonjet_mass_3, dileptonjet_mass_4, 
											dileptonjet_mass_5, dileptonjet_mass_6])		
											
						cos_leptons_1 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NL_j_eta, NL_j_phi, NL_j_pT)
						cos_leptons_2 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NNL_j_eta, NNL_j_phi, NNL_j_pT)
						cos_leptons_3 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						cos_leptons_4 = self.cos_theta_star(NL_j_eta, NL_j_phi, NL_j_pT, NNL_j_eta, NNL_j_phi, NNL_j_pT)
						cos_leptons_5 = self.cos_theta_star(NL_j_eta, NL_j_phi, NL_j_pT, NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						cos_leptons_6 = self.cos_theta_star(NNL_j_eta, NNL_j_phi, NNL_j_pT, NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						cos_leptons = np.mean([cos_leptons_1, cos_leptons_2, cos_leptons_3, cos_leptons_4, cos_leptons_5, cos_leptons_6])
						
						cos_jets_1 = self.cos_theta_star(NNL_j_eta, NNL_j_phi, NNL_j_pT, NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						cos_jets_2 = self.cos_theta_star(NL_j_eta, NL_j_phi, NL_j_pT, NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						cos_jets_3 = self.cos_theta_star(NL_j_eta, NL_j_phi, NL_j_pT, NNL_j_eta, NNL_j_phi, NNL_j_pT)
						cos_jets_4 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NNNL_j_eta, NNNL_j_phi, NNNL_j_pT)
						cos_jets_5 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NNL_j_eta, NNL_j_phi, NNL_j_pT)
						cos_jets_6 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NL_j_eta, NL_j_phi, NL_j_pT)
						cos_jets = np.mean([cos_jets_1, cos_jets_2, cos_jets_3, cos_jets_4, cos_jets_5, cos_jets_6])
						
						cos_leptonjet_1 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NNL_j_eta, NNL_j_phi, NNL_j_pT)
						cos_leptonjet_2 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NL_j_eta, NL_j_phi, NL_j_pT)
						cos_leptonjet_3 = self.cos_theta_star(L_j_eta, L_j_phi, L_j_pT, NL_j_eta, NL_j_phi, NL_j_pT)
						cos_leptonjet_4 = self.cos_theta_star(NL_j_eta, NL_j_phi, NL_j_pT, L_j_eta, L_j_phi, L_j_pT)
						cos_leptonjet_5 = self.cos_theta_star(NL_j_eta, NL_j_phi, NL_j_pT, L_j_eta, L_j_phi, L_j_pT)
						cos_leptonjet_6 = self.cos_theta_star(NNL_j_eta, NNL_j_phi, NNL_j_pT, L_j_eta, L_j_phi, L_j_pT)
						cos_leptonjet = np.mean([cos_leptonjet_1, cos_leptonjet_2, cos_leptonjet_3, cos_leptonjet_4, cos_leptonjet_5, 
										   cos_leptonjet_6])
						
						Sph_leptons_1 = self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								     self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT))
						Sph_leptons_2 = self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								     self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT))
						Sph_leptons_3 = self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								     self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_leptons_4 = self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)) + \
								     self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT))
						Sph_leptons_5 = self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT)) + \
								     self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_leptons_6 = self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)) + \
								     self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_leptons = (Sph_leptons_1 + Sph_leptons_2 + Sph_leptons_3 + Sph_leptons_4 + Sph_leptons_5 + Sph_leptons_6) / 6
						
						Sph_jets_1 = self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_jets_2 = self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_jets_3 = self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT))
						Sph_jets_4 = self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NNNL_j_eta, NNNL_j_phi, NNNL_j_pT))
						Sph_jets_5 = self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NNL_j_eta, NNL_j_phi, NNL_j_pT))
						Sph_jets_6 = self.Sphericity_Tensor(self.Three_Vec(L_j_eta, L_j_phi, L_j_pT)) + \
								  self.Sphericity_Tensor(self.Three_Vec(NL_j_eta, NL_j_phi, NL_j_pT))
						Sph_jets += (Sph_jets_1 + Sph_jets_2 + Sph_jets_3 + Sph_jets_4 + Sph_jets_5 + Sph_jets_6) / 6
						
						Sph_global = Sph_leptons + Sph_jets

						S_leptons, TS_leptons, AP_leptons, P_leptons = self.Event_Shape_Variables(Sph_leptons)						
						S_jets, TS_jets, AP_jets, P_jets = self.Event_Shape_Variables(Sph_jets)
						S_global, TS_global, AP_global, P_global = self.Event_Shape_Variables(Sph_global)		
						

					
					# Append event info to design matrix
					X.append([e_Num, p_Num, j_Num, H_T_leptons, H_T_jets, Delta_R_leptons, Delta_R_jets, Delta_R_leptonjet, dilepton_mass, 
					          dijet_mass, dileptonjet_mass, cos_leptons, cos_jets, cos_leptonjet, MET, S_leptons, TS_leptons, AP_leptons, 
					          P_leptons, S_jets, TS_jets, AP_jets, P_jets, S_global, TS_global, AP_global, P_global])
	        
				# Increment total event count
				Tot_event_counter += 1
	            
				# Re-initialize all features
				e_Num, L_e_pT, NL_e_pT, L_e_eta, L_e_phi, NL_e_eta, NL_e_phi = 0, 0, 0, 0, 0, 0, 0
				p_Num, L_p_pT, NL_p_pT, L_p_eta, L_p_phi, NL_p_eta, NL_p_phi = 0, 0, 0, 0, 0, 0, 0
				j_Num, L_j_pT, L_j_eta, L_j_phi, NL_j_pT, NL_j_eta, NL_j_phi = 0, 0, 0, 0, 0, 0, 0
				NNL_j_pT, NNL_j_eta, NNL_j_phi, NNNL_j_pT, NNNL_j_eta, NNNL_j_phi = 0, 0, 0, 0, 0, 0
				H_T_jets = 0
				b_Num = 0
				MET = 0
				Sph_global = np.zeros((3,3), dtype = 'float')
				Sph_jets = np.zeros((3,3), dtype = 'float')
				Sph_leptons = np.zeros((3,3), dtype = 'float')
	            

				# Increment line number
				line += 1
	     
		
		
		# Define percentage of total events that passed basic selection cuts
		perc = event_counter / Tot_event_counter
		
		# Define weight and cross section
		if CF_Flag:
			
			w = sigma * (CF_prob_sum / event_counter) / Tot_event_counter
			sigma_new = w * event_counter
			
		elif JF1_Flag:
			
			w = sigma * JF1_prob / Tot_event_counter
			sigma_new = w * event_counter
			
		elif JF2_Flag:
			
			w = sigma * JF2_prob / Tot_event_counter
			sigma_new = w * event_counter
			
		else:
			
			w = sigma / Tot_event_counter			
			sigma_new = w * event_counter
	
		# Reset all flags in case the same class instance is used for multiple backgrounds
		CF_Flag, JF1_Flag, JF2_Flag = False, False, False
		
		# Return design matrix, adjusted cross section, cross section weight, and percentage of events that passed selection cuts		
		return X, sigma_new, w, perc



	def Hist_Constructor(self, df, feature, sigma, w, min_val, max_val, bin_size):
		
		# Create a bin structure
		bin_list = np.arange(min_val, max_val + 2 * bin_size, bin_size)
		
		# Initialize a list of aggregate bin weights
		bin_weights = []
		for bin_ in range(len(bin_list) - 1):
			bin_weights.append( len( df[ (df[feature] >= bin_list[bin_]) & (df[feature] <= bin_list[bin_+1]) ] ) * w / sigma  )
		
		# Fill bin structure with weight increments
		bin_list = tuple(itertools.chain(*zip(bin_list, bin_list)))[1:-1]
		bin_weights = tuple(itertools.chain(*zip(bin_weights, bin_weights)))
		
		return bin_list, bin_weights
		
		

if __name__ == '__main__':
	
	# Define some paths
	work_PATH = os.getcwd()
	data_PATH = '/Users/pwinslow/ACFI/Research_Projects/Current_Projects/LNV@100TeV/Analysis_Codes/Data'	
	
	# Define feature list
	Feature_List = ['electron number', 'positron number', 'jet Number', 'H_T(e)', 'H_T(j)', 'Delta_R(L_e, NL_e)', 
	                'Delta_R(L_j, NL_j)', 'Delta_R(L_e, L_j)', 'm(L_e, NL_e)', 'm(L_j, NL_j)', 'm(L_e, L_j)', 
	                'cos(L_e, NL_e)', 'cos(L_j, NL_j)', 'cos(L_e, L_j)', 'MET', 'Sphericity_leptonic', 
	                'Transverse_Sphericity_leptonic', 'Aplanarity_leptonic', 'Planarity_leptonic', 'Sphericity_hadronic', 
	                'Transverse_Sphericity_hadronic', 'Aplanarity_hadronic', 'Planarity_hadronic', 'Sphericity_global', 
	                'Transverse_Sphericity_global', 'Aplanarity_global', 'Planarity_global']
																	
	# Initiate a Wrangler class instance for each background
	DiBoson_Wrangler = Wrangler(Ne_1=2, Np_1=0, Ne_2=0, Np_2=2, Nj=2, Nb=0)
	ChargeFlip_Wrangler = Wrangler(Ne_1=1, Np_1=1, Ne_2=1, Np_2=1, Nj=2, Nb=0)
	OneJetFake_Wrangler = Wrangler(Ne_1=1, Np_1=0, Ne_2=0, Np_2=1, Nj=3, Nb=0)
	TwoJetFake_Wrangler = Wrangler(Ne_1=0, Np_1=0, Ne_2=0, Np_2=0, Nj=4, Nb=0)
	
	# Initiate a Wrangler class instance for the signal 
	Signal_Wrangler = Wrangler(Ne_1=2, Np_1=0, Ne_2=0, Np_2=2, Nj=2, Nb=0)
	
	# Prepare for data imports
	os.chdir(data_PATH)
	
	# Import Diboson backgrounds, creating a dataframe for each one
	print 'Beginning wrangling process...'
	
	# Create dataframe for diboson WW background
	X, sigma_jjWW, w_jjWW, jjWW_fraction = DiBoson_Wrangler.Feature_Architect('full_jjWWBG_lhco_events.dat')
	X_jjWW = DataFrame(np.asarray(X))
	X_jjWW.columns = Feature_List
	X_jjWW['Class'] = 'Background'
	X_jjWW['Subclass'] = 'DiBoson WW'
	print 'Diboson-WW backgrounds wrangled...'
	
	# Create dataframe for diboson WZ background
	X, sigma_jjWZ, w_jjWZ, jjWZ_fraction = DiBoson_Wrangler.Feature_Architect('full_jjWZBG_lhco_events.dat')
	X_jjWZ = DataFrame(np.asarray(X))
	X_jjWZ.columns = Feature_List
	X_jjWZ['Class'] = 'Background'
	X_jjWZ['Subclass'] = 'DiBoson WZ'
	print 'Diboson-WZ backgrounds wrangled...'
	
	# Create dataframe for diboson ZZ background
	X, sigma_jjZZ, w_jjZZ, jjZZ_fraction = DiBoson_Wrangler.Feature_Architect('full_jjZZBG_lhco_events.dat')
	X_jjZZ = DataFrame(np.asarray(X))
	X_jjZZ.columns = Feature_List
	X_jjZZ['Class'] = 'Background'
	X_jjZZ['Subclass'] = 'DiBoson ZZ'
	print 'Diboson-ZZ backgrounds wrangled...'
	
	# Import ChargeFlip backgrounds, creating a dataframe for each one
	
	# Create dataframe for ChargeFlip Zy background
	X, sigma_Zy, w_Zy, Zy_fraction = ChargeFlip_Wrangler.Feature_Architect('full_jjZyCFBG_lhco_events.dat')
	X_Zy = DataFrame(np.asarray(X))
	X_Zy.columns = Feature_List
	X_Zy['Class'] = 'Background'
	X_Zy['Subclass'] = 'ChargeFlip Zy'
	print 'ChargeFlip-Zy backgrounds wrangled...'
	
	# Create dataframe for ChargeFlip ttbar background
	X, sigma_ttCF, w_ttCF, ttCF_fraction = ChargeFlip_Wrangler.Feature_Architect('full_ttbarCFBG_lhco_events.dat')
	X_ttCF = DataFrame(np.asarray(X))
	X_ttCF.columns = Feature_List
	X_ttCF['Class'] = 'Background'
	X_ttCF['Subclass'] = 'ChargeFlip ttbar'
	print 'ChargeFlip-ttbar backgrounds wrangled...'
	
	# Import JetFake backgrounds, creating a dataframe for each one
	
	# Create dataframe for JetFake ttbar background
	X, sigma_ttJF, w_ttJF, ttJF_fraction = OneJetFake_Wrangler.Feature_Architect('full_ttbarJFBG_lhco_events.dat')
	X_ttJF = DataFrame(np.asarray(X))
	X_ttJF.columns = Feature_List
	X_ttJF['Class'] = 'Background'
	X_ttJF['Subclass'] = 'JetFake ttbar'
	print 'JetFake-ttbar backgrounds wrangled...'
	
	# Create dataframe for JetFake W+jets background
	X, sigma_WjJF, w_WjJF, WjJF_fraction = OneJetFake_Wrangler.Feature_Architect('full_WjetsJFBG_lhco_events.dat')
	X_WjJF = DataFrame(np.asarray(X))
	X_WjJF.columns = Feature_List
	X_WjJF['Class'] = 'Background'
	X_WjJF['Subclass'] = 'JetFake Wjets'
	print 'JetFake-W+jets backgrounds wrangled...'
	
	# Create dataframe for JetFake single top background
	X, sigma_tJF, w_tJF, tJF_fraction = OneJetFake_Wrangler.Feature_Architect('full_SingletJFBG_lhco_events.dat')
	X_tJF = DataFrame(np.asarray(X))
	X_tJF.columns = Feature_List
	X_tJF['Class'] = 'Background'
	X_tJF['Subclass'] = 'JetFake Single top'
	print 'JetFake-single t backgrounds wrangled...'
	
	# Create dataframe for JetFake pure QCD background
	X, sigma_QCD, w_QCD, QCD_fraction = TwoJetFake_Wrangler.Feature_Architect('full_4jetJFBG_lhco_events.dat')
	X_QCD = DataFrame(np.asarray(X))
	X_QCD.columns = Feature_List
	X_QCD['Class'] = 'Background'
	X_QCD['Subclass'] = 'JetFake QCD'
	print 'JetFake-QCD backgrounds wrangled...'
	
	
	# Import Signal into a dataframe
	X, sigma_Signal, w_Signal, Signal_fraction = Signal_Wrangler.Feature_Architect('full_Signal_lhco_events.dat')
	X_Signal = DataFrame(np.asarray(X))
	X_Signal.columns = Feature_List
	X_Signal['Class'] = 'Signal'
	X_Signal['Subclass'] = 'Signal'
	print 'Signal backgrounds wrangled...'
	
	
	print 'All backgrounds and signal wrangled. Combining and exporting full dataset to dataframe...' 
	
	# Put all BG events into one dataframe and write to csv
	BG_df = pd.concat([X_jjWW, X_jjWZ, X_jjZZ, X_Zy, X_ttCF, X_ttJF, X_WjJF, X_tJF, X_QCD], axis=0)
	BG_df.to_csv('BGonly_df.csv', index=False)	
	
	# Write all BG cross section and weight info csv in data_PATH
	BGsigma_list = [sigma_jjWW, sigma_jjWZ, sigma_jjZZ, sigma_Zy, sigma_ttCF, sigma_ttJF, sigma_WjJF, sigma_tJF, sigma_QCD]
	BGweight_list = [w_jjWW, w_jjWZ, w_jjZZ, w_Zy, w_ttCF, w_ttJF, w_WjJF, w_tJF, w_QCD]
	BGcs_df = DataFrame(np.concatenate((np.asarray(BGsigma_list).reshape(1,9), np.asarray(BGweight_list).reshape(1,9))), 
				 index = ['cross section (pb)', 'cross section weight (pb)'],
	                  columns=list(Series(BG_df['Subclass'].values.ravel()).unique()))
	BGcs_df.to_csv('BGonly_cross_section_and_weights_df.csv')
	
	# Put all events into one dataframe and write to csv
	Full_df = pd.concat([X_jjWW, X_jjWZ, X_jjZZ, X_Zy, X_ttCF, X_ttJF, X_WjJF, X_tJF, X_QCD, X_Signal], axis=0)
	Full_df.to_csv('Full_df.csv', index=False)
	
	# Write cross section and weight info csv in data_PATH
	sigma_list = [sigma_jjWW, sigma_jjWZ, sigma_jjZZ, sigma_Zy, sigma_ttCF, sigma_ttJF, sigma_WjJF, sigma_tJF, 
	              sigma_QCD, sigma_Signal]
	weight_list = [w_jjWW, w_jjWZ, w_jjZZ, w_Zy, w_ttCF, w_ttJF, w_WjJF, w_tJF, w_QCD, w_Signal]
	cs_df = DataFrame(np.concatenate((np.asarray(sigma_list).reshape(1,10), np.asarray(weight_list).reshape(1,10))), 
				 index = ['cross section (pb)', 'cross section weight (pb)'],
	                  columns=list(Series(Full_df['Subclass'].values.ravel()).unique()))
	cs_df.to_csv('cross_section_and_weights_df.csv')
	
	# Change back to current working directory
	os.chdir(work_PATH)
	
	print 'Done.'
	print 'Wrangled dataframe exported to Data folder with name: {}'.format('Full_df.csv')
	print 'Cross Section and Cross Section Weight dataframe exported to Data folder with name: {}'.format('cross_section_and_weights_df.csv')
