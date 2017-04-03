#!/usr/bin/python

#####################################################################################################################################
#																    #
# The purpose of this script is to collect data generated from parallel computation of parton level cross sections into one 	    #
# dat file. This file should contain the average matched cross section (after PYTHIA) and all events over all runs. The script also #
# creates a repository to store this dat file and all root files for sharing with collaborators.				    #
#																    #
#####################################################################################################################################



# Imports
import os.path
import sys
import re
import numpy as np


# Define the background to collect results for
BG = 'jjWZ'

# Create an event repository for the results
repoBase = '/fdata/hepx/store/user/pwinslow/' + BG + '_Results/'
if os.path.isdir(repoBase) == True:
	sys.exit('Repository already exists...')
else:
	os.system('mkdir ' + repoBase)


# Loop through MG5 run folders and populate the repository with the corresponding pythia log files and delphes root + lhco files
print '\nPopulating event repository...'
for run in range(1,21,1):

	# Define path to run files	
	EventBase = '/fdata/hepx/store/user/pwinslow/MGRecord/100TeV_LNV_Results/DiBoson/' + BG + 'BG/job{0}/MG5_aMC_v2_3_3/'.format(run) + BG + 'BG_100TeV/Events/'

	# Copy relevant files to event repository
	os.system('cp ' + EventBase + 'pythia_output.log' + ' ' + repoBase + 'pythia_output_job{0}.log'.format(run))
	os.system('cp ' + EventBase + 'delphes_events.root' + ' ' + repoBase + 'delphes_events_job{0}.root'.format(run))	
	os.system('cp ' + EventBase + 'delphes_events.lhco' + ' ' + repoBase + 'delphes_events_job{0}.lhco'.format(run))	
print 'Done populating repository.'

# Enter event repository 
os.chdir(repoBase)


# Open a dat file to hold the full set of amalgamated events and averaged matched cross section information
print 'Amalgamating full LHCO events...'
with open('full_' + BG + '_lhco_events.dat', 'w') as full_event_file:

	# Create list to store all matched cross sections
	sigma_list = []

	# Loop through all MG5 run folders and extract the average matched cross section
	for arg in range(1,21,1):

		# Define pythia file
		pythia_file = 'pythia_output_job{0}.log'.format(arg)		

		# Check if pythia file exists
		if os.path.isfile(pythia_file) == False:
			print 'File not found...'

		# Open pythia log file and extract the matched cross section, saving them all to a single list 
		with open(pythia_file, 'r+') as File:
		
			sigma_string = File.readlines()[-1]
			sigma = float(re.findall("-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", sigma_string)[0])
			sigma_list.append(sigma) 

	# Write the average of all the matched cross sections to the dat file
	full_event_file.write('Average matched cross section (pb): {0}\n'.format(np.mean(sigma_list)))
	# Indicate beginning of event info 
	full_event_file.write('Begin event output...\n\n')
	# Include header info for events
	full_event_file.write('   #  typ      eta      phi      pt    jmas   ntrk   btag  had/em   dum1   dum2\n')



	# Loop through all MG5 runs again, this time extracting all events from all delphes event files
	for run in range(1,21,1):

		# Define delphes file
		delphes_file = 'delphes_events_job{0}.lhco'.format(run)		

		# Check if delphes file exists
		if os.path.isfile(delphes_file) == False:
			print 'File not found...'

		# Open delphes file and read in all events
		with open(delphes_file, 'r+') as File:
			delphes_events = File.readlines()

		# While skipping header info, parse all events, printing each event separated by a line with a single 0
		line = 1
		while line < len(delphes_events):

			if float(delphes_events[line].strip().split()[0]) != 0:

				full_event_file.write(delphes_events[line])	
				line += 1
			else:
	
				full_event_file.write('0\n')
				line += 1

# Delete individual leftover lhco files
print 'Cleaning repository...'
os.system('rm *.lhco')

print 'Full LHCO events collected and stored in repository.'
print 'Repository is complete.\n'
