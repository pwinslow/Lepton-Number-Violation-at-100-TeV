#/bin/sh

####################################################################################################################################
#																   #
#  The purpose of this script is to submit parallel runs of Madgraph5 (for monte carlo calculation of parton level cross sections) #
#  to the titan cluster PBS (Portable Batch System) resource manager.                                   			   #
#																   #
####################################################################################################################################


# Define a few useful PATHs
fdataBase="/data2/pwinslow/"
MGbase=$fdataBase"MG5_aMC_v2_3_3"

# Begin a for loop generating 20 MG5 runs, 10^6 events total
for (( i=1; i<=20; i++ ));
do

# Define a folder to run a MG5 instance in
JobBase=$fdataBase"MGRecord/100TeV_LNV_Results/DiBoson/jjWWBG/job$i"
mkdir $JobBase
cp -r $MGbase $JobBase
cd $JobBase

# Create the bash script for batch submission
#############################################################################
echo "#!/bin/bash" >> script.sh
echo "#PBS -N Run$i" >> script.sh
echo "" >> script.sh

echo "$JobBase/MG5_aMC_v2_3_3/jjWWBG_100TeV/bin/generate_events 0 run$i" >> script.sh
#############################################################################

# Modify the random seed for parton-level generation
sed -i "33s/.*/      $i       = iseed   ! rnd seed (0=assigned automatically=default))/" $JobBase"/MG5_aMC_v2_3_3/jjWWBG_100TeV/Cards/run_card.dat"

# Submit bash script 
chmod +x script.sh
qsub -e berr.log -o bout.log script.sh

# Exit loop
done

