#!/bin/sh

###################################################################################################################
# The purpose of this script is to run the machine learning pipeline on each parameter point of the scan. The	  #
# result is a dataset which assigns a statistical significance to each parameter point.				  #
###################################################################################################################

# Set some basic paths
Base=/data2/pwinslow/100TeV_Scan_Results
BGBase=$Base/Data
cd $Base

if [ -f Scan_Results.csv ]
then
	rm Scan_Results.csv
fi

if [ -f Reports.dat ]
then
	rm Reports.dat
fi

for ((run=1; run<=200; run++))
do

	if [ -f script ]
	then
		rm script
	fi
	
	echo "#/bin/sh" >> script
	echo "#PBS -N Scan$run" >> script
	echo "python $Base/MLpipeline.py $Base/run$run\_Results.lhco $BGBase $run" >> script
	
	chmod +x script
	qsub -e berr$run.log -o bout$run.log script

done
