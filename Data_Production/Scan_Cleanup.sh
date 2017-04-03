#!/bin/sh

#####################################################################################################################################
#  The purpose of this script is to reduce and collect the results of the parameter scan performed by the Scan_Sub_script.sh	    #
#  script. After running, each parameter point should be associated with a single event file.					    #
#####################################################################################################################################

runBase="/data2/pwinslow/MGjobs"
resultsBase="/data2/pwinslow/Results"

numRuns=($runBase/*)
numRuns=${#numRuns[@]}

for (( run=1; run<=200; run++ ))
do
	jobBase="$runBase/run$run"
	numJobs=($jobBase/job*)
	numJobs=${#numJobs[@]}
	cd $jobBase
	
	if [ -e $jobBase/run$run\_Results.lhco ]
	then
		rm run$run\_Results.lhco
	fi
	
	echo "Analyzing run$run..."
	
	avg_sigma=0
	finished_job=0
	for (( job=1; job<=$numJobs; job++ ))
	do
		eventBase="$jobBase/job$job/MG5_aMC_v2_3_3/SignalScan/Events/Run$job"
		
		if [ -e $eventBase/tag_1_delphes_events.lhco.gz ]
		then 
			zcat $eventBase/tag_1_delphes_events.lhco.gz > $eventBase/tag_1_delphes_events.lhco
			
			mass=$(grep "mSp$" $eventBase/tag_1_delphes_events.lhco | grep -oP '^\D*\d+\D*\K\d+')
			coupling=$(grep "C1$" $eventBase/tag_1_delphes_events.lhco | grep -oP '^\D*\d+\D*\K\d+.\d+|^\D*\d+\D*\K\d+')
			
			matched_sigma=$(grep "##  Matched Integrated weight (pb)  :" $eventBase/tag_1_delphes_events.lhco | grep -oP '^\D*\K\d+.\d+')
			matched_sigma=$( echo "scale=10; $matched_sigma" | bc -l)
			avg_sigma=$( echo "scale=10; $avg_sigma + $matched_sigma" | bc -l )
			
			cutoff=$(grep -n "   #  typ      eta      phi      pt    jmas   ntrk   btag  had/em   dum1   dum2" $eventBase/tag_1_delphes_events.lhco | grep -oP '\d{4}')
			if [ -z $cutoff ]
			then
				tail -n +1248 $eventBase/tag_1_delphes_events.lhco >> tmp_events.lhco
				else
				tail -n +$(( $cutoff + 1 )) $eventBase/tag_1_delphes_events.lhco >> tmp_events.lhco
			fi

			finished_job=$(( $finished_job + 1 ))
		fi
	done
	
	if [ $finished_job -ge 4 ]
	then
		avg_sigma=$( echo "scale=10; $avg_sigma/$numJobs" | bc -l )
		echo "Averaged matched cross section: $avg_sigma" >> run$run\_Results.lhco
		echo "mass: $mass" >> run$run\_Results.lhco
		echo "coupling: $coupling" >> run$run\_Results.lhco
		echo >> run$run\_Results.lhco
		cat tmp_events.lhco >> run$run\_Results.lhco
		rm tmp_events.lhco
		cp run$run\_Results.lhco $resultsBase
		echo "Analyzed $finished_job jobs for run$run..."
		else
		echo "Not enough completed runs..."
	fi
done
