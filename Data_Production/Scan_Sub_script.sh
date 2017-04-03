#/bin/sh

####################################################################################################################################
#                                                                                                                                  #
#  The purpose of this script is to submit multiple parallel runs of Madgraph5+Pythia+Delphes to the titan cluster PBS             #
#  (Portable Batch System) resource manager for the purpose of scanning the model parameter space.                                 #
#                                                                                                                                  #
####################################################################################################################################


# Set some basic paths
workPATH='/data2/pwinslow'
MGbase=$HOME/local/MG5_aMC_v2_4_3
Base=$workPATH/MGjobs

if [ ! -d $Base ]
then
mkdir $Base
fi

# Set some global parameters and define the parameter grid to scan over
pi=$(echo "scale=10; 4*a(1)" | bc -l)
Nc=3
masses=$(awk 'BEGIN{for(i=250; i<=5000; i+=250)print i}')
m_arr=($masses)
couplings=$(awk 'BEGIN{for(i=0.2; i<=2; i+=0.2)print i}')
y_arr=($couplings)

echo "Beginning parameter scan setup..."

# Initialize a parameter point counter and scan over parameter grid
run=1
seed=1
for y in ${y_arr[*]}
do
for m in ${m_arr[*]}
do

	if [ $(echo "$y < 0.4" |bc -l) ] && [ $m -ge 3000 ]
	then

        # Define some useful paths
        runBase=$Base/run$run
	mkdir $runBase

	# Calculate widths of BSM particles
	GammaS=$(echo "scale=10; $m*$Nc*$y^2 / (8 * $pi)" | bc -l)
	xS=$(echo "scale=10; $GammaS^2/($m^2)" | bc -l)
	GammaF=$(echo "scale=10; ( $y^4 * $Nc / (256 * $pi^3 ) ) * $m * (1 - 2 * sqrt( $xS ) * a(1/(sqrt($xS))) + $xS * l(1 + 1/($xS)) )" | bc -l)

	echo "Analyzing parameter point: $run"

	for ((i=1; i<=5; i++))
	do

        # Define some useful paths
	jobBase=$runBase/job$i
        cardBase=$jobBase/MG5_aMC_v2_4_3/SignalScan/Cards

        echo "Setting up run$run|$i..."

        # Copy a MG5 package to jobBase
        mkdir $jobBase
	cp -r $MGbase $jobBase

        # Set MG5 run information
        sed -i "33s/.*/      $seed       = iseed   ! rnd seed (0=assigned automatically=default))/" $cardBase/run_card.dat
        sed -i "32s/.*/  9999991 $m # mSp/" $cardBase/param_card.dat
        sed -i "33s/.*/  9999992 $m # mF0/" $cardBase/param_card.dat
        sed -i "50s/.*/    1 $y # C1/" $cardBase/param_card.dat
        sed -i "51s/.*/    2 $y # C2/" $cardBase/param_card.dat
        sed -i "77s/.*/DECAY 9999991 $GammaS # wSp/" $cardBase/param_card.dat
        sed -i "78s/.*/DECAY 9999992 $GammaF # wF0/" $cardBase/param_card.dat

	# Create the bash script for batch submission
	echo "#/bin/sh" >> $jobBase/MGscript.sh
        echo "#PBS -N Scan$run'_'$i" >> $jobBase/MGscript.sh
        echo "$jobBase/MG5_aMC_v2_4_3/SignalScan/bin/generate_events 0 run$i" >> $jobBase/MGscript.sh

        # Submit run to cluster 
        echo "Submitting run$run|$i..."
        cd $jobBase
        chmod +x MGscript.sh
	qsub -e berr.log -o bout.log MGscript.sh

	seed=$(($seed+1))
        echo "Finished setting up run$run|$i..."
	done

	fi

	# Update parameter point count
	run=$(($run+1))

done
done

echo "Finished parameter scan setup..."
