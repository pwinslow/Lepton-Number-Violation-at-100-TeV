#/bin/sh

#######################################################################################################################################
#																      #
# The purpose of this script is to 1) run PYTHIA on existing parton level events to simulate parton showering with matching effects   #
# accounted for and 2) to run the delphes detector simulator using our own specialized detector card simulating a O(100 TeV) detector #  
# environment.															      #
# 																      #
#######################################################################################################################################



# Define Delphes PATH
DelphesBase="/data2/pwinslow/Delphes-3.3.2"


# Loop over all jobs, running PYTHIA and then Delphes on each one
for (( i=1; i<=20; i++ ));
do

EventBase="/data2/pwinslow/MGRecord/100TeV_LNV_Results/DiBoson/jjZZBG/job$i/MG5_aMC_v2_3_3/jjZZBG_100TeV/Events"
cd $EventBase

# If the job has finished running, cp and unzip unweighted events for future PYTHIA use
if [ "$(ls -A $EventBase)" ]; 
then
if [ ! -f unweighted_events.lhe ];
then
cp run$i/unweighted_events.lhe.gz .
gunzip unweighted_events.lhe.gz 
fi
else
echo "job$i hasn't finished running yet..."
fi

# Create PYTHIA card using default values
if [ ! -f ../Cards/pythia_card.dat ];
then
cp ../Cards/pythia_card_default.dat ../Cards/pythia_card.dat
fi

# Run PYTHIA
if [ ! -f pythia_output.log ];
then
../bin/internal/run_pythia ../../pythia-pgs/src/ 0 run$i >> pythia_output.log
fi

# Run Delphes
if [ ! -f delphes_output.log ];
then
$DelphesBase/DelphesSTDHEP $DelphesBase/cards/PW_delphes_FCC.tcl $EventBase/delphes_events.root $EventBase/pythia_events.hep >> delphes_output.log
fi

# Convert root file to lhco format
if [ ! -f root2lhco_output.log ];
then
$DelphesBase/root2lhco $EventBase/delphes_events.root $EventBase/delphes_events.lhco >> root2lhco_output.log
fi

done
